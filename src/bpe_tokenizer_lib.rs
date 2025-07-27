// BYTE PAIR TOKENIZER SECTION
use core::{fmt};
use std::{collections::HashSet, hash::Hash, cmp};
use hashbrown::HashMap; 
use ahash::AHasher;   
use std::hash::BuildHasherDefault;
use rayon::{prelude::*}; 
use serde::{Serialize, Deserialize};
use crate::text_normalizer::TextNormalizer;

extern crate serde;
extern crate bincode;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use regex::Regex;
use lazy_static::lazy_static;

// <=========== Regex for Pre-tokenization ============>
lazy_static! {
    static ref PRE_TOKENIZER_RE: Regex = Regex::new(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+").unwrap();
}

// <=========== Thread Local Storage for Tokenizer Cache ============>

thread_local! {
    static TOKENIZER_CACHE: std::cell::RefCell<HashMap<usize, BPETokenizer>> =  std::cell::RefCell::new(HashMap::new());
}

// Error handling for batch encoding
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Encoding failed: {0}")]
    EncodingError(String),
    #[error("Text too long: {len} characters, max allowed: {max}")]
    TextTooLong { len: usize, max: usize },
    #[error("Lock poisoned")]
    LockPoisoned,
    #[error("Normalization failed: {0}")]
    NormalizationError(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

pub type TokenizedResult<T> = Result<T, TokenizerError>;
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct TokenPair(pub u32, pub u32);

impl TokenPair {
    pub fn display(&self, tokenizer : &BPETokenizer) -> String {
        format!("(\"{}\", \"{}\")", 
            tokenizer.get_token_from_id(self.0), 
            tokenizer.get_token_from_id(self.1))
    }
}

impl fmt::Display for TokenPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(\"{}\", \"{}\")", self.0, self.1)
    }
}

// Configuration for batch encoding
pub struct BatchEncodingConfig{
    pub max_length: Option<usize>, // Maximum length of the encoded sequence
    pub parallel_threshold: usize, // Maximum number of threads to use (None for automatic)
    pub max_threads: Option<usize>, // Maximum number of threads to use (None for automatic)
    pub use_thread_local_cache: bool, // Whether to use thread-local caching
    pub thread_cache_size: usize, // Cache size per thread
}   

impl Default for BatchEncodingConfig{
    fn default() -> Self {
        Self {
            max_length: Some(1_000_000), // 1 MB limit by default
            parallel_threshold: 10, // Use parallel processing for large batches
            max_threads: None, // Use all available threads by default
            use_thread_local_cache: true, // Enable thread-local caching by default
            thread_cache_size: 50_000, // Default cache size per thread
        }
    }
}

// <=========== Incremental Tokenizer ============>
#[derive(Debug, Clone)]
pub struct IncrementalTrainingConfig{
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub chunk_size: usize,          // Number of documents to process at once
    pub merge_frequency: usize,     // How often to perform merges (for every n chunks)
    pub save_frequency: usize,      // How often to save checkpoint (for every n chunks)
    pub special_tokens: Vec<String>,
    pub show_progress: bool,
    pub n_threads: Option<usize>,
    pub checkpoint_path: Option<String>,
}

impl Default for IncrementalTrainingConfig{
    fn default() -> Self {
        Self {
            vocab_size: 50_000,
            min_frequency: 2,
            chunk_size: 1000,
            merge_frequency: 10,
            save_frequency: 100,
            special_tokens: vec![
                "<pad>".to_string(),
                "<unk>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
                "<mask>".to_string(),
                "<cls>".to_string(),
                "</w>".to_string(),
            ],
            show_progress: true,
            n_threads: Some(num_cpus::get()),
            checkpoint_path: Some("bpe_checkpoint.json".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalTrainingState {
    #[serde(serialize_with = "utils::non_string_map_serialization::serialize")]
    #[serde(deserialize_with = "utils::non_string_map_serialization::deserialize")] 
    pub word_segments: FastHashMap<Vec<u32>, usize>,

    #[serde(skip_serializing, default)] 
    pub pair_freq: FastHashMap<TokenPair, usize>, // Frequency of token pairs
    pub chunks_processed: usize,
    pub total_documents: usize,
    pub current_vocab_size: usize,
    pub last_merge_iteration: usize,
}

impl IncrementalTrainingState{
    pub fn new() -> Self {
        Self {
            word_segments: FastHashMap::default(),
            pair_freq: FastHashMap::default(),
            chunks_processed: 0,
            total_documents: 0,
            current_vocab_size: 0,
            last_merge_iteration: 0,
        }
    }

    
    pub fn reconstruct_pair_freq(&mut self, tokenizer : &BPETokenizer){
        self.pair_freq.clear();

        for (segment_ids, count) in &self.word_segments {
            if segment_ids.len() < 2 {
                continue;
            }
            
            for window in segment_ids.windows(2) {
                let pair = TokenPair(window[0], window[1]);
                *self.pair_freq.entry(pair).or_insert(0) += count;
            }
        }
    }

    // Cleanup methods for low-frequency and long segments 
    pub fn cleanup_low_frequency_segments(&mut self, min_frequency: usize) {
        println!("🧹 Cleaning up segments with frequency < {}", min_frequency);
        let initial_count = self.word_segments.len();
        
        self.word_segments.retain(|_, &mut count| count >= min_frequency);
        
        let cleaned_count = initial_count - self.word_segments.len();
        println!("Removed {} low-frequency segments", cleaned_count);
    }

    pub fn cleanup_long_segments(&mut self, max_length: usize) {
        println!("🧹 Cleaning up segments longer than {} tokens", max_length);
        let initial_count = self.word_segments.len();
        
        self.word_segments.retain(|segment, _| segment.len() <= max_length);
        
        let cleaned_count = initial_count - self.word_segments.len();
        println!("Removed {} overly long segments", cleaned_count);
    }

    pub fn aggressive_cleanup(&mut self, frequency_threshold: usize, length_threshold: usize) {
        println!("🧹 Performing aggressive cleanup...");
        let initial_segments = self.word_segments.len();
        let initial_pairs = self.pair_freq.len();
        
        // Clean up segments
        self.cleanup_low_frequency_segments(frequency_threshold);
        self.cleanup_long_segments(length_threshold);
        
        // Clear pair frequencies (will be reconstructed when needed)
        self.pair_freq.clear();
        
        println!("Cleanup complete: segments {} -> {}, pairs {} -> {}", 
                initial_segments, self.word_segments.len(), 
                initial_pairs, self.pair_freq.len());
    }

    pub fn memory_usage_stats(&self) -> (usize, usize) {
        let segments_mem = self.word_segments.len() * (
            std::mem::size_of::<Vec<u32>>() + 
            std::mem::size_of::<usize>() + 
            32 // average segment length estimate
        );
        
        let pairs_mem = self.pair_freq.len() * (
            std::mem::size_of::<TokenPair>() + 
            std::mem::size_of::<usize>()
        );
        
        (segments_mem, pairs_mem)
    }
}



// <=========== UTILITY TYPES ============>
pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;

pub mod utils {

    pub mod non_string_map_serialization {
        use serde::{Serialize, Serializer, Deserialize, Deserializer};
        use hashbrown::HashMap; 
        use std::hash::{BuildHasher, Hash};

        pub fn serialize<S, K, V, H>(map: &HashMap<K, V, H>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
            K: Serialize + Eq + Hash,
            V: Serialize,
            H: BuildHasher,
        {
            let vec: Vec<(&K, &V)> = map.iter().collect();
            vec.serialize(serializer)
        }

        pub fn deserialize<'de, D, K, V, H>(deserializer: D) -> Result<HashMap<K, V, H>, D::Error>
        where
            D: Deserializer<'de>, K: Deserialize<'de> + Eq + Hash, V: Deserialize<'de>, H: BuildHasher + Default,
        {
            let vec: Vec<(K, V)> = Vec::deserialize(deserializer)?;
            Ok(vec.into_iter().collect())
        }
    }

    // --- Cache Submodule ---
    pub mod cache {
        use hashbrown::HashMap;
        use std::collections::VecDeque;
        use serde::{Deserialize, Serialize}; // For serialization

        // LruCache implements a Least Recently Used cache for storing encoded results.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(bound(
            serialize = "K: Eq + std::hash::Hash + Serialize + Clone + std::fmt::Debug, V: Serialize + Clone",
            deserialize = "K: Eq + std::hash::Hash + for<'kde> Deserialize<'kde> + Clone + std::fmt::Debug, V: for<'vde> Deserialize<'vde> + Clone"
        ))]
        pub struct LruCache<K, V>
        where
            K: Eq + std::hash::Hash + Serialize + for<'kde> Deserialize<'kde> + Clone + std::fmt::Debug,
            V: Serialize + for<'vde> Deserialize<'vde> + Clone,
        {
            capacity: usize,
            map: HashMap<K, V>,
            order: VecDeque<K>, // Stores keys in order of usage (front is most recent)
        }

        impl<K, V> Default for LruCache<K, V>
        where
            K: Clone + std::hash::Hash + Eq + std::fmt::Debug + Serialize + for<'kde> Deserialize<'kde>,
            V: Clone + Serialize + for<'vde> Deserialize<'vde>,
        {
            fn default() -> Self {
                Self::new(50_000) // Or some other default capacity
            }
        }

        impl<K, V> LruCache<K, V> 
        where
            K: Clone + std::hash::Hash + Eq + std::fmt::Debug + Serialize + for<'kde> Deserialize<'kde>,
            V: Clone + Serialize + for<'vde> Deserialize<'vde>,
        {
            pub fn new(capacity: usize) -> Self {
                Self {
                    capacity,
                    map: HashMap::with_capacity(capacity),
                    order: VecDeque::with_capacity(capacity),
                }
            }

            pub fn get(&mut self, key: &K) -> Option<&V> {
                if self.map.contains_key(key) {
                    self.order.retain(|k| k != key);
                    self.order.push_front(key.clone());
                    self.map.get(key)
                } else {
                    None
                }
            }

            pub fn put(&mut self, key: K, value: V) {
                if self.map.contains_key(&key) {
                    self.order.retain(|k| k != &key);
                } else if self.map.len() >= self.capacity {
                    if let Some(old_key) = self.order.pop_back() {
                        self.map.remove(&old_key);
                    }
                }
                self.order.push_front(key.clone());
                self.map.insert(key, value);
            }
        }
    }
}

// --- Training Configuration Struct ---
// Holds parameters for the BPE training process.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub vocab_size: usize,      // Target size of the final vocabulary.
    pub min_frequency: usize,   // Minimum frequency for a pair to be considered (not directly used in this simplified BPE, but good to have).
    pub special_tokens: Vec<String>, // List of special tokens to include in the vocabulary.
    pub show_progress: bool,    // Whether to print training progress.
    pub n_threads: Option<usize>,       // Number of threads to use for parallel processing.
}

// --- Default Implementation for TrainingConfig ---
// Provides default values for the BPE training configuration.

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            min_frequency: 2,
            special_tokens: vec![
                "<pad>".to_string(),
                "<unk>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
                "<mask>".to_string(),
                "<cls>".to_string(),
                "</w>".to_string(),
            ],
            show_progress: true,
            n_threads: Some(num_cpus::get()), // To detect number of available CPU cores
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    pub vocab: FastHashMap<String, u32>,
    #[serde(serialize_with = "utils::non_string_map_serialization::serialize")]
    #[serde(deserialize_with = "utils::non_string_map_serialization::deserialize")]
    pub reverse_vocab: FastHashMap<u32, String>, // Reverse mapping for decoding
    pub merges: Vec<(TokenPair, u32)>, // List of merges applied during training
    pub unknown_tokens: String,
    pub special_tokens_ids: FastHashMap<String, u32>,
    pub end_of_word_token: String,

    // cache for encoded results to avoid recomputing
    // This cache stores encoded results for previously seen texts to speed up repeated encodings.
    #[serde(skip)]
    pub cache: utils::cache::LruCache<String, Vec<u32>>,
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}



impl BPETokenizer {
    pub fn new() -> BPETokenizer {
        let config = TrainingConfig::default();

        // Default for unknown tokens
        let unk_token_value = config
            .special_tokens
            .iter()
            .find(|&s| s == "<unk>")
            .cloned()
            .unwrap_or_else(|| "<unk>".to_string());
        
        // end of word token default
        let eow_token_value = config
            .special_tokens
            .iter()
            .find(|&s| s == "⁄")
            .cloned()
            .unwrap_or_else(|| "⁄".to_string());

        // Vocab and ID tracking
        let mut vocab = FastHashMap::default();
        let mut next_id = 0;

        // Add default characters to vocab

        // Lowercase
        for ch in 'a'..='z' {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }

        // Uppercase
        for ch in 'A'..='Z' {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }

        // Digits
        for ch in '0'..='9' {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }

        // Punctuation / special characters
        let special_chars = [
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '.', ',', ':', ';', '-',
            '?', '/', '<', '>', '[', ']', '{', '}', '|', '\\', '`', '~', '"', '\'', ' ', '\t', '\n', '¢', 
            '£', '€', '¥', '₹', '©', '®', '™', '°', '±', 'µ', '÷', '×', '§', '¶', '•', '–', '—', '‘', '’',
            '“', '”', '«', '»', '‹', '›', '¡', '¿', '‽', '…', '‰', '′', '″', '℅', 'ℓ', '∂', '∑', '∏', '√',
            '∫', '∮', '≠', '≈', '≡', '≤', '≥', '≪', '≫', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∅', '∠', '∟',
            '⊥', '∥', '∝', '∞', '∃', '∀', '∇', '∈', '∉', '⊕', '⊗', '⋅', '⋆', '⋈', '⋉', '⋊', '⋋', '⋌',
            '⋍', '⋎', '⋏', '⋐', '⋑', '⋒', '⋓', '⋔', '\u{00A0}', '\u{200B}', '\u{202F}', '±', '×', '÷',
            '√', '∛', '∜', '∐', '∮', '∯', '∰', '∝', '∋', '∌', '∧', '∨', '⊘', '⊙', '∴', '∵', '≬', '≻', '≺',
            '―', '⁃', '‗', '‖', '‴', 'ⁱ', 'ⁿ', 'ℵ', 'ℶ', 'ℷ', 'ℸ', 'ℹ', 'ℼ', 'ℽ', 'ℾ', 'ℿ', '⅀', '⅁', '⅂', '⅃', '⅄',
            '⅊', '⅋', '⅌', '⅍', 'ⅎ', '⅏', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ',
            'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'ς',
            'Μ', 'Ν', 'Ξ','Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω', '\u{200C}', '\u{200D}',  '\u{2060}',  '\u{FEFF}',  '\u{180E}',
        ];
        for &ch in &special_chars {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }

        vocab.insert(eow_token_value.clone(), next_id);
        next_id += 1;

        // Reserve space for reverse_vocab and special_tokens_ids
        let mut reverse_vocab = FastHashMap::default();
        let mut special_tokens_ids = FastHashMap::default();

        // Assign IDs to special tokens (example: <unk>, </w>) but first — override if they already exist
        for token_str in &config.special_tokens {
            if !vocab.contains_key(token_str) {
                vocab.insert(token_str.clone(), next_id);
                special_tokens_ids.insert(token_str.clone(), next_id);
                next_id += 1;
            } else {
                let id = *vocab.get(token_str).unwrap();
                special_tokens_ids.insert(token_str.clone(), id);
            }
        }

        BPETokenizer {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            end_of_word_token: eow_token_value,
            unknown_tokens: unk_token_value,
            special_tokens_ids,
            cache: utils::cache::LruCache::new(150_000),
        }
    }



// The `unk_ids` function returns the ID associated with unknown tokens from a map of special token
// IDs.
//
// Returns:
// 
// The `unk_ids` function is returning a `u32` value. This value is obtained by looking up the
// `unknown_tokens` key in the `special_tokens_ids` map and returning the corresponding value. If the
// key is not found in the map, the function will panic with the message "UNKNOWN TOKENS MUST BE
// INITIALIZED IN SPECIAL TOKEN IDS".
    fn unk_ids(&self) -> u32 {
        *self.special_tokens_ids
            .get(&self.unknown_tokens)
            .expect("UNKNOWN TOKENS MUST BE INITIALIZED IN SPECIAL TOKEN IDS")
    }


    pub fn merge_pair_inplace(segments: &mut Vec<u32>, pair: &TokenPair, new_token_id: u32) {
        let mut write_idx = 0;
        let mut read_idx = 0;

        while read_idx < segments.len() {
            if read_idx + 1 < segments.len() 
                && segments[read_idx] == pair.0 
                && segments[read_idx + 1] == pair.1 {
                // Found pair to merge
                segments[write_idx] = new_token_id;
                write_idx += 1;
                read_idx += 2; // Skip both tokens of the pair
            } else {
                // Keep token as is
                if write_idx != read_idx {
                    segments[write_idx] = segments[read_idx];
                }
                write_idx += 1;
                read_idx += 1;
            }
        }

        // Truncate to remove unused elements
        segments.truncate(write_idx);
    }

    // Check if word contains the pair before processing
    fn _word_contains_pair(segments: &[u32], pair: &TokenPair) -> bool {
        segments.windows(2).any(|window| window[0] == pair.0 && window[1] == pair.1)
    }

    fn apply_merge_to_corpus_optimized(
        corpus_segments_map: &mut FastHashMap<Vec<u32>, usize>,
        best_pair_ids: TokenPair,
        new_id: u32,
        reverse_vocab: &FastHashMap<u32, String>, // Use reverse_vocab instead of id_to_token
        _vocab: &FastHashMap<String, u32>, // Keeping vocab as read-only reference because I want to
        _eow_token: &String,
    ) {
        let mut new_corpus_segments_map: FastHashMap<Vec<u32>, usize> = FastHashMap::default();

        let p1_str = reverse_vocab.get(&best_pair_ids.0).expect("ID not in vocab").clone();
        let p2_str = reverse_vocab.get(&best_pair_ids.1).expect("ID not in vocab").clone();
        let _new_token_str = format!("{}{}", p1_str, p2_str); 

        for (segment_ids, count) in corpus_segments_map.drain() {
            let mut new_segment_ids = Vec::new();
            let mut i = 0;
            
            while i < segment_ids.len() {
                if i + 1 < segment_ids.len() 
                    && segment_ids[i] == best_pair_ids.0 
                    && segment_ids[i + 1] == best_pair_ids.1 {
                    // Found the pair to merge
                    new_segment_ids.push(new_id);
                    i += 2; // Skip both tokens of the pair
                } else {
                    // Keep token as is
                    new_segment_ids.push(segment_ids[i]);
                    i += 1;
                }
            }
            
            *new_corpus_segments_map.entry(new_segment_ids).or_insert(0) += count;
        }
        
        *corpus_segments_map = new_corpus_segments_map;
    }

    // train function : use when your dataset is present in the local filesystem and your RAM can handle it.
    // Example usage: Tokenizing a 4 GB text file on 16 GB RAM
    pub fn train(&mut self, corpus: &[String], config : TrainingConfig) {
        println!("Starting BPE Training Sequence in Rust WHOOHOOO!!!");
        println!("Corpus size: {} documents", corpus.len());
        println!("Number of merges to perform: {}", config.vocab_size.saturating_sub(self.vocab.len())); // Merges until vocab_size
        println!("Using {:?} threads for training.", config.n_threads);

        // Initializing a thread pool for parallel processing
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.n_threads.unwrap_or(num_cpus::get()))
            .build_global()
            .unwrap();

        // Pre-process Corpus into initial character segments + EOW token
        let mut word_segments: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
        let mut initial_vocabulary_chars: HashSet<String> = HashSet::new();

        let normalizer = TextNormalizer::new()
                        .to_strip_accents();

        // Use a temporary map for parallel collection to merge later
        let initial_processing_results: Vec<(FastHashMap<Vec<u32>, usize>, HashSet<String>)> = corpus.par_iter().map(|doc| {
            let normalized_doc = normalizer.normalize(doc);

            let words: Vec<String> = PRE_TOKENIZER_RE.find_iter(&normalized_doc)
                .map(|m| m.as_str().to_string())
                .collect();

            let mut local_segments_map: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
            let mut local_initial_chars: HashSet<String> = HashSet::new();

            for word in words {
                let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                chars.push(self.end_of_word_token.clone());
                
                for char_token in word.chars() {
                    local_initial_chars.insert(char_token.to_string());
                }

                let char_ids: Vec<u32> = chars.iter()
                .map(|s| {
                    if let Some(&id) = self.vocab.get(s) {
                        id
                    } else {
                        println!("Unknown character found: '{}' (Unicode: U+{:04X})", s, s.chars().next().unwrap() as u32);
                        self.unk_ids()
                    }
                })
                .collect();
                *local_segments_map.entry(char_ids).or_insert(0) += 1;
            }
            (local_segments_map, local_initial_chars)
        }).collect();

        // Combine results from all threads into the main word_segments map
        // This is done to avoid contention on the main map during parallel processing.
        // Each thread processes its own segment of the corpus and returns a local map.
        for (local_segments_map, local_initial_chars) in initial_processing_results {
            for (local_segments_ids, count) in local_segments_map {
                *word_segments.entry(local_segments_ids).or_insert(0) += count;
            }
            initial_vocabulary_chars.extend(local_initial_chars);
        }

        // Initializing the vocabulary with special tokens and initial characters
        // We ensure that all special tokens are present in the vocabulary.
        let mut current_ids = self.vocab.len() as u32;
        for char_token in &initial_vocabulary_chars {
            if !self.vocab.contains_key(char_token) {
                self.vocab.insert(char_token.clone(), current_ids);
                self.reverse_vocab.insert(current_ids, char_token.clone());
                current_ids += 1;
            }
        }

        for (token, &id) in &self.vocab {
            self.reverse_vocab.entry(id).or_insert_with(|| token.clone());
        }   

        println!("Initial vocabulary size: {}", self.vocab.len());

        let mut pair_freq : FastHashMap<TokenPair, usize> = FastHashMap::default();
        pair_freq.reserve(100000); // Reserve space for pairs to avoid reallocations
        
        while self.vocab.len() < config.vocab_size{
            pair_freq.clear(); // Clear pair frequency map for each merge iteration
            
            let global_pair_counts = std::sync::Mutex::new(FastHashMap::default());
            let word_segments_vec : Vec<_> = word_segments.iter().collect(); // Convert to Vec for parallel processing
            // Parallel processing of word segments to count token pairs
            
            word_segments_vec.par_iter().for_each(|(segments_ids, count)| {
                if segments_ids.len() < 2 {
                    return;
                }
                let mut local_pair_counts: FastHashMap<TokenPair, usize> = FastHashMap::default();
                for window in segments_ids.windows(2) {
                    let pair = TokenPair(window[0], window[1]);
                    *local_pair_counts.entry(pair).or_insert(0) += *count;
                }
                let mut global = global_pair_counts.lock().unwrap();
                for (pair, local_count) in local_pair_counts {
                    *global.entry(pair).or_insert(0) += local_count;
                }
            });
            pair_freq = global_pair_counts.into_inner().unwrap();

            let Some((best_pair, max_freq)) = pair_freq
            .iter()
            .max_by_key(|&(_,freq)| freq)
            .map(|(pair, &freq)| (pair.clone(), freq))
            else{
                println!("No more existing valid pair found :(");
                break;
            };

            if max_freq <= config.min_frequency {
                println!("Stopping early: Max pair frequency ({}) is below min_frequency ({}).", max_freq, config.min_frequency);
                break;
            }

            
            let token1_str = self.get_token_from_id(best_pair.0).clone();
            let token2_str = self.get_token_from_id(best_pair.1).clone();
            let new_token_str = format!("{}{}", token1_str, token2_str);
            
            if self.vocab.contains_key(&new_token_str) {
                continue;
            }

            let new_id = current_ids;
            // Insert the new token into the vocabulary
            self.vocab.insert(new_token_str.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token_str.clone());
            current_ids += 1;
            self.merges.push((TokenPair(best_pair.0.clone(), best_pair.1.clone()), new_id));

            if config.show_progress {
                println!(
                    "Merge {}: Merging (\"{}\" + \"{}\") with freq {}. New token: \"{}\" (Vocab Size: {})",
                    self.merges.len(),
                    token1_str,
                    token2_str,
                    max_freq,
                    new_token_str,
                    self.vocab.len()
                );
                
            }

            let best_pair_ids = best_pair;

            Self::apply_merge_to_corpus_optimized(
                &mut word_segments,
                best_pair_ids,
                new_id,
                &self.reverse_vocab,  // Pass reverse_vocab instead of mutable reference
                &self.vocab,          // Pass vocab as read-only reference
                &self.end_of_word_token,
            );

            
        }

        println!("--- BPE Training Finished ---");
        println!("Final Vocabulary size: {}", self.vocab.len());
        println!("Total Merges performed: {}", self.merges.len());
    }


    pub fn incremental_stream_train<I>(&mut self, corpus: I, config: IncrementalTrainingConfig, mut state : &mut IncrementalTrainingState) 
    where 
        I: Iterator<Item = String> 
    {
        println!("🚀 Starting Incremental BPE Training");
        println!("Chunk size: {}, Merge frequency: {}", config.chunk_size, config.merge_frequency);
        
        let mut chunk_buffer = Vec::with_capacity(config.chunk_size);
        let normalizer = TextNormalizer::new().to_strip_accents();
        
        // Initialize vocabulary with base characters
        self.initialize_base_vocabulary(&config);
        state.current_vocab_size = self.vocab.len();
        
        for (doc_idx, document) in corpus.enumerate() {
            chunk_buffer.push(document);
            
            // Process chunk when buffer is full
            if chunk_buffer.len() >= config.chunk_size {
                self.process_chunk_incremental(&chunk_buffer, &mut state, &normalizer, &config);
                state.chunks_processed += 1;
                state.total_documents += chunk_buffer.len();
                
                // Memory management: Clean up every 50 chunks
                if state.chunks_processed % 50 == 0 {
                    println!("🧹 Performing periodic cleanup at chunk {}", state.chunks_processed);
                    state.aggressive_cleanup(2, 100); // Remove segments seen < 2 times, longer than 100 tokens
                }
                
                // Perform merges periodically (but not immediately)
                if state.chunks_processed >= 3 && state.chunks_processed % config.merge_frequency == 0 {
                    self.perform_incremental_merges(&mut state, &config);
                }
                
                chunk_buffer.clear();
                
                if config.show_progress {
                    let (seg_mem, pair_mem) = state.memory_usage_stats();
                    println!("Processed {} chunks, {} documents, vocab size: {}, memory: {:.1}MB", 
                            state.chunks_processed, state.total_documents, self.vocab.len(),
                            (seg_mem + pair_mem) as f64 / 1_000_000.0);
                }

                if self.vocab.len() >= config.vocab_size {
                    println!("✅ Target vocab size {} reached at chunk {}. Stopping training.", config.vocab_size, state.chunks_processed);
                    break;
                }

            }
        }
        
        // Process remaining documents
        if self.vocab.len() >= config.vocab_size {
            println!("⚠️ Skipping leftover chunk: vocab size already reached.");
        } else if !chunk_buffer.is_empty() {
            self.process_chunk_incremental(&chunk_buffer, &mut state, &normalizer, &config);
            state.chunks_processed += 1;
            state.total_documents += chunk_buffer.len();
        }

        
        // Final cleanup and merge pass
        state.aggressive_cleanup(1, 200); // More lenient for final pass
        self.perform_incremental_merges(&mut state, &config);
        
        println!("Training completed!");
        println!("Total documents: {}, Final vocab size: {}", state.total_documents, self.vocab.len());
    }

    
    // Process a single chunk and update word segments
    pub fn process_chunk_incremental(
        &mut self,
        chunk: &[String],
        state: &mut IncrementalTrainingState,
        normalizer: &TextNormalizer,
        config: &IncrementalTrainingConfig,
    ) {
        let chunk_results: Vec<(FastHashMap<Vec<u32>, usize>, HashSet<String>)> = chunk.par_iter()
            .map(|doc| {
                let normalized_doc = normalizer.normalize(doc);
                let words: Vec<String> = PRE_TOKENIZER_RE.find_iter(&normalized_doc)
                    .map(|m| m.as_str().to_string())
                    .collect();
                
                let mut local_segments: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
                let mut local_new_chars: HashSet<String> = HashSet::new();

                for word in words {
                    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                    chars.push(self.end_of_word_token.clone());

                    // Track new chars (to insert into vocab later)
                    for ch in &chars {
                        local_new_chars.insert(ch.clone());
                    }

                    let char_ids: Vec<u32> = chars.iter()
                        .map(|s| *self.vocab.get(s).unwrap_or(&self.unk_ids()))
                        .collect();

                    *local_segments.entry(char_ids).or_insert(0) += 1;
                }

                (local_segments, local_new_chars)
            })
            .collect();

        // === SEQUENTIAL merge ===

        for (local_segments, local_new_chars) in chunk_results {
            // Merge segments
            for (segment_ids, count) in local_segments {
                *state.word_segments.entry(segment_ids).or_insert(0) += count;
            }

            // Insert new chars into vocab
            for ch in local_new_chars {
                if !self.vocab.contains_key(&ch) {
                    let new_id = self.vocab.len() as u32;
                    self.vocab.insert(ch.clone(), new_id);
                    self.reverse_vocab.insert(new_id, ch);
                }
            }
        }
    }

    
    // Perform merges on accumulated data
    pub fn perform_incremental_merges(
        &mut self,
        state: &mut IncrementalTrainingState,
        config: &IncrementalTrainingConfig,
    ) {
        // Check if the tokenizer has enough segments to perform merges
        if self.vocab.len() >= config.vocab_size {
            return;
        }

        
        
        println!("Performing incremental merges...");
        
        // Calculate pair frequencies using all CPU cores because I have paid for the whole cpu and I will use all its power!
        // Normal comment: Calculate pair frequencies from current word segments
        state.pair_freq.clear();
        
        // let word_segments_vec: Vec<_> = state.word_segments.iter().collect();
        let pair_frequencies: FastHashMap<TokenPair, usize> = state.word_segments
            .par_iter()
            .map(|(segment_tokens, &count)| {
                let mut local_freq = FastHashMap::default();
                for window in segment_tokens.windows(2) {
                    let mut pair = TokenPair(window[0], window[1]);
                    
                    *local_freq.entry(pair).or_insert(0) += count;
                }
                local_freq
            })
            .reduce(FastHashMap::default, |mut map1, map2| {
                for (pair, count) in map2 {
                    *map1.entry(pair).or_insert(0) += count;
                }
                map1
            });
        // Merge the pair frequencies from all chunks
        state.pair_freq = pair_frequencies;

        // Perform merges until vocabulary target or no more valid pairs
        let mut merges_this_round = 0;
        while self.vocab.len() < config.vocab_size && merges_this_round < 1000 {
            let Some((best_pair, max_freq)) = state.pair_freq
                .par_iter()
                .max_by_key(|&(_, freq)| freq)
                .map(|(pair, &freq)| (pair.clone(), freq))
            else {
                break;
            };
            
            if max_freq < config.min_frequency {
                break;
            }

            let token1_str = self.get_token_from_id(best_pair.0);
            let token2_str = self.get_token_from_id(best_pair.1);
            let new_token_str = format!("{}{}", token1_str, token2_str);
            if self.vocab.contains_key(&new_token_str) {
                state.pair_freq.remove(&best_pair);
                continue;
            }
            
            // Add new merged token
            let new_id = self.vocab.len() as u32;
            self.vocab.insert(new_token_str.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token_str.clone());
            self.merges.push((TokenPair(best_pair.0.clone(), best_pair.1.clone()), new_id));
            
            if config.show_progress {
                println!("  Merge {}: {} + {} → '{}' (freq: {})",
                        self.merges.len(), best_pair.0, best_pair.1, new_token_str, max_freq);
            }
            
            // Apply merge to word segments
            let best_pair_ids = best_pair;
            
            Self::apply_merge_to_corpus_optimized(
                &mut state.word_segments,
                best_pair_ids,
                new_id,
                &self.reverse_vocab,
                &self.vocab,
                &self.end_of_word_token,
            );
            
            // Remove the merged pair and recalculate affected pairs
            state.pair_freq.remove(&best_pair);
            self.update_pair_freq_after_merge(&mut state.pair_freq, &best_pair, &mut state.word_segments, new_id);
            
            merges_this_round += 1;
        }
        
        state.last_merge_iteration = state.chunks_processed;
        println!("Completed {} merges this round", merges_this_round);
    }
    
    // Update pair frequencies after a merge
    pub fn update_pair_freq_after_merge(
        &self,
        pair_freq: &mut FastHashMap<TokenPair, usize>,
        merged_pair: &TokenPair,
        word_segments: &mut FastHashMap<Vec<u32>, usize>, // Made mutable to update in place
        new_token_id: u32,
    ) {
        let pair_to_find = (merged_pair.0, merged_pair.1);

        // Temporary map for new segments to avoid modifying while iterating
        let mut new_segments_to_add = FastHashMap::default();
        let mut old_segments_to_remove = Vec::new();

        // --- Main Logic: A single pass over all word segments ---
        for (segment_ids, &count) in word_segments.iter() {
            // Find segments that contain the pair we just merged
            let has_pair = segment_ids.windows(2).any(|w| w[0] == pair_to_find.0 && w[1] == pair_to_find.1);

            if has_pair {
                
                // 1. Decrement frequencies for all old pairs in this segment
                if segment_ids.len() >= 2 {
                    for window in segment_ids.windows(2) {
                        let pair_str = TokenPair(window[0], window[1]);
                        if let Some(freq) = pair_freq.get_mut(&pair_str) {
                            *freq = freq.saturating_sub(count);
                        }
                    }
                }
                
                // 2. Create the new segment by applying the merge
                let mut new_segment_ids = Vec::with_capacity(segment_ids.len() - 1);
                let mut i = 0;
                while i < segment_ids.len() {
                    if i + 1 < segment_ids.len() && segment_ids[i] == pair_to_find.0 && segment_ids[i+1] == pair_to_find.1 {
                        new_segment_ids.push(new_token_id);
                        i += 2; // Skip both parts of the merged pair
                    } else {
                        new_segment_ids.push(segment_ids[i]);
                        i += 1;
                    }
                }

                // 3. Increment frequencies for all new pairs in the new segment
                if new_segment_ids.len() >= 2 {
                    for window in new_segment_ids.windows(2) {
                        let pair_str = TokenPair(window[0], window[1]);
                        *pair_freq.entry(pair_str).or_insert(0) += count;
                    }
                }

                // 4. Stage the changes to the word_segments map
                old_segments_to_remove.push(segment_ids.clone());
                *new_segments_to_add.entry(new_segment_ids).or_insert(0) += count;
            }
        }

        // --- Finalize: Apply the changes ---
        for seg in old_segments_to_remove {
            word_segments.remove(&seg);
        }
        for (seg, count) in new_segments_to_add {
            *word_segments.entry(seg).or_insert(0) += count;
        }
        
        // Finally, remove the pair that we just merged from the frequency map
        pair_freq.remove(merged_pair);
    }
    
    // Initialize base vocabulary with common characters
    pub fn initialize_base_vocabulary(&mut self, _config: &IncrementalTrainingConfig) {
        // This should already be done in new(), but ensure it's complete
        for (token, &id) in &self.vocab {
            self.reverse_vocab.entry(id).or_insert_with(|| token.clone());
        }
    }

    pub fn save_checkpoint(&self, path: &str, state: IncrementalTrainingState, format : &mut &str) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Serialize)]
        struct Checkpoint<'a> {
            tokenizer: &'a BPETokenizer,
            state: &'a IncrementalTrainingState,
        }

        let checkpoint = Checkpoint {
            tokenizer: self,
            state: &state,
        };

        let format = format.to_lowercase();

        // Create a temporary file to write the checkpoint
        // This is to ensure atomic writes and avoid corruption in case of errors.
        let temp_path = format!("{}.tmp", path);
        {
            let file = File::create(&temp_path)?;
            let writer = BufWriter::new(file);

            match format.as_str() {
                "json" => {
                    serde_json::to_writer_pretty(writer, &checkpoint)?;
                },
                "bincode" => {
                    bincode::serialize_into(writer, &checkpoint)?;
                },
                _ => {
                    return Err(Box::new(TokenizerError::UnsupportedFormat(format)));
                }
            }
        }

        // Atomic rename to replace the old checkpoint file
        // This ensures that if the write fails, the original file remains intact.
        std::fs::rename(&temp_path, path)?;
        Ok(())
        
    }
    
    // Load checkpoint for resuming incremental training
    pub fn load_checkpoint(path: &str) -> Result<(Self, IncrementalTrainingState), Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct Checkpoint {
            tokenizer: BPETokenizer,
            state: IncrementalTrainingState,
        }
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Check file extension to determine format
        let checkpoint: Checkpoint = if path.ends_with(".json") {
            serde_json::from_reader(reader)?
        } else {
            // Assume bincode for any other extension
            bincode::deserialize_from(reader)?
        };
        
        Ok((checkpoint.tokenizer, checkpoint.state))
    }

    pub fn get_token_from_id(&self, id: u32) -> &String {
        debug_assert_eq!(self.vocab.len(), self.reverse_vocab.len(), "Mismatch between vocab and reverse_vocab!");

        match self.reverse_vocab.get(&id) {
            Some(token) => token,
            None => {
                eprintln!(
                    "[PANIC DEBUG] Tried to get token for ID {}, but it's not in reverse_vocab.",
                    id
                );
                eprintln!("reverse_vocab keys: {:?}", self.reverse_vocab.keys());
                panic!("ID not found in vocab");
            }
        }
    }


    pub fn get_id_to_token_map(&self) -> FastHashMap<u32, String> {
        self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect()
    }


    fn _get_word_char_segments(&self, word: &str) -> Vec<String> {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push(self.end_of_word_token.clone());
        chars
    }

    pub fn get_initial_word_char_segments(&self, word : &str) -> Vec<String>{
        let mut chars : Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push(self.end_of_word_token.clone());
        chars
    }

    // Encodes a single text string into a sequence of token IDs.
    // Handles Caching to avoid recomputing results for the same text.
    // This method uses the pre-tokenizer regex to split the text into segments,
    // normalizes the text, and then encodes each segment into token IDs.
    pub fn encode(&mut self, text: &str) -> TokenizedResult<Vec<u32>> {
        // Check cache first
        if let Some(cached_result) = self.cache.get(&text.to_string()) {
            return Ok(cached_result.clone()); 
        }

        let mut encoded_ids: Vec<u32> = Vec::new();
        
        // Text normalization
        let normalizer = TextNormalizer::new()
            .to_lowercase()
            .to_strip_accents();
        let normalized_text = normalizer.normalize(text);

        // Pre-tokenization
        let pre_tokens = PRE_TOKENIZER_RE.find_iter(&normalized_text)
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();

        // Process each word segment
        for word_segments_str in pre_tokens {
            let char_tokens: Vec<String> = self.get_initial_word_char_segments(word_segments_str);

            // Convert initial characters to token IDs
            let mut current_word_ids: Vec<u32> = char_tokens.iter()
                .map(|s| *self.vocab.get(s).unwrap_or(&self.unk_ids()))
                .collect();

            // Apply BPE merges
            loop {
                let mut best_merge: Option<(&TokenPair, &u32)> = None;
                let mut best_merge_pos = usize::MAX;

                // Find the best merge (earliest position in merge list)
                for (pair, new_id) in &self.merges {
                    for i in 0..current_word_ids.len().saturating_sub(1) {
                        if current_word_ids[i] == pair.0 && current_word_ids[i + 1] == pair.1 {
                            // Found an applicable merge at the earliest position
                            if i < best_merge_pos {
                                best_merge = Some((pair, new_id));
                                best_merge_pos = i;
                            }
                            break; // Move to next pair
                        }
                    }
                }

                // Apply the best merge if found
                if let Some((best_pair_to_merge, new_id)) = best_merge {
                    Self::merge_pair_inplace(&mut current_word_ids, best_pair_to_merge, *new_id);
                } else {
                    // No more merges can be applied
                    break;
                }
            }

            // Add processed word IDs to final result
            encoded_ids.extend(current_word_ids);
        }

        // Cache the result
        self.cache.put(text.to_string(), encoded_ids.clone());
        Ok(encoded_ids)
    }

    // Encodes a batch of text strings into sequences of token IDs.
    // This method allows for parallel processing of multiple texts, utilizing a thread pool if configured.
    pub fn encode_batch(&mut self, text: &[String], config : BatchEncodingConfig) -> TokenizedResult<Vec<Vec<u32>>>{
    if let Some(max_threads) = config.max_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
            .map_err(|e| TokenizerError::EncodingError(e.to_string()))?;

        pool.install(|| {
            thread_local! {
                static TOKENIZER_CACHE: std::cell::RefCell<Option<BPETokenizer>> =
                    std::cell::RefCell::new(None);
            }

            let chunk_size = cmp::max(1, text.len() / config.parallel_threshold);
            // Split the text into chunks for parallel processing

            text.par_chunks(chunk_size)
            .map(|chunk| {
                let mut tokenizer_to_use = if config.use_thread_local_cache {
                    TOKENIZER_CACHE.with(|cache| {
                        let mut cache_ref = cache.borrow_mut();
                        if cache_ref.is_none() {
                            let mut new_tokenizer = self.clone();
                            new_tokenizer.cache = utils::cache::LruCache::new(config.thread_cache_size);
                            *cache_ref = Some(new_tokenizer);
                        }
                        cache_ref.as_mut().unwrap().clone() 
                    })
                } else {
                    // If no thread-local cache, simply clone for each chunk/task
                    self.clone()
                };

                // Process chunk sequentially within thread
                chunk.iter()
                    .map(|text| tokenizer_to_use.encode(text))
                    .collect::<TokenizedResult<Vec<Vec<u32>>>>()
            })
            .collect::<TokenizedResult<Vec<Vec<Vec<u32>>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
        })
    } else {
        // Fallback if no max_threads is set
        let chunk_size = cmp::max(1, text.len() / config.parallel_threshold);

        text.par_chunks(chunk_size)
        .map(|chunk| {
            let mut tokenizer_to_use = if config.use_thread_local_cache {
                // May implement thread-local cache here as well
                self.clone()
            } else {
                self.clone()
            };

            chunk.iter()
                .map(|text| tokenizer_to_use.encode(text))
                .collect::<TokenizedResult<Vec<Vec<u32>>>>()
        })
        .collect::<TokenizedResult<Vec<Vec<Vec<u32>>>>>()
        .map(|chunks| chunks.into_iter().flatten().collect())
    }
    }

    // decodes a sequence of token IDs back into a string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut decoded_parts = Vec::new();

        for &id in token_ids {
            let token_str = self.reverse_vocab
                .get(&id)
                .cloned()
                .unwrap_or_else(|| self.unknown_tokens.clone());

            // Skip special tokens like <PAD>, <MASK>, etc.
            if self.special_tokens_ids.contains_key(&token_str)
                && token_str != self.unknown_tokens
                && token_str != self.end_of_word_token
            {
                continue;
            }

            decoded_parts.push(token_str);
        }

        // Join everything
        let joined = decoded_parts.join("");

        // Normalize EOW token if present
        let clean_text = joined
            .replace(&self.end_of_word_token, " ") // e.g., "</w>"
            .replace("⁄", " ")                    // just in case any ⁄ slipped in
            .replace('\u{2044}', " ");            // backup for unicode literal

        clean_text.trim().to_string()
    }


    // Save the vocab and merges to a JSON file
    pub fn save(&self, path: &str ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    // Load a BPETokenizer Vocab from a JSON file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut tokenizer: Self = serde_json::from_reader(reader)?; // Deserialize into a new BPETokenizer
        
        // Re-initialize transient fields after deserialization
        tokenizer.cache = utils::cache::LruCache::new(10000); // Re-create cache
        
        Ok(tokenizer)
    }

    // method to get vocabulary statistics
    pub fn get_stats(&self) -> BPEStats {
        let num_character_tokens = self.vocab.len()
                                .saturating_sub(self.special_tokens_ids.len())
                                .saturating_sub(self.merges.len());
        BPEStats {
            vocab_size: self.vocab.len(),
            num_merges: self.merges.len(),
            num_special_tokens: self.special_tokens_ids.len(),
            num_character_tokens,
        }
    }
}

// statistics struct for better introspection
#[derive(Debug)]
pub struct BPEStats {
    pub vocab_size: usize,
    pub num_merges: usize,
    pub num_special_tokens: usize,
    pub num_character_tokens: usize,
}

impl fmt::Display for BPEStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BPE Statistics:\n  Total Vocabulary: {}\n  Special Tokens: {}\n  Character Tokens: {}\n  Merged Tokens: {}",
            self.vocab_size, self.num_special_tokens, self.num_character_tokens, self.num_merges
        )
    }
}

// BYTE PAIR TOKENIZER END UwU
