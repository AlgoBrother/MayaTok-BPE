// BYTE PAIR TOKENIZER SECTION

use core::fmt;
use std::{collections::HashSet, hash::Hash, cmp};

use hashbrown::HashMap;
use ahash::AHasher;
use std::hash::BuildHasherDefault;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use crate::bpe_tokenizer_lib::utils::cache::LruCache;
use crate::text_normalizer::TextNormalizer;

extern crate serde;
extern crate bincode;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use regex::Regex;
use lazy_static::lazy_static;

// =========== Regex for Pre-tokenization ============
lazy_static! {
    static ref PRE_TOKENIZER_RE: Regex =
        Regex::new(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+").unwrap();
}

// =========== Error handling ============
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

// =========== Utility types ============
pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct TokenPair(pub u32, pub u32);

impl TokenPair {
    pub fn display(&self, tokenizer: &BPETokenizer) -> String {
        format!(
            "(\"{}\", \"{}\")",
            tokenizer.get_token_from_id(self.0),
            tokenizer.get_token_from_id(self.1)
        )
    }
}

impl fmt::Display for TokenPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

// =========== LRU Cache ============
// This implementation uses a HashMap of nodes forming an intrusive doubly-linked
// list via indices. Both get() and put() are O(1).
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
            D: Deserializer<'de>,
            K: Deserialize<'de> + Eq + Hash,
            V: Deserialize<'de>,
            H: BuildHasher + Default,
        {
            let vec: Vec<(K, V)> = Vec::deserialize(deserializer)?;
            Ok(vec.into_iter().collect())
        }
    }

    pub mod cache {
        use std::borrow::Borrow;

        use hashbrown::HashMap;
        use serde::{Deserialize, Serialize};

        // Node in the doubly-linked list. Indices are into the `nodes` Vec.
        // usize::MAX is used as a null sentinel.
        #[derive(Clone, Serialize, Deserialize, Debug)]
        struct Node<K, V> {
            key: K,
            value: V,
            prev: usize,
            next: usize,
        }

        /// O(1) LRU cache backed by a HashMap + index-linked list.
        #[derive(Clone, Serialize, Deserialize, Debug)]
        #[serde(bound(
            deserialize = "K: Deserialize<'de>, V: Deserialize<'de>",
            serialize = "K: Serialize, V: Serialize"
        ))]
        pub struct LruCache<K, V>
        where
            K: Eq + std::hash::Hash + Clone + std::fmt::Debug,
            V: Clone,
        {
            capacity: usize,
            map: HashMap<K, usize>,
            nodes: Vec<Node<K, V>>,
            head: usize,
            tail: usize,
            free: Vec<usize>,
        }

        impl<K, V> Default for LruCache<K, V>
        where
            K: Eq + std::hash::Hash + Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>,
            V: Clone + Serialize + for<'de> Deserialize<'de>,
        {
            fn default() -> Self { Self::new(50_000) }
        }

        impl<K, V> LruCache<K, V>
        where
            K: Eq + std::hash::Hash + Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>,
            V: Clone + Serialize + for<'de> Deserialize<'de>,
        {
            pub fn new(capacity: usize) -> Self {
                Self {
                    capacity,
                    map: HashMap::with_capacity(capacity),
                    nodes: Vec::with_capacity(capacity),
                    head: usize::MAX,
                    tail: usize::MAX,
                    free: Vec::new(),
                }
            }

            // Detach a node from wherever it is in the list
            fn detach(&mut self, idx: usize) {
                let prev = self.nodes[idx].prev;
                let next = self.nodes[idx].next;
                if prev != usize::MAX { self.nodes[prev].next = next; }
                else { self.head = next; }
                if next != usize::MAX { self.nodes[next].prev = prev; }
                else { self.tail = prev; }
                self.nodes[idx].prev = usize::MAX;
                self.nodes[idx].next = usize::MAX;
            }

            // Attach a node at the head (most-recently-used end)
            fn attach_head(&mut self, idx: usize) {
                self.nodes[idx].prev = usize::MAX;
                self.nodes[idx].next = self.head;
                if self.head != usize::MAX {
                    self.nodes[self.head].prev = idx;
                }
                self.head = idx;
                if self.tail == usize::MAX {
                    self.tail = idx;
                }
            }

            pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
            where
                K: Borrow<Q>,
                Q: std::hash::Hash + Eq + ?Sized,
            {
                let idx = *self.map.get(key)?; // ? early return if not found
                self.detach(idx); // move to head (most recently used)
                self.attach_head(idx); // reattach at head
                Some(&self.nodes[idx].value) // return value
            }

            pub fn put(&mut self, key: K, value: V) {
                if let Some(&idx) = self.map.get(&key) {
                    self.nodes[idx].value = value;
                    self.detach(idx);
                    self.attach_head(idx);
                    return;
                }

                // Evict LRU if at capacity
                let idx = if self.map.len() >= self.capacity {
                    let evict_idx = self.tail;
                    self.detach(evict_idx);
                    let old_key = self.nodes[evict_idx].key.clone();
                    self.map.remove(&old_key);
                    evict_idx
                } else {
                    let idx = self.free.pop().unwrap_or_else(|| {
                        let i = self.nodes.len();
                        // push a placeholder; will be overwritten below
                        self.nodes.push(Node {
                            key: key.clone(),
                            value: value.clone(),
                            prev: usize::MAX,
                            next: usize::MAX,
                        });
                        i
                    });
                    idx
                };

                self.nodes[idx] = Node { key: key.clone(), value, prev: usize::MAX, next: usize::MAX };
                self.map.insert(key, idx);
                self.attach_head(idx);
            }
        }
    }
}

// =========== Training configs ============

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub special_tokens: Vec<String>,
    pub show_progress: bool,
    pub n_threads: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_000,
            min_frequency: 2,
            special_tokens: vec![
                "<pad>".to_string(), "<unk>".to_string(), "<s>".to_string(),
                "</s>".to_string(), "<mask>".to_string(), "<cls>".to_string(),
                "</w>".to_string(),
            ],
            show_progress: true,
            n_threads: Some(num_cpus::get()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchEncodingConfig {
    pub max_length: Option<usize>,
    /// Minimum number of texts per chunk before going parallel.
    pub parallel_threshold: usize,
    pub max_threads: Option<usize>,
    pub use_thread_local_cache: bool,
    pub thread_cache_size: usize,
}

impl Default for BatchEncodingConfig {
    fn default() -> Self {
        Self {
            max_length: Some(1_000_000),
            parallel_threshold: 32,
            max_threads: None,
            use_thread_local_cache: true,
            thread_cache_size: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IncrementalTrainingConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub chunk_size: usize,
    pub merge_frequency: usize,
    pub save_frequency: usize,
    pub special_tokens: Vec<String>,
    pub show_progress: bool,
    pub n_threads: Option<usize>,
    pub checkpoint_path: Option<String>,
}

impl Default for IncrementalTrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_000,
            min_frequency: 2,
            chunk_size: 1000,
            merge_frequency: 10,
            save_frequency: 100,
            special_tokens: vec![
                "<pad>".to_string(), "<unk>".to_string(), "<s>".to_string(),
                "</s>".to_string(), "<mask>".to_string(), "<cls>".to_string(),
                "</w>".to_string(),
            ],
            show_progress: true,
            n_threads: Some(num_cpus::get()),
            checkpoint_path: Some("bpe_checkpoint.json".to_string()),
        }
    }
}

// =========== Incremental training state ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalTrainingState {
    #[serde(serialize_with = "utils::non_string_map_serialization::serialize")]
    #[serde(deserialize_with = "utils::non_string_map_serialization::deserialize")]
    pub word_segments: FastHashMap<Vec<u32>, usize>,
    #[serde(skip_serializing, default)]
    pub pair_freq: FastHashMap<TokenPair, usize>,
    pub chunks_processed: usize,
    pub total_documents: usize,
    pub current_vocab_size: usize,
    pub last_merge_iteration: usize,
}

impl IncrementalTrainingState {
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

    pub fn reconstruct_pair_freq(&mut self) {
        self.pair_freq.clear();
        for (segment_ids, count) in &self.word_segments {
            if segment_ids.len() < 2 { continue; }
            for window in segment_ids.windows(2) {
                let pair = TokenPair(window[0], window[1]);
                *self.pair_freq.entry(pair).or_insert(0) += count;
            }
        }
    }

    pub fn cleanup_low_frequency_segments(&mut self, min_frequency: usize) {
        self.word_segments.retain(|_, &mut count| count >= min_frequency);
    }

    pub fn cleanup_long_segments(&mut self, max_length: usize) {
        self.word_segments.retain(|segment, _| segment.len() <= max_length);
    }

    pub fn aggressive_cleanup(&mut self, frequency_threshold: usize, length_threshold: usize) {
        self.cleanup_low_frequency_segments(frequency_threshold);
        self.cleanup_long_segments(length_threshold);
        self.pair_freq.clear();
    }
}

// ========== Encoding BLOCK ============

// I AM TIRED OF COPY PASTING TWO FUNCTIONS IN TOKENIZER AND THREAD
// HAVE THIS ONE HERE. REST I WILL SEE IT. IF IT WORKS NOW, IT WORKS
// I WILL SEE DESYNC LATER IF IT HAPPENS
fn encode_text_inner(
    text: &str,
    vocab: &FastHashMap<String, u32>,
    merge_rank: &FastHashMap<TokenPair, usize>,
    merge_id: &FastHashMap<TokenPair, u32>,
    special_tokens_ids: &FastHashMap<String, u32>,
    end_of_word_token: &str,
    unk_id: u32,
    cache: &mut LruCache<String, Arc<Vec<u32>>>,
) -> TokenizedResult<Vec<u32>> {
    // Special token check
    if let Some(&id) = special_tokens_ids.get(text) {
        return Ok(vec![id]);
    }

    // Cache check
    if let Some(cached) = cache.get(text) {
        return Ok((**cached).clone());
    }

    let normalizer = TextNormalizer::new().to_strip_accents();
    let normalized = normalizer.normalize(text);
    let mut ids: Vec<u32> = Vec::new();

    for word in PRE_TOKENIZER_RE.find_iter(&normalized).map(|m| m.as_str()) {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push(end_of_word_token.to_string());

        let mut word_ids: Vec<u32> = chars.iter()
            .map(|s| *vocab.get(s).unwrap_or(&unk_id))
            .collect();

        loop {
            if word_ids.len() < 2 { break; }
            let best = word_ids.windows(2).enumerate().filter_map(|(pos, w)| {
                let pair = TokenPair(w[0], w[1]);
                merge_rank.get(&pair).map(|&rank| (rank, pos, pair))
            }).min_by_key(|&(rank, _, _)| rank);

            match best {
                None => break,
                Some((_, pos, pair)) => {
                    let new_id = merge_id[&pair];
                    word_ids[pos] = new_id;
                    let mut write = pos + 1;
                    let mut read = pos + 2;
                    while read < word_ids.len() {
                        word_ids[write] = word_ids[read];
                        write += 1;
                        read += 1;
                    }
                    word_ids.truncate(write);
                }
            }
        }
        ids.extend(word_ids);
    }

    cache.put(text.to_string(), Arc::new(ids.clone()));
    Ok(ids)
}

#[derive(Debug, Clone)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,  // 1 = real token, 0 = padding
    pub token_type_ids: Vec<u32>,  // 0 = sentence A, 1 = sentence B
}

impl Encoding{

    // build encoding from ids with options for padding, truncation, and special tokens
    pub fn build_with_pair(
        mut ids_a: Vec<u32>,
        ids_b: Option<Vec<u32>>,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        special_tokens_ids: &FastHashMap<String, u32>,
    ) -> Self {
        // Add special tokens
        if add_special_tokens {
            if let Some(&bos) = special_tokens_ids.get("<s>") {
                ids_a.insert(0, bos);
            }
            if let Some(&eos) = special_tokens_ids.get("</s>") {
                ids_a.push(eos);
            }
        }

        let len_a = ids_a.len();

        // Combine with text_b if present
        let ids_b_processed = match ids_b {
            Some(mut b) => {
                if add_special_tokens {
                    if let Some(&eos) = special_tokens_ids.get("</s>") {
                        b.push(eos);
                    }
                }
                b
            }
            None => vec![],
        };

        let len_b = ids_b_processed.len();

        let mut ids = ids_a;
        ids.extend(ids_b_processed);

        // Truncation — trim from longer sequence
        if truncation {
            if let Some(max) = max_length {
                ids.truncate(max);
            }
        }

        let real_len = ids.len();

        // Padding
        if padding {
            if let Some(max) = max_length {
                let pad_id = *special_tokens_ids.get("<pad>").unwrap_or(&0);
                ids.resize(max, pad_id);
            }
        }

        let seq_len = ids.len();

        // token_type_ids: 0 for A, 1 for B, 0 for padding
        let token_type_ids: Vec<u32> = (0..seq_len)
            .map(|i| {
                if i < len_a { 0 }
                else if i < len_a + len_b { 1 }
                else { 0 }
            })
            .collect();

        let attention_mask: Vec<u32> = (0..seq_len)
            .map(|i| if i < real_len { 1 } else { 0 })
            .collect();

        Encoding { input_ids: ids, attention_mask, token_type_ids }
    }

    pub fn build(
        ids: Vec<u32>,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        special_tokens_ids: &FastHashMap<String, u32>,
    ) -> Self {
        Self::build_with_pair(ids, None, max_length, padding, truncation, add_special_tokens, special_tokens_ids)
    }

    

    // function to build encoding from ids with default config (no truncation, no padding, no special tokens)
    pub fn single(ids: Vec<u32>) -> Self {
        let len = ids.len();
        Encoding {
            attention_mask: vec![1u32; len],
            token_type_ids: vec![0u32; len],
            input_ids: ids,
        }
    }
}


// =========== Core tokenizer struct ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    pub vocab: FastHashMap<String, u32>,
    #[serde(serialize_with = "utils::non_string_map_serialization::serialize")]
    #[serde(deserialize_with = "utils::non_string_map_serialization::deserialize")]
    pub reverse_vocab: FastHashMap<u32, String>,
    pub merges: Vec<(TokenPair, u32)>,
    pub unknown_tokens: String,
    pub special_tokens_ids: FastHashMap<String, u32>,
    pub end_of_word_token: String,

    #[serde(skip)]
    pub merge_rank: FastHashMap<TokenPair, usize>, // pair -> priority (lower = higher priority)
    #[serde(skip)]
    pub merge_id: FastHashMap<TokenPair, u32>,     // pair -> resulting token id

    #[serde(skip)]
    pub cache: utils::cache::LruCache<String, Arc<Vec<u32>>>,
}

impl Default for BPETokenizer {
    fn default() -> Self { Self::new() }
}

impl BPETokenizer {
    pub fn new() -> Self {
        let config = TrainingConfig::default();

        let unk_token = "<unk>".to_string();
        let eow_token = "⁄".to_string();

        let mut vocab: FastHashMap<String, u32> = FastHashMap::default();
        let mut next_id: u32 = 0;

        // Seed vocabulary — identical set to original
        for ch in 'a'..='z' { vocab.insert(ch.to_string(), next_id); next_id += 1; }
        for ch in 'A'..='Z' { vocab.insert(ch.to_string(), next_id); next_id += 1; }
        for ch in '0'..='9' { vocab.insert(ch.to_string(), next_id); next_id += 1; }

        let special_chars = [
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '.', ',', ':', ';',
            '?', '/', '<', '>', '[', ']', '{', '}', '|', '\\', '`', '~', '"', '\'', ' ', '\t', '\n',
            '¢', '£', '€', '¥', '₹', '©', '®', '™', '°', '±', 'µ', '÷', '×', '§', '¶', '•', '–', '—',
            '\u{00A0}', '\u{200B}', '\u{202F}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}', '\u{180E}',
        ];
        for &ch in &special_chars {
            vocab.entry(ch.to_string()).or_insert_with(|| { let id = next_id; next_id += 1; id });
        }

        vocab.entry(eow_token.clone()).or_insert_with(|| { let id = next_id; next_id += 1; id });

        let mut special_tokens_ids: FastHashMap<String, u32> = FastHashMap::default();
        for token in &config.special_tokens {
            let id = vocab.entry(token.clone()).or_insert_with(|| {
                let id = next_id; next_id += 1; id
            });
            special_tokens_ids.insert(token.clone(), *id);
        }

        let mut reverse_vocab: FastHashMap<u32, String> = FastHashMap::default();
        for (token, &id) in &vocab {
            reverse_vocab.insert(id, token.clone());
        }

        BPETokenizer {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            end_of_word_token: eow_token,
            unknown_tokens: unk_token,
            special_tokens_ids,
            merge_rank: FastHashMap::default(),
            merge_id: FastHashMap::default(),
            cache: LruCache::new(150_000),
        }
    }

    // =========== Merge lookup table construction ============
    pub fn build_merge_lookups(&mut self) {
        self.merge_rank = self.merges.iter().enumerate()
            .map(|(rank, (pair, _))| (*pair, rank))
            .collect();
        self.merge_id = self.merges.iter()
            .map(|(pair, id)| (*pair, *id))
            .collect();
    }

    fn unk_ids(&self) -> u32 {
        *self.special_tokens_ids
            .get(&self.unknown_tokens)
            .expect("UNKNOWN TOKENS MUST BE INITIALIZED IN SPECIAL TOKEN IDS")
    }

    pub fn get_initial_word_char_segments(&self, word: &str) -> Vec<String> {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push(self.end_of_word_token.clone());
        chars
    }

    pub fn get_token_from_id(&self, id: u32) -> &String {
        self.reverse_vocab.get(&id).unwrap_or_else(|| {
            panic!("ID {} not found in reverse_vocab", id)
        })
    }

    pub fn get_id_to_token_map(&self) -> FastHashMap<u32, String> {
        self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect()
    }

    pub fn get_stats(&self) -> BPEStats {
        BPEStats {
            vocab_size: self.vocab.len(),
            num_merges: self.merges.len(),
            num_special_tokens: self.special_tokens_ids.len(),
            num_character_tokens: self.vocab.len()
                .saturating_sub(self.special_tokens_ids.len())
                .saturating_sub(self.merges.len()),
        }
    }

    // =========== Merge application helpers ============

    pub fn merge_pair_inplace(segments: &mut Vec<u32>, pair: &TokenPair, new_token_id: u32) {
        let mut write = 0;
        let mut read = 0;
        while read < segments.len() {
            if read + 1 < segments.len()
                && segments[read] == pair.0
                && segments[read + 1] == pair.1
            {
                segments[write] = new_token_id;
                write += 1;
                read += 2;
            } else {
                if write != read { segments[write] = segments[read]; }
                write += 1;
                read += 1;
            }
        }
        segments.truncate(write);
    }

    // encode and encode_batch are inference level tokenization functions, with options for padding, truncation, and special tokens
    pub fn encode(
        &mut self,
        text: &str,
        text_b: Option<&str>,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> TokenizedResult<Encoding> {
        let ids_a = self.encode_text_inner(text)?;
        let ids_b = match text_b {
            Some(b) => Some(self.encode_text_inner(b)?),
            None => None,
        };
        Ok(Encoding::build_with_pair(
            ids_a,
            ids_b,
            max_length,
            padding,
            truncation,
            add_special_tokens,
            &self.special_tokens_ids,
        ))
    }

    fn encode_text_inner(&mut self, text: &str) -> TokenizedResult<Vec<u32>> {
        let unk = self.unk_ids();
        encode_text_inner(
            text,
            &self.vocab,
            &self.merge_rank,
            &self.merge_id,
            &self.special_tokens_ids,
            &self.end_of_word_token,
            unk,
            &mut self.cache,
        )
    }

    // =========== BATCH ENCODE ============
    //  parallel_threshold is now the minimum texts-per-chunk to bother going parallel.
    //  For small batches, falls back to single-threaded.
    pub fn encode_batch(
        &self,
        texts: &[String],
        texts_b: Option<&[String]>,
        config: BatchEncodingConfig,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> TokenizedResult<Vec<Encoding>> {

        let shared = Arc::new(SharedTokenizerData {
            vocab: &self.vocab,
            merge_rank: &self.merge_rank,
            merge_id: &self.merge_id,
            special_tokens_ids: &self.special_tokens_ids,
            end_of_word_token: &self.end_of_word_token,
            unknown_tokens: &self.unknown_tokens,
        });

        // Single-threaded path for small batches
        if texts.len() < config.parallel_threshold {
            let mut local = self.clone();
            return texts.iter().enumerate().map(|(i, text)| {
                let text_b = texts_b.and_then(|b| b.get(i).map(|s| s.as_str()));
                local.encode(text, text_b, max_length, padding, truncation, add_special_tokens)
            }).collect();
        }

        let n_threads = config.max_threads.unwrap_or_else(rayon::current_num_threads);
        let chunk_size = cmp::max(config.parallel_threshold, texts.len() / n_threads);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| TokenizerError::EncodingError(e.to_string()))?;

        pool.install(|| {
            texts.par_chunks(chunk_size).enumerate().map(|(chunk_idx, chunk)| {
                let mut thread_tok = ThreadLocalTokenizer::new(
                    Arc::clone(&shared),
                    config.thread_cache_size,
                );
                let offset = chunk_idx * chunk_size;
                // in pool.install closure
                chunk.iter().enumerate().map(|(i, text)| {
                    let text_b = texts_b
                        .and_then(|b| b.get(offset + i).map(|s| s.as_str()));
                    thread_tok.encode(
                        text, text_b, max_length, padding, truncation, add_special_tokens
                    )
                }).collect::<TokenizedResult<Vec<Encoding>>>()
            })
            .collect::<TokenizedResult<Vec<Vec<Encoding>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
        })
    }

    // encode_fast and encode_batch_fast are made for PRETRAINING 
    pub fn encode_fast(&mut self, text: &str) -> TokenizedResult<Vec<u32>> {
        self.encode_text_inner(text)
    }

    pub fn encode_batch_fast(
        &self,
        texts: &[String],
        config: BatchEncodingConfig,
    ) -> TokenizedResult<Vec<Vec<u32>>> {
        let shared = Arc::new(SharedTokenizerData {
            vocab: &self.vocab,
            merge_rank: &self.merge_rank,
            merge_id: &self.merge_id,
            special_tokens_ids: &self.special_tokens_ids,
            end_of_word_token: &self.end_of_word_token,
            unknown_tokens: &self.unknown_tokens,
        });

        if texts.len() < config.parallel_threshold {
            let mut local = self.clone();
            return texts.iter().map(|t| local.encode_text_inner(t)).collect();
        }

        let n_threads = config.max_threads.unwrap_or_else(rayon::current_num_threads);
        let chunk_size = cmp::max(config.parallel_threshold, texts.len() / n_threads);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| TokenizerError::EncodingError(e.to_string()))?;

        pool.install(|| {
            texts.par_chunks(chunk_size).map(|chunk| {
                let mut thread_tok = ThreadLocalTokenizer::new(
                    Arc::clone(&shared),
                    config.thread_cache_size,
                );
                chunk.iter()
                    .map(|text| thread_tok.encode_text_inner(text))
                    .collect::<TokenizedResult<Vec<Vec<u32>>>>()
            })
            .collect::<TokenizedResult<Vec<Vec<Vec<u32>>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
        })
    }

    // =========== Decoder ============
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut parts: Vec<&str> = Vec::with_capacity(token_ids.len());

        for &id in token_ids {
            match self.reverse_vocab.get(&id) {
                None => parts.push(&self.unknown_tokens),
                Some(token) => {
                    if self.special_tokens_ids.contains_key(token)
                        && token != &self.unknown_tokens
                        && token != &self.end_of_word_token
                    {
                        continue;
                    }
                    parts.push(token);
                }
            }
        }

        let clean_text = parts.join("")
            .replace(&self.end_of_word_token, " ")
            .replace('⁄', " ")
            .replace('\u{2044}', " ");

        lazy_static! {
            static ref PUNCT_RE: Regex = Regex::new(r" ([.,!?;:])").unwrap();
        }
        PUNCT_RE.replace_all(clean_text.trim(), "$1").to_string()
    }


    pub fn decode_batch(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        batch_ids.par_iter()
                .map(|ids| self.decode(ids))
                .collect()
    }

    // =========== TRAIN THE TOKENIZER ============
    // after each merge, only the affected segments update pair counts.
    pub fn train(&mut self, corpus: &[String], config: TrainingConfig) {
        println!("Starting BPE Training");
        println!("Corpus: {} documents, target vocab: {}", corpus.len(), config.vocab_size);

        rayon::ThreadPoolBuilder::new()
            .num_threads(config.n_threads.unwrap_or(num_cpus::get()))
            .build_global()
            .ok(); // ignore if already initialized

        let normalizer = TextNormalizer::new().to_strip_accents();

        // --- Initial segmentation (parallel) ---
        let init_results: Vec<(FastHashMap<Vec<u32>, usize>, HashSet<String>)> =
            corpus.par_iter().map(|doc| {
                let norm = normalizer.normalize(doc);
                let mut local_segs: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
                let mut local_chars: HashSet<String> = HashSet::new();

                for word in PRE_TOKENIZER_RE.find_iter(&norm).map(|m| m.as_str()) {
                    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                    chars.push(self.end_of_word_token.clone());
                    for ch in word.chars() { local_chars.insert(ch.to_string()); }
                    let ids: Vec<u32> = chars.iter()
                        .map(|s| *self.vocab.get(s).unwrap_or(&self.unk_ids()))
                        .collect();
                    *local_segs.entry(ids).or_insert(0) += 1;
                }
                (local_segs, local_chars)
            }).collect();

        let mut word_segments: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
        let mut init_chars: HashSet<String> = HashSet::new();
        for (segs, chars) in init_results {
            for (seg, count) in segs { *word_segments.entry(seg).or_insert(0) += count; }
            init_chars.extend(chars);
        }

        let mut current_id = self.vocab.len() as u32;
        for ch in &init_chars {
            if !self.vocab.contains_key(ch) {
                self.vocab.insert(ch.clone(), current_id);
                self.reverse_vocab.insert(current_id, ch.clone());
                current_id += 1;
            }
        }
        for (token, &id) in &self.vocab {
            self.reverse_vocab.entry(id).or_insert_with(|| token.clone());
        }

        println!("Initial vocab size: {}", self.vocab.len());

        // ---build pair_freq once, then update incrementally ---
        let mut pair_freq: FastHashMap<TokenPair, usize> = FastHashMap::default();
        pair_freq.reserve(200_000);

        // Initial full build
        for (seg, &count) in &word_segments {
            for w in seg.windows(2) {
                *pair_freq.entry(TokenPair(w[0], w[1])).or_insert(0) += count;
            }
        }

        let n_merges = config.vocab_size.saturating_sub(self.vocab.len());
        let mut merges_done = 0;

        while self.vocab.len() < config.vocab_size {
            // Find best pair
            let Some((best_pair, max_freq)) = pair_freq.iter()
            .max_by_key(|(_, f)| *f)
            .map(|(pair, freq)|(*pair, *freq))
            else {
                println!("No valid pairs remain.");
                break;
            };
            
            if max_freq < config.min_frequency {
                println!("Stopping: max pair freq {} < min_frequency {}", max_freq, config.min_frequency);
                break;
            }

            let t1 = self.get_token_from_id(best_pair.0).clone();
            let t2 = self.get_token_from_id(best_pair.1).clone();
            let new_token = format!("{}{}", t1, t2);

            if self.vocab.contains_key(&new_token) {
                // Token already exists — remove this pair so we don't loop forever
                pair_freq.remove(&best_pair);
                continue;
            }

            let new_id = current_id;
            self.vocab.insert(new_token.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token.clone());
            current_id += 1;
            self.merges.push((best_pair, new_id));
            merges_done += 1;

            if config.show_progress {
                println!(
                    "Merge {}/{}: \"{}\" + \"{}\" (freq {}) → \"{}\" (vocab {})",
                    merges_done, n_merges, t1, t2, max_freq, new_token, self.vocab.len()
                );
            }

            // --- Incremental pair_freq update ---
            // For every segment containing best_pair:
            //   1. Subtract old pair counts for that segment
            //   2. Apply merge to get new segment
            //   3. Add new pair counts for new segment
            //   4. Update word_segments
            pair_freq.remove(&best_pair);

            let affected: Vec<(Vec<u32>, usize)> = word_segments
                .iter()
                .filter(|(seg, _)| {
                    seg.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1)
                })
                .map(|(seg, &count)| (seg.clone(), count))
                .collect();

            for (old_seg, count) in affected {
                // Decrement old pairs
                for w in old_seg.windows(2) {
                    let p = TokenPair(w[0], w[1]);
                    if let Some(f) = pair_freq.get_mut(&p) {
                        *f = f.saturating_sub(count);
                    }
                }

                // Build new segment
                let mut new_seg = old_seg.clone();
                Self::merge_pair_inplace(&mut new_seg, &best_pair, new_id);

                // Increment new pairs
                for w in new_seg.windows(2) {
                    *pair_freq.entry(TokenPair(w[0], w[1])).or_insert(0) += count;
                }

                // Update corpus
                word_segments.remove(&old_seg);
                *word_segments.entry(new_seg).or_insert(0) += count;
            }
        }

        println!("Training complete. Merges: {}, Vocab: {}", self.merges.len(), self.vocab.len());

        // Build lookup tables once after training
        self.build_merge_lookups();
    }

    // =========== Incremental training ============

    // Takes an iterator of documents, processes in chunks,
    // and performs merges every N chunks. 
    // State is maintained in IncrementalTrainingState.
    pub fn incremental_stream_train<I>(
        &mut self,
        corpus: I,
        config: IncrementalTrainingConfig,
        state: &mut IncrementalTrainingState,
    ) where I: Iterator<Item = String> {
        println!("Starting Incremental BPE Training");
        self.initialize_base_vocabulary(&config);
        state.current_vocab_size = self.vocab.len();

        let mut chunk_buffer = Vec::with_capacity(config.chunk_size);
        let normalizer = TextNormalizer::new().to_strip_accents();

        for (doc_idx, document) in corpus.enumerate() {
            chunk_buffer.push(document);
            if config.show_progress && doc_idx % 10_000 == 0 && doc_idx > 0 {
                println!("Documents seen: {}", doc_idx);
            }

            if chunk_buffer.len() >= config.chunk_size {
                self.process_chunk_incremental(&chunk_buffer, state, &normalizer, &config);
                state.chunks_processed += 1;
                state.total_documents += chunk_buffer.len();

                if state.chunks_processed % 50 == 0 {
                    state.aggressive_cleanup(2, 100);
                }
                if state.chunks_processed >= 3 && state.chunks_processed % config.merge_frequency == 0 {
                    self.perform_incremental_merges(state, &config);
                }
                chunk_buffer.clear();

                if self.vocab.len() >= config.vocab_size { break; }
            }
        }

        if self.vocab.len() < config.vocab_size && !chunk_buffer.is_empty() {
            self.process_chunk_incremental(&chunk_buffer, state, &normalizer, &config);
            state.chunks_processed += 1;
            state.total_documents += chunk_buffer.len();
        }

        state.aggressive_cleanup(1, 200);
        self.perform_incremental_merges(state, &config);
        self.build_merge_lookups();
        println!("Incremental training complete. Docs: {}, Vocab: {}", state.total_documents, self.vocab.len());
    }

    pub fn process_chunk_incremental(
        &mut self,
        chunk: &[String],
        state: &mut IncrementalTrainingState,
        normalizer: &TextNormalizer,
        config: &IncrementalTrainingConfig,
    ) {
        let chunk_results: Vec<(FastHashMap<Vec<u32>, usize>, HashSet<String>)> = chunk
            .par_iter()
            .map(|doc| {
                let norm = normalizer.normalize(doc);
                let mut local_segs: FastHashMap<Vec<u32>, usize> = FastHashMap::default();
                let mut local_chars: HashSet<String> = HashSet::new();
                for word in PRE_TOKENIZER_RE.find_iter(&norm).map(|m| m.as_str()) {
                    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                    chars.push(self.end_of_word_token.clone());
                    for ch in &chars { local_chars.insert(ch.clone()); }
                    let ids: Vec<u32> = chars.iter()
                        .map(|s| *self.vocab.get(s).unwrap_or(&self.unk_ids()))
                        .collect();
                    *local_segs.entry(ids).or_insert(0) += 1;
                }
                (local_segs, local_chars)
            })
            .collect();

        for (local_segs, local_chars) in chunk_results {
            for (seg, count) in local_segs {
                if count >= config.min_frequency {
                    *state.word_segments.entry(seg).or_insert(0) += count;
                }
            }
            for ch in local_chars {
                if !self.vocab.contains_key(&ch) {
                    let id = self.vocab.len() as u32;
                    self.vocab.insert(ch.clone(), id);
                    self.reverse_vocab.insert(id, ch);
                }
            }
        }
    }

    pub fn perform_incremental_merges(
        &mut self,
        state: &mut IncrementalTrainingState,
        config: &IncrementalTrainingConfig,
    ) {
        if self.vocab.len() >= config.vocab_size { return; }

        println!("Performing incremental merges...");

        state.pair_freq = state.word_segments
            .par_iter()
            .map(|(seg, &count)| {
                let mut local: FastHashMap<TokenPair, usize> = FastHashMap::default();
                for w in seg.windows(2) {
                    *local.entry(TokenPair(w[0], w[1])).or_insert(0) += count;
                }
                local
            })
            .reduce(FastHashMap::default, |mut a, b| {
                for (p, c) in b { *a.entry(p).or_insert(0) += c; }
                a
            });

        let mut merges_this_round = 0;
        while self.vocab.len() < config.vocab_size && merges_this_round < 1000 {
            let Some((best_pair, max_freq)) = state.pair_freq.iter()
            .max_by_key(|(_, f)| *f)
            .map(|(&pair, &freq)| (pair, freq))  
            else { break; };

            if max_freq < config.min_frequency { break; }

            let t1 = self.get_token_from_id(best_pair.0).clone();
            let t2 = self.get_token_from_id(best_pair.1).clone();
            let new_token = format!("{}{}", t1, t2);

            if self.vocab.contains_key(&new_token) {
                state.pair_freq.remove(&best_pair);
                continue;
            }

            let new_id = self.vocab.len() as u32;
            self.vocab.insert(new_token.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token.clone());
            self.merges.push((best_pair, new_id));

            if config.show_progress {
                println!("  Merge {}: {} + {} → '{}' (freq: {})", self.merges.len(), t1, t2, new_token, max_freq);
            }

            // Incremental update
            state.pair_freq.remove(&best_pair);

            let affected: Vec<(Vec<u32>, usize)> = state.word_segments
                .iter()
                .filter(|(seg, _)| seg.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1))
                .map(|(seg, &count)| (seg.clone(), count))
                .collect();

            for (old_seg, count) in affected {
                for w in old_seg.windows(2) {
                    let p = TokenPair(w[0], w[1]);
                    if let Some(f) = state.pair_freq.get_mut(&p) { *f = f.saturating_sub(count); }
                }
                let mut new_seg = old_seg.clone();
                Self::merge_pair_inplace(&mut new_seg, &best_pair, new_id);
                for w in new_seg.windows(2) {
                    *state.pair_freq.entry(TokenPair(w[0], w[1])).or_insert(0) += count;
                }
                state.word_segments.remove(&old_seg);
                *state.word_segments.entry(new_seg).or_insert(0) += count;
            }

            merges_this_round += 1;
        }

        state.last_merge_iteration = state.chunks_processed;
        println!("Completed {} merges this round", merges_this_round);
    }

    pub fn initialize_base_vocabulary(&mut self, _config: &IncrementalTrainingConfig) {
        for (token, &id) in &self.vocab {
            self.reverse_vocab.entry(id).or_insert_with(|| token.clone());
        }
    }

    // =========== Serialization ============

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(())
    }

    // Load tokenizer and rebuild merge lookup tables.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut tokenizer: Self = serde_json::from_reader(BufReader::new(file))?;
        tokenizer.cache = LruCache::new(150_000);
        tokenizer.build_merge_lookups(); 
        Ok(tokenizer)
    }

    pub fn save_checkpoint(
        &self,
        path: &str,
        state: IncrementalTrainingState,
        format: &mut &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Serialize)]
        struct Checkpoint<'a> {
            tokenizer: &'a BPETokenizer,
            state: &'a IncrementalTrainingState,
        }
        let checkpoint = Checkpoint { tokenizer: self, state: &state };
        let temp = format!("{}.tmp", path);
        {
            let writer = BufWriter::new(File::create(&temp)?);
            match format.to_lowercase().as_str() {
                "json" => serde_json::to_writer_pretty(writer, &checkpoint)?,
                "bincode" => bincode::serialize_into(writer, &checkpoint)?,
                fmt => return Err(Box::new(TokenizerError::UnsupportedFormat(fmt.to_string()))),
            }
        }
        std::fs::rename(&temp, path)?;
        Ok(())
    }

    pub fn load_checkpoint(path: &str) -> Result<(Self, IncrementalTrainingState), Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct Checkpoint { tokenizer: BPETokenizer, state: IncrementalTrainingState }
        let reader = BufReader::new(File::open(path)?);
        let mut checkpoint: Checkpoint = if path.ends_with(".json") {
            serde_json::from_reader(reader)?
        } else {
            bincode::deserialize_from(reader)?
        };
        checkpoint.tokenizer.build_merge_lookups();
        Ok((checkpoint.tokenizer, checkpoint.state))
    }
}

// =========== Thread-local tokenizer for encode_batch ============
// Holds a reference to the shared immutable tokenizer data and its own LRU cache.
// No full clone of vocab/merges/lookups per thread.
struct SharedTokenizerData<'a> {
    vocab: &'a FastHashMap<String, u32>,
    merge_rank: &'a FastHashMap<TokenPair, usize>,
    merge_id: &'a FastHashMap<TokenPair, u32>,
    special_tokens_ids: &'a FastHashMap<String, u32>,
    end_of_word_token: &'a str,
    unknown_tokens: &'a str,
}

// SAFETY: SharedTokenizerData only holds shared references into the Arc'd tokenizer.
// All fields are read-only during encode_batch. Rayon requires Send; &T is Send when T: Sync,
// and all our field types are Sync.
unsafe impl<'a> Send for SharedTokenizerData<'a> {}
unsafe impl<'a> Sync for SharedTokenizerData<'a> {}

struct ThreadLocalTokenizer<'a> {
    data: Arc<SharedTokenizerData<'a>>,
    cache: LruCache<String, Arc<Vec<u32>>>,
}

impl<'a> ThreadLocalTokenizer<'a> {
    fn new(data: Arc<SharedTokenizerData<'a>>, cache_size: usize) -> Self {
        Self { data, cache: LruCache::new(cache_size) }
    }

    fn unk_id(&self) -> u32 {
        *self.data.special_tokens_ids.get(self.data.unknown_tokens)
            .expect("unknown token missing")
    }

    fn encode(
        &mut self,
        text: &str,
        text_b: Option<&str>,  // now used
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> TokenizedResult<Encoding> {
        let ids_a = self.encode_text_inner(text)?;
        
        let ids_b = match text_b {
            Some(b) => Some(self.encode_text_inner(b)?),
            None => None,
        };

        Ok(Encoding::build_with_pair(
            ids_a,
            ids_b,
            max_length,
            padding,
            truncation,
            add_special_tokens,
            &self.data.special_tokens_ids,
        ))
    }

    fn encode_text_inner(&mut self, text: &str) -> TokenizedResult<Vec<u32>> {
        let unk = self.unk_id();
        encode_text_inner(
            text,
            &self.data.vocab,
            &self.data.merge_rank,
            &self.data.merge_id,
            &self.data.special_tokens_ids,
            &self.data.end_of_word_token,
            unk,
            &mut self.cache,
        )
    }

}

// =========== STATISTICS ============

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
            "BPE Statistics:\n  Vocabulary: {}\n  Special tokens: {}\n  Character tokens: {}\n  Merged tokens: {}",
            self.vocab_size, self.num_special_tokens, self.num_character_tokens, self.num_merges
        )
    }
}