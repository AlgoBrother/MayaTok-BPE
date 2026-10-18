use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

static WHITESPACE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

#[derive(Debug, Clone)]
pub enum NormalisedFormat {
    NFD,  // Canonical Decomposition
    NFC,  // Canonical Composition (most common for storage)
    NFKD, // Compatibility Decomposition
    NFKC, // Compatibility Composition (most common for text processing)
}

impl Default for NormalisedFormat {
    fn default() -> Self {
        NormalisedFormat::NFKC
    }
}

#[derive(Debug, Clone)]
/// A struct for normalizing text with optional math-preservation logic.
pub struct TextNormalizer {
    pub form: NormalisedFormat,
    pub lowercase: bool,
    pub strip_accents: bool,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self {
            form: NormalisedFormat::default(),
            lowercase: false,
            strip_accents: false,
        }
    }
}

impl TextNormalizer {
    /// Creates a new `TextNormalizer` instance with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Unicode normalization form for the `TextNormalizer`.
    pub fn form(mut self, form: NormalisedFormat) -> Self {
        self.form = form;
        self
    }

    pub fn to_lowercase(mut self) -> Self {
        self.lowercase = true;
        self
    }

    pub fn to_strip_accents(mut self) -> Self {
        self.strip_accents = true;
        self
    }

    /// Normalizes punctuation characters to standard equivalents.
    pub fn normalize_punctuation(text: &str) -> String {
        text.replace('—', "-")
            .replace('–', "-")
            .replace('’', "'")
            .replace('‘', "'")
            .replace('“', "\"")
            .replace('”', "\"")
            .replace('…', "...")
            .replace('\u{00A0}', " ")
    }

    /// Checks whether a character belongs to Unicode mathematical, symbol, or script ranges
    /// to bypass heavy NFKD decomposition and accent stripping.
    #[inline]
    fn is_math_script(c: char) -> bool {
        matches!(c, 
            '\u{2070}'..='\u{209D}' | // Superscripts and Subscripts
            '\u{00B2}' | '\u{00B3}' | '\u{00B9}' | // ², ³, ¹
            '\u{2200}'..='\u{22FF}' | // Mathematical Operators (∀, ∃, ∇, ∈, ∑, ∏, √, ∝, ∞, ∧, ∨, ∩, ∪, ∫, ≈, ≠, ≤, ≥)
            '\u{2190}'..='\u{21FF}' | // Arrows (←, ↑, →, ↓, ↔, ⇐, ⇒, ⇔, ↦)
            '\u{2100}'..='\u{214F}' | // Letterlike Symbols (ℏ, ℓ, ℝ, ℂ, ℕ, ℤ, ℚ, ℒ, 𝒪)
            '\u{2300}'..='\u{23FF}' | // Miscellaneous Technical (⌈, ⌉, ⌊, ⌋, 〈, 〉)
            '\u{2A00}'..='\u{2AFF}' | // Supplemental Mathematical Operators
            '\u{27C0}'..='\u{27EF}' | // Miscellaneous Mathematical Symbols-A
            '\u{2980}'..='\u{29FF}'   // Miscellaneous Mathematical Symbols-B
        )
    }

    /// Main normalization entry point.
    pub fn normalize(&self, text: &str) -> String {
        // 1. Bypass heavy normalization and accent stripping if text contains math/technical symbols
        if text.chars().any(Self::is_math_script) {
            return text.to_string();
        }

        // 2. Normalize standard punctuation
        let punch_text = Self::normalize_punctuation(text);

        // 3. Apply selected Unicode normalization form
        let mut result = match self.form {
            NormalisedFormat::NFC => punch_text.nfc().collect::<String>(),
            NormalisedFormat::NFD => punch_text.nfd().collect::<String>(),
            NormalisedFormat::NFKD => punch_text.nfkd().collect::<String>(),
            NormalisedFormat::NFKC => punch_text.nfkc().collect::<String>(),
        };

        // 4. Strip accents if explicitly requested
        if self.strip_accents {
            let nfd_text = result.nfd().collect::<String>();
            result = nfd_text.chars().filter(|&c| !is_combining_mark(c)).collect(); 
        }

        if self.lowercase {
            result = result.to_lowercase();
        }

        self.normalize_whitespace(&result)
    }

    pub fn normalize_whitespace(&self, text: &str) -> String {
        WHITESPACE_REGEX.replace_all(text, " ").trim().to_string()
    }
}
