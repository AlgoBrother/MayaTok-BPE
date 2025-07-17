use rayon::result;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
use regex::Regex;
use once_cell::sync::Lazy;
static WHITESPACE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

#[derive(Debug, Clone)]	
pub enum NormalisedFormat {
    NFD,  // Canonical Decomposition
    NFC,  // Canonical Composition (most common for storage)
    NFKD, // Compatibility Decomposition
    NFKC, // Compatibility Composition (most common for text processing)
}

impl Default for NormalisedFormat{
	fn default() -> Self{
		NormalisedFormat::NFKC
	}
}

#[derive(Debug, Clone)]
pub struct TextNormalizer{
	pub form: NormalisedFormat,
	pub lowercase : bool,
	pub strip_accents : bool,
}

impl Default for TextNormalizer{
	fn default() -> Self {
		Self { 
			form: NormalisedFormat::default(), 
			lowercase: false,
			strip_accents: false,
		}
	}
}

impl TextNormalizer{
	pub fn new() -> Self{
		Self::default()
	}

	pub fn form(mut self, form : NormalisedFormat) -> Self{
		self.form = form;
		self
	} 

	pub fn to_lowercase(mut self) -> Self{
		self.lowercase = true;
		self
	}

	pub fn to_strip_accents(mut self) -> Self{
		self.strip_accents = true;
		self
	}

	pub fn normalize_punctuation(text: &str) -> String{
		text.replace('—', "-")
			.replace('–', "-")
			.replace('’', "'")
			.replace('‘', "'")
			.replace('“', "\"")
			.replace('”', "\"")
			.replace('…', "...")
			.replace('\u{00A0}', " ")
	}

	pub fn normalize(&self, text: &str) -> String{

		let punch_text = Self::normalize_punctuation(text);
		let mut result = match self.form{
			NormalisedFormat::NFC => punch_text.nfc().collect::<String>(),
			NormalisedFormat::NFD => punch_text.nfd().collect::<String>(),
			NormalisedFormat::NFKD => punch_text.nfkd().collect::<String>(),
			NormalisedFormat::NFKC => punch_text.nfkc().collect::<String>(),
		};

		if self.strip_accents{
			let nfd_text = result.nfd().collect::<String>();
			result = nfd_text.chars().filter(|&c| !is_combining_mark(c)).collect();	
		}

		// if self.lowercase{
		// 	result = result.to_lowercase();
		// }

		self.normalize_whitespace(&result)
	}

	pub fn normalize_whitespace(&self, text : &str) -> String{
		WHITESPACE_REGEX.replace_all(text, " ").trim().to_string()
	}
}