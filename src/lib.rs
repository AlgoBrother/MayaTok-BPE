use std::sync::{Arc, Mutex};

use pyo3::{prelude::*, types::PyIterator};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::path::PathBuf;


mod text_normalizer;
mod bpe_tokenizer_lib;
pub use bpe_tokenizer_lib::{
    BPETokenizer, BPEStats, BatchEncodingConfig, TrainingConfig, 
    TokenizerError, IncrementalTrainingConfig, IncrementalTrainingState, Encoding
};

// ========== VOCAB FILE MANAGEMENT ===========
fn get_cache_dir() -> PathBuf {
    let mut path = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    path.push("mayatok");
    std::fs::create_dir_all(&path).ok();
    path
}

fn load_or_download(name: &str, url: &str) -> Result<PathBuf, String> {
    let mut cache_path = get_cache_dir();
    cache_path.push(format!("{}.json", name));

    if cache_path.exists() {
        return Ok(cache_path);
    }

    // Download
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;
    let content = response.into_string()
        .map_err(|e| format!("Read failed: {}", e))?;
    std::fs::write(&cache_path, content)
        .map_err(|e| format!("Write failed: {}", e))?;

    Ok(cache_path)
}

lazy_static::lazy_static! {
    static ref REGISTRY: std::collections::HashMap<&'static str, &'static str> = {
        let mut m = std::collections::HashMap::new();
        m.insert("mayatok-base", 
            "https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json");
        m
    };
}

#[pyfunction]
fn get_tokenizer(py: Python, name: String) -> PyResult<PyBPETokenizer> {
    let url = REGISTRY.get(name.as_str())
        .ok_or_else(|| PyValueError::new_err(
            format!("Unknown tokenizer '{}'. Available: {:?}", name, 
                    REGISTRY.keys().collect::<Vec<_>>())
        ))?;

    let path = load_or_download(&name, url)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    PyBPETokenizer::load(py, path.to_string_lossy().to_string())
}

#[pyfunction]
fn list_tokenizers() -> Vec<String> {
    REGISTRY.keys().map(|s| s.to_string()).collect()
}
// =========== VOCAB FILES ENDS ===========

// =========== PyErr conversion ===========
impl From<TokenizerError> for PyErr {
    fn from(err: TokenizerError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// =========== PyEncoding ===========
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct PyEncoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,
}

#[pymethods]
impl PyEncoding {
    fn __repr__(&self) -> String {
        format!(
            "Encoding(input_ids={:?}, attention_mask={:?}, token_type_ids={:?})",
            self.input_ids, self.attention_mask, self.token_type_ids
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("input_ids", self.input_ids.clone())?;
        dict.set_item("attention_mask", self.attention_mask.clone())?;
        dict.set_item("token_type_ids", self.token_type_ids.clone())?;
        Ok(dict.into())
    }
}

impl From<Encoding> for PyEncoding {
    fn from(enc: Encoding) -> Self {
        PyEncoding {
            input_ids: enc.input_ids,
            attention_mask: enc.attention_mask,
            token_type_ids: enc.token_type_ids,
        }
    }
}

// =========== Config classes ===========
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default)]
pub struct PyTrainConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub n_threads: Option<usize>,
    pub special_tokens: Vec<String>,
    pub show_progress: bool,
}

#[pymethods]
impl PyTrainConfig {
    #[new]
    fn new(
        vocab_size: Option<usize>,
        min_frequency: Option<usize>,
        n_threads: Option<usize>,
        special_tokens: Option<Vec<String>>,
        show_progress: Option<bool>,
    ) -> Self {
        let default_config = TrainingConfig::default();
        PyTrainConfig {
            vocab_size: vocab_size.unwrap_or(default_config.vocab_size),
            min_frequency: min_frequency.unwrap_or(default_config.min_frequency),
            n_threads: n_threads.or(default_config.n_threads),
            special_tokens: special_tokens.unwrap_or(default_config.special_tokens),
            show_progress: show_progress.unwrap_or(default_config.show_progress),
        }
    }
}

impl From<TrainingConfig> for PyTrainConfig {
    fn from(config: TrainingConfig) -> Self {
        PyTrainConfig {
            vocab_size: config.vocab_size,
            min_frequency: config.min_frequency,
            n_threads: config.n_threads,
            special_tokens: config.special_tokens,
            show_progress: config.show_progress,
        }
    }
}

impl From<PyTrainConfig> for TrainingConfig {
    fn from(config: PyTrainConfig) -> Self {
        TrainingConfig {
            vocab_size: config.vocab_size,
            min_frequency: config.min_frequency,
            special_tokens: config.special_tokens,
            show_progress: config.show_progress,
            n_threads: config.n_threads,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct PyIncrementalTrainingConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub chunk_size: usize,
    pub merge_frequency: usize,
    pub save_frequency: usize,
    pub checkpoint_path: Option<String>,
    pub show_progress: bool,
    pub special_tokens: Vec<String>,
}

#[pymethods]
impl PyIncrementalTrainingConfig {
    #[new]
    fn new(
        vocab_size: Option<usize>,
        min_frequency: Option<usize>,
        chunk_size: Option<usize>,
        merge_frequency: Option<usize>,
        save_frequency: Option<usize>,
        checkpoint_path: Option<String>,
        show_progress: Option<bool>,
        special_tokens: Option<Vec<String>>,
    ) -> Self {
        PyIncrementalTrainingConfig {
            vocab_size: vocab_size.unwrap_or(30000),
            min_frequency: min_frequency.unwrap_or(2),
            chunk_size: chunk_size.unwrap_or(1000),
            merge_frequency: merge_frequency.unwrap_or(10),
            save_frequency: save_frequency.unwrap_or(100),
            checkpoint_path,
            show_progress: show_progress.unwrap_or(true),
            special_tokens: special_tokens.unwrap_or_else(|| vec![
                "<pad>".to_string(), "<unk>".to_string(),
                "<s>".to_string(), "</s>".to_string(),
            ]),
        }
    }
}

impl From<PyIncrementalTrainingConfig> for IncrementalTrainingConfig {
    fn from(config: PyIncrementalTrainingConfig) -> Self {
        IncrementalTrainingConfig {
            vocab_size: config.vocab_size,
            min_frequency: config.min_frequency,
            chunk_size: config.chunk_size,
            merge_frequency: config.merge_frequency,
            save_frequency: config.save_frequency,
            checkpoint_path: config.checkpoint_path,
            show_progress: config.show_progress,
            special_tokens: config.special_tokens,
            n_threads: None,
        }
    }
}

#[pyclass(name = "IncrementalTrainingState")]
#[derive(Debug, Clone)]
pub struct PyIncrementalTrainingState {
    pub inner: Arc<Mutex<IncrementalTrainingState>>,
}

#[pymethods]
impl PyIncrementalTrainingState {
    #[new]
    fn new() -> Self {
        PyIncrementalTrainingState {
            inner: Arc::new(Mutex::new(IncrementalTrainingState::new())),
        }
    }

    #[getter]
    fn chunks_processed(&self) -> usize {
        self.inner.lock().unwrap().chunks_processed
    }

    #[getter]
    fn total_documents(&self) -> usize {
        self.inner.lock().unwrap().total_documents
    }

    #[getter]
    fn current_vocab_size(&self) -> usize {
        self.inner.lock().unwrap().current_vocab_size
    }

    #[getter]
    fn last_merge_iteration(&self) -> usize {
        self.inner.lock().unwrap().last_merge_iteration
    }

    #[getter]
    fn word_segments_count(&self) -> usize {
        self.inner.lock().unwrap().word_segments.len()
    }

    fn __repr__(&self) -> String {
        let state = self.inner.lock().unwrap();
        format!(
            "IncrementalTrainingState(chunks={}, docs={}, vocab={}, last_merge={})",
            state.chunks_processed, state.total_documents,
            state.current_vocab_size, state.last_merge_iteration
        )
    }

    fn reconstruct_pair_freq(&mut self) -> PyResult<()> {
        let mut state = self.inner.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        state.reconstruct_pair_freq();
        Ok(())
    }
}

impl From<IncrementalTrainingState> for PyIncrementalTrainingState {
    fn from(state: IncrementalTrainingState) -> Self {
        PyIncrementalTrainingState {
            inner: Arc::new(Mutex::new(state)),
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default)]
pub struct PyBatchEncodingConfig {
    pub max_length: Option<usize>,
    pub parallel_threshold: usize,
    pub max_threads: Option<usize>,
    pub use_thread_local_cache: bool,
    pub thread_cache_size: usize,
}

#[pymethods]
impl PyBatchEncodingConfig {
    #[new]
    fn new(
        max_length: Option<usize>,
        parallel_threshold: Option<usize>,
        max_threads: Option<usize>,
        use_thread_local_cache: Option<bool>,
        thread_cache_size: Option<usize>,
    ) -> Self {
        let default_config = BatchEncodingConfig::default();
        PyBatchEncodingConfig {
            max_length: max_length.or(default_config.max_length),
            parallel_threshold: parallel_threshold.unwrap_or(default_config.parallel_threshold),
            max_threads: max_threads.or(default_config.max_threads),
            use_thread_local_cache: use_thread_local_cache.unwrap_or(default_config.use_thread_local_cache),
            thread_cache_size: thread_cache_size.unwrap_or(default_config.thread_cache_size),
        }
    }
}

impl From<PyBatchEncodingConfig> for BatchEncodingConfig {
    fn from(config: PyBatchEncodingConfig) -> Self {
        BatchEncodingConfig {
            max_length: config.max_length,
            parallel_threshold: config.parallel_threshold,
            max_threads: config.max_threads,
            use_thread_local_cache: config.use_thread_local_cache,
            thread_cache_size: config.thread_cache_size,
        }
    }
}

// =========== PyCorpusIterator ===========
#[pyclass]
struct PyCorpusIterator {
    inner: Vec<String>,
    index: usize,
}

#[pymethods]
impl PyCorpusIterator {
    #[new]
    fn new(corpus: Vec<String>) -> Self {
        PyCorpusIterator { inner: corpus, index: 0 }
    }
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }
    fn __next__(mut slf: PyRefMut<Self>) -> Option<String> {
        if slf.index < slf.inner.len() {
            let item = slf.inner[slf.index].clone();
            slf.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

// =========== PyBPETokenizer ===========
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyBPETokenizer {
    tokenizer: Arc<Mutex<BPETokenizer>>,
}

#[pymethods]
impl PyBPETokenizer {
    #[new]
    fn new() -> Self {
        Self { tokenizer: Arc::new(Mutex::new(BPETokenizer::new())) }
    }

    fn train(&mut self, corpus: Vec<String>, config: Option<PyTrainConfig>) -> PyResult<()> {
        let rust_config = config.map_or_else(TrainingConfig::default, |c| c.into());
        let mut tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.train(&corpus, rust_config);
        Ok(())
    }

    fn train_incremental(
        &mut self, py: Python,
        corpus: Vec<String>,
        config: PyIncrementalTrainingConfig,
        state: PyIncrementalTrainingState,
    ) -> PyResult<()> {
        let rust_config = config.into();
        py.allow_threads(|| {
            let mut inner_state = state.inner.lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut tokenizer = self.tokenizer.lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.incremental_stream_train(corpus.into_iter(), rust_config, &mut *inner_state);
            Ok::<(), PyErr>(())
        })?;
        Ok(())
    }

    #[pyo3(name = "train_stream")]
    fn train_incremental_iter(
        &mut self,
        corpus_iter: &Bound<'_, PyAny>,
        config: PyIncrementalTrainingConfig,
        state: &mut PyIncrementalTrainingState,
    ) -> PyResult<()> {
        let rust_config = config.into();
        let py_iter = PyIterator::from_object(corpus_iter)?;
        let (sender, receiver) = crossbeam_channel::bounded::<String>(128);
        let tokenizer_clone = self.tokenizer.clone();
        let state_clone = state.inner.clone();

        std::thread::spawn(move || {
            let mut tokenizer_guard = tokenizer_clone.lock().unwrap();
            let mut state_guard = state_clone.lock().unwrap();
            tokenizer_guard.incremental_stream_train(
                receiver.into_iter(), rust_config, &mut *state_guard
            );
        });

        for item in py_iter {
            if let Ok(line) = item.and_then(|obj| obj.extract::<String>()) {
                if sender.send(line).is_err() { break; }
            }
        }
        drop(sender);
        Ok(())
    }

    // =========== ENCODE ===========
    // encode → List[int], 
    #[pyo3(signature = (text))]
    fn encode(&mut self, text: String) -> PyResult<Vec<u32>> {
        let mut tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode_fast(&text).map_err(PyErr::from)
    }

    // encode_batch → List[List[int]],  
    #[pyo3(signature = (texts, config=None))]
    fn encode_batch(
        &self,
        texts: Vec<String>,
        config: Option<PyBatchEncodingConfig>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let rust_config = config.map_or_else(BatchEncodingConfig::default, |c| c.into());
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode_batch_fast(&texts, rust_config)
            .map_err(PyErr::from)
    }

    // encode_plus → PyEncoding with masks, for fine-tuning/inference
    #[pyo3(signature = (
        text,
        text_b=None,
        max_length=None,
        padding=false,
        truncation=false,
        add_special_tokens=false
    ))]
    fn encode_plus(
        &mut self,
        text: String,
        text_b: Option<String>,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let mut tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode(
            &text,
            text_b.as_deref(),
            max_length,
            padding,
            truncation,
            add_special_tokens,
        )
        .map(PyEncoding::from)
        .map_err(PyErr::from)
    }

    // encode_batch_plus → List[PyEncoding]
    #[pyo3(signature = (
        texts,
        texts_b=None,
        config=None,
        max_length=None,
        padding=false,
        truncation=false,
        add_special_tokens=false
    ))]
    fn encode_batch_plus(
        &self,
        texts: Vec<String>,
        texts_b: Option<Vec<String>>,
        config: Option<PyBatchEncodingConfig>,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let rust_config = config.map_or_else(BatchEncodingConfig::default, |c| c.into());
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode_batch(
            &texts,
            texts_b.as_deref(),
            rust_config,
            max_length,
            padding,
            truncation,
            add_special_tokens,
        )
        .map(|encs| encs.into_iter().map(PyEncoding::from).collect())
        .map_err(PyErr::from)
    }

    fn get_token_from_id(&self, id: u32) -> PyResult<String> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.reverse_vocab
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "<UNK>".to_string()))
    }


    // =========== DECODE ===========
    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.decode(&token_ids))
    }

    // =========== BATCH DECODE ===========
    fn decode_batch(&self, batch_ids: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.decode_batch(&batch_ids))
    }

    // =========== save / load ===========
    fn save(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let tokenizer = self.tokenizer.lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.save(&path).map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[staticmethod]
    fn load(py: Python, path: String) -> PyResult<PyBPETokenizer> {
        py.allow_threads(|| {
            BPETokenizer::load(&path)
                .map(|tokenizer| PyBPETokenizer {
                    tokenizer: Arc::new(Mutex::new(tokenizer)),
                })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    fn save_checkpoint(
        &self, py: Python,
        path: String,
        state: &PyIncrementalTrainingState,
        format: String,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let rust_state = state.inner.lock()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let rust_state_clone = (*rust_state).clone();
            let mut format_str: &str = format.as_str();
            let tokenizer = self.tokenizer.lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.save_checkpoint(&path, rust_state_clone, &mut format_str)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[staticmethod]
    fn load_checkpoint(
        py: Python,
        path: String,
    ) -> PyResult<(PyBPETokenizer, PyIncrementalTrainingState)> {
        py.allow_threads(|| {
            BPETokenizer::load_checkpoint(&path)
                .map(|(tokenizer, state)| {
                    let py_tokenizer = PyBPETokenizer {
                        tokenizer: Arc::new(Mutex::new(tokenizer)),
                    };
                    (py_tokenizer, state.into())
                })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    // =========== getters ===========
    fn get_stats(&self) -> PyResult<PyBPEStats> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.get_stats().into())
    }

    fn get_vocab(&self) -> PyResult<std::collections::HashMap<String, u32>> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.vocab.iter().map(|(k, &v)| (k.clone(), v)).collect())
    }

    fn token_to_id(&self, token: String) -> PyResult<Option<u32>> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.vocab.get(&token).copied())
    }

    fn id_to_token(&self, id: u32) -> PyResult<Option<String>> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.reverse_vocab.get(&id).cloned())
    }

    fn add_tokens(&mut self, tokens: Vec<String>) -> PyResult<usize> {
        let mut tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut added = 0;
        for token in tokens {
            if !tokenizer.vocab.contains_key(&token) {
                let new_id = tokenizer.vocab.len() as u32;
                tokenizer.vocab.insert(token.clone(), new_id);
                tokenizer.reverse_vocab.insert(new_id, token);
                added += 1;
            }
        }
        Ok(added)
    }

    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.vocab.len())
    }

    #[getter]
    fn num_merges(&self) -> PyResult<usize> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.merges.len())
    }

    #[getter]
    fn print_vocab(&self) -> PyResult<String> {
        let tokenizer = self.tokenizer.lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.vocab.iter()
            .map(|(token, id)| format!("{}: {}", token, id))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

// =========== PyBPEStats ===========
#[pyclass(get_all)]
#[derive(Debug, Clone)]
struct PyBPEStats {
    vocab_size: usize,
    num_merges: usize,
    num_special_tokens: usize,
    num_character_tokens: usize,
}

#[pymethods]
impl PyBPEStats {
    #[new]
    fn new(
        vocab_size: usize,
        num_merges: usize,
        num_special_tokens: usize,
        num_character_tokens: usize,
    ) -> Self {
        PyBPEStats { vocab_size, num_merges, num_special_tokens, num_character_tokens }
    }
}

impl From<BPEStats> for PyBPEStats {
    fn from(stats: BPEStats) -> Self {
        PyBPEStats {
            vocab_size: stats.vocab_size,
            num_merges: stats.num_merges,
            num_special_tokens: stats.num_special_tokens,
            num_character_tokens: stats.num_character_tokens,
        }
    }
}

// =========== Module ===========
#[pymodule]
fn mayatok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(list_tokenizers, m)?)?;
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyTrainConfig>()?;
    m.add_class::<PyIncrementalTrainingConfig>()?;
    m.add_class::<PyIncrementalTrainingState>()?;
    m.add_class::<PyBatchEncodingConfig>()?;
    m.add_class::<PyBPETokenizer>()?;
    m.add_class::<PyBPEStats>()?;
    m.add_class::<PyCorpusIterator>()?;
    Ok(())
}