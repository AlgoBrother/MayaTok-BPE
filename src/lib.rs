use std::sync::{Arc, Mutex};

use pyo3::{prelude::*, types::PyIterator};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

mod text_normalizer;
mod bpe_tokenizer_lib;
pub use bpe_tokenizer_lib::{BPETokenizer, BPEStats, BatchEncodingConfig, TrainingConfig, TokenizerError, IncrementalTrainingConfig, IncrementalTrainingState};

// Function to convert Rust's TokenizerError to PyO3's PyErr
impl From<TokenizerError> for PyErr {
    fn from(err: TokenizerError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// Python class for training configuration
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

// Python class for Incremental Training Configuration
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
                "<pad>".to_string(),
                "<unk>".to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
            ]),
        }
    }
}

impl From<IncrementalTrainingConfig> for PyIncrementalTrainingConfig {
    fn from(config: IncrementalTrainingConfig) -> Self {
        PyIncrementalTrainingConfig {
            vocab_size: config.vocab_size,
            min_frequency: config.min_frequency,
            chunk_size: config.chunk_size,
            merge_frequency: config.merge_frequency,
            save_frequency: config.save_frequency,
            checkpoint_path: config.checkpoint_path,
            show_progress: config.show_progress,
            special_tokens: config.special_tokens,
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
            n_threads: None, // Incremental training typically uses a single thread
        }
    }
}

// Python class for Incremental Training State
#[pyclass(name = "IncrementalTrainingState")]
#[derive(Debug, Clone)]
pub struct PyIncrementalTrainingState {                  
    pub inner : Arc<Mutex<IncrementalTrainingState>>,
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
    fn chunks_processed(&self) -> usize{
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
            "IncrementalTrainingState(chunks_processed={}, total_documents={}, current_vocab_size={}, last_merge_iteration={}, word_segments_count={})",
            state.chunks_processed, state.total_documents, state.current_vocab_size, state.last_merge_iteration, state.word_segments.len()
        )
    }
    
    #[pyo3(text_signature = "(self, tokenizer)")]
    fn reconstruct_pair_freq(&self, tokenizer: &PyBPETokenizer) -> PyResult<()> {
        let mut state = self.inner.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tokenizer = tokenizer.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Call the method on the actual IncrementalTrainingState struct
        state.reconstruct_pair_freq(&tokenizer);

        Ok(())
    }
}

impl From<IncrementalTrainingState> for PyIncrementalTrainingState {
    fn from(state: IncrementalTrainingState) -> Self {
        PyIncrementalTrainingState {
            inner : Arc::new(Mutex::new(state)),
        }
    }
}

// Python class for BatchEncodingConfig
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

impl From<BatchEncodingConfig> for PyBatchEncodingConfig {
    fn from(config: BatchEncodingConfig) -> Self {
        PyBatchEncodingConfig {
            max_length: config.max_length,
            parallel_threshold: config.parallel_threshold,
            max_threads: config.max_threads,
            use_thread_local_cache: config.use_thread_local_cache,
            thread_cache_size: config.thread_cache_size,
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

// Python Iterator wrapper for incremental training
#[pyclass]
struct PyCorpusIterator {
    inner: Vec<String>,
    index: usize,
}

#[pymethods]
impl PyCorpusIterator {
    #[new]
    fn new(corpus: Vec<String>) -> Self {
        PyCorpusIterator {
            inner: corpus,
            index: 0,
        }
    }
    
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
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

// Define a Python class for BPETokenizer
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyBPETokenizer {
    tokenizer: Arc<Mutex<BPETokenizer>>,
}

#[pymethods]
impl PyBPETokenizer {
    #[new]
    fn new() -> Self {
        Self {
            tokenizer: Arc::new(Mutex::new(BPETokenizer::new())),
        }
    }

    #[pyo3(text_signature = "(self, corpus, config=None)")]
    fn train(&mut self, corpus: Vec<String>, config: Option<PyTrainConfig>) -> PyResult<()> {
        let rust_config = config.map_or_else(TrainingConfig::default, |c| c.into());
        let mut tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.train(&corpus, rust_config);
        Ok(())
    }

    #[pyo3(text_signature = "(self, corpus, config, state)")]
    fn train_incremental(&mut self, py: Python, corpus: Vec<String>, config: PyIncrementalTrainingConfig, state: PyIncrementalTrainingState) -> PyResult<()> {
        let rust_config = config.into();

        py.allow_threads(|| {
            let mut inner_state = state.inner.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.incremental_stream_train(corpus.into_iter(), rust_config, &mut *inner_state);
            Ok::<(), PyErr>(())
        })?;

        Ok(())
    }


    #[pyo3(name = "train_stream", text_signature = "(self, corpus_iter, config)")]
    fn train_incremental_iter(&mut self, corpus_iter: &Bound<'_, PyAny>, config: PyIncrementalTrainingConfig, state: &mut PyIncrementalTrainingState) -> PyResult<()> {
        let rust_config = config.into();
        let py_iter = PyIterator::from_object(corpus_iter)?;

        // Channel for line streaming
        let (sender, receiver) = crossbeam_channel::bounded::<String>(128);

        // Clone tokenizer for the training thread
        let tokenizer_clone = self.tokenizer.clone();

        // Creating a static handle to the shared state
        let state_clone = state.inner.clone();

        // Spawn training in a new thread (streaming!)
        std::thread::spawn(move || {
            let mut tokenizer_guard = tokenizer_clone.lock().unwrap();
            let mut inner_state_guard = state_clone.lock().unwrap();

            tokenizer_guard.incremental_stream_train(receiver.into_iter(), rust_config, &mut *inner_state_guard);
        });

        // Feed Python iterator on this thread
        for item in py_iter {
            if let Ok(line) = item.and_then(|obj| obj.extract::<String>()) {
                if sender.send(line).is_err() {
                    break; // Receiver dropped
                }
            } else {
                eprintln!("Skipping line: failed to extract string.");
            }
        }

        drop(sender); // Signal completion to training thread

        Ok(())
    }

    #[pyo3(text_signature = "(self, text)")]
    fn encode(&mut self, text: String) -> PyResult<Vec<u32>> {
        let mut tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode(&text).map_err(PyErr::from)
    }

    #[pyo3(text_signature = "(self, texts, config=None)")]
    fn encode_batch(&mut self, texts: Vec<String>, config: Option<PyBatchEncodingConfig>) -> PyResult<Vec<Vec<u32>>> {
        let rust_config = config.map_or_else(BatchEncodingConfig::default, |c| c.into());
        let mut tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tokenizer.encode_batch(&texts, rust_config).map_err(PyErr::from)
    }

    #[pyo3(text_signature = "(self, token_ids)")]
    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.decode(&token_ids))
    }

    #[pyo3(text_signature = "(self, path)")]
    fn save(&self, py: Python, path: String) -> PyResult<()> {
        py.allow_threads(|| {
            let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.save(&path).map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn load(py: Python, path: String) -> PyResult<PyBPETokenizer> {
        
        py.allow_threads(|| {
            BPETokenizer::load(&path)
                .map(|tokenizer| {
                    let py_tokenizer = PyBPETokenizer { 
                        // wrap the loaded tokenizer for thread-safety, doing it manually
                        tokenizer : Arc::new(Mutex::new(tokenizer)),
                    };
                    py_tokenizer
                })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
    
    #[pyo3(text_signature = "(self, path, state, format)")]
    fn save_checkpoint(&self, py: Python, path: String, state: &PyIncrementalTrainingState, format: String) -> PyResult<()> {
        py.allow_threads(|| {
            // Change `.unwrap()` to properly handle the PoisonError
            let rust_state = state.inner.lock()
                .map_err(|e| PyValueError::new_err(format!("Cannot save checkpoint: lock was poisoned by a failing thread. Original error: {}", e)))?;

            let rust_state_clone = (*rust_state).clone();
            let mut format_str: &str = format.as_str();
            let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tokenizer.save_checkpoint(&path, rust_state_clone, &mut format_str)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn load_checkpoint(py: Python, path: String) -> PyResult<(PyBPETokenizer, PyIncrementalTrainingState)> {
        py.allow_threads(|| {
            BPETokenizer::load_checkpoint(&path)
                .map(|(tokenizer, state)| {
                    let py_tokenizer = PyBPETokenizer { 
                        // wrap the loaded tokenizer for thread-safety, doing it manually
                        tokenizer : Arc::new(Mutex::new(tokenizer)),
                    };
                    
                    let py_state = state.into();
                    (py_tokenizer, py_state)
                })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
    
    #[pyo3(text_signature = "(self)")]
    fn get_stats(&self) -> PyResult<PyBPEStats> {
        let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.get_stats().into())
    }
    
    #[getter]
    fn print_vocab(&self) -> PyResult<String> {
        let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let vocab_str = tokenizer.vocab.iter()
            .map(|(token, id)| format!("{}: {}", token, id))
            .collect::<Vec<String>>()
            .join("\n");
        
        Ok(vocab_str)
    }

    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.vocab.len())
    }
    
    #[getter]
    fn num_merges(&self) -> PyResult<usize> {
        let tokenizer = self.tokenizer.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(tokenizer.merges.len())
    }
}

// Define a Python class for BPEStats
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
        PyBPEStats {
            vocab_size,
            num_merges,
            num_special_tokens,
            num_character_tokens,
        }
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

// This macro creates the Python module
#[pymodule]
fn mayatok_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainConfig>()?;
    m.add_class::<PyIncrementalTrainingConfig>()?;
    m.add_class::<PyIncrementalTrainingState>()?;
    m.add_class::<PyBatchEncodingConfig>()?;
    m.add_class::<PyBPETokenizer>()?;
    m.add_class::<PyBPEStats>()?;
    m.add_class::<PyCorpusIterator>()?;
    Ok(())
}














