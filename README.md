# MayaTok
MayaTok is a Byte-Pair Encoding (BPE) tokenizer written in Rust. Built with performance and extensibility in mind. I made this project just because I wanted to study how Byte Pair Encoding Works. 

> Version: **2.1.4**

## ⚡️ Features
 
- Multithreaded training for fast vocab generation 

- Persistent merges 

- Checkpoint saving

- Focus on raw speed — built for performance benchmarking

## 🚀 Installation

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (required for 'from source' approach)
- Python 3.9+ (for Python bindings)

### PIP Installation

```bash
pip install mayatok
```

### From Source
```bash
git clone https://github.com/AlgoBrother/MayaTok-BPE.git
cd mayatok-bpe
```

Use maturin for building wheels.

```bash
pip install maturin
maturin build --release
pip install target/wheels/*.whl
```

### Quick Start 

## Using with Python

To use MayaTok with Python:

```python
import mayatok as bpe

my_tokenizer =  bpe.get_tokenizer("vb100k") # or 'mayatok-base' if you wish to use v1 tokenizer
test = "Hello, world!"
tokens = my_tokenizer.encode(test)
print(tokens)
decoded_text = my_tokenizer.decode(tokens)
print(decoded_text)
```

Output of the sample code above
```
[11617, 77, 3646, 62]
Hello, world!
```
## If you want to create your own Vocab 

If you are using HuggingFace Datasets, refer to [this](dataset_training\train.py) for creating your own vocab.

### If your dataset is in your local machine
 
> Make sure you have forked/cloned the rust tokenizer code and have built the /target/wheels as mentioned in previous steps

[stream method](examples/train_your_own_vocab.py) - If you have a large dataset and want to stream your data in chunks to not overload your machine. Use this.

[non-stream method](examples/non_stream_train_your_own_vocab.py) - If you have a dataset which your RAM can handle after being loaded, use this for much faster training.

## 📈 Benchmarks

### Batch Encoding

| Tokenizer   | Tokens/sec | Avg Compression Ratio |
| ----------- | ---------- | --------------------- |
| **MayaTok-BPE** | **2,277,698**     | **3.92**                  |
| tiktoken-cl100k   | 284,016    | 4.11              |
| tiktoken-p50k   | 268,657    | 4.08            |
| GPT2        | 1,107,899    | 4.04             |
| Falcon-7B   | 938,424   | 4.07           |


### Normal Encoding

| Tokenizer   | Tokens/sec | Compression Ratio |
| ----------- | ---------- | ----------------- |
| **MayaTok** | **302,276**      | **3.92**              |
| tiktoken-cl100k   | 678,801   | 4.11             |
| tiktoken-p50k   | 750,891    | 4.08             |
| GPT2        | 215,831 | 4.04            |
| Falcon-7B   | 177,707   | 4.07             |


**Note: This benchmark is general average of multiple runs of `examples\benchmark.py` resulted in. You may find different results when you test it in your own systems. 
Performance optimizations are ongoing** 

## 💽 Corpus Used for V2

cosmopedia-v2

c4-english

wikipedia

openwebtext

github-top-code

arxiv-papers

> Check dataset_training/train.py for more details

## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or optimizations.

## 📄 License

Apache-2.0



## Future Targets [3.0]

- [ ] Training a multilingual vocab 
- [ ] Better Compression
- Still Thinking


  


