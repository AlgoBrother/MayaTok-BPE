# MayaTok
MayaTok is a Byte-Pair Encoding (BPE) tokenizer written in Rust. Built with performance and extensibility in mind. I made this project just because I wanted to study how Byte Pair Encoding Works. 

> Version: **V1** (under development, more optimizations & improved token compression coming soon!)

## ⚡️ Features (More optimizations in Progress)
 
- Multithreaded training for fast vocab generation 

- Persistent merges 

- Checkpoint saving

- Focus on raw speed — built for performance benchmarking


## 🚀 Installation

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (required)
- Python 3.7+ (for Python bindings)

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


## 📥 Download Pretrained Vocab

Download the trained vocab and merges:

### Quick Start (Recommended)

#### Using Curl

```bash
curl -O https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json
```

#### Using Invoke-WebRequest 

```bash
Invoke-WebRequest -Uri https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json -OutFile bpe_tokenizer_py.json
```

### Create your own Vocab 

> If you wish to create your own vocab file with a different corpus file.
> 
> Make sure you have forked/cloned the rust tokenizer code and have built the /target/wheels as mentioned in previous steps

[stream method](examples/train_your_own_vocab.py) - If you have a large dataset and want to stream your data in chunks to not overload your machine. Use this.

[non-stream method](examples/non_stream_train_your_own_vocab.py) - If you have a dataset which your RAM can handle after being loaded, use this for much faster training.



## Using with Python

To use MayaTok with Python:

```python
import mayatok_bpe as bpe

my_tokenizer = bpe.PyBPETokenizer.load("bpe_tokenizer_py.json")
test = "Hello, world!"
tokens = my_tokenizer.encode(test)
print(tokens)
decoded_text = my_tokenizer.decode(tokens)
print(decoded_text)
```

Output of the sample code above
```
[732, 21843, 345, 535, 576, 335, 725]
Hello , world !
```

## 📈 Benchmarks

### Batch Encoding

| Tokenizer   | Tokens/sec | Avg Compression Ratio |
| ----------- | ---------- | --------------------- |
| **MayaTok-BPE** | **59,066**     | **2.44**                  |
| Falcon-7B   | 784,577    | 3.26                  |
| GPT2        | 1,116,972  | 2.94                  |

### Normal Encoding

| Tokenizer   | Tokens/sec | Compression Ratio |
| ----------- | ---------- | ----------------- |
| **MayaTok-BPE** | **5,471**      | **2.19**              |
| Falcon-7B   | 186,368    | 4.38              |
| GPT2        | 266,627    | 4.38              |

**Note: Performance optimizations are ongoing**

## 💽 Corpus Used for V1

[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)

[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)


## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or optimizations.

## 📄 License

Apache-2.0



## Future Targets

- [ ] PyPI package distribution
      
- [ ] the `examples` folder has lot of python implementation. Will experiment to integrate this in rust side of code and make python side of code more smaller and easier for users.
      
- [ ] Enhanced CPU utilization and faster merging algorithms
      
- [ ] Improved merge quality and compression ratios
      
- [ ] Bincode support for faster model loading
      


  


