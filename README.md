# MayaTok
MayaTok is a blazing-fast, multithreaded Byte-Pair Encoding (BPE) tokenizer written in Rust. Built with performance and extensibility in mind, it’s ideal for anyone working on custom LLM pipelines, research, or production-ready NLP infrastructure.

This is version 1 (V1)

## ⚡️ Features

Custom BPE tokenizer from scratch
 
Multithreaded training for fast vocab generation 

Persistent merges 

Checkpoint saving

Focus on raw speed — built for performance benchmarking



Make sure you have Rust installed. If not, [Install Rust](https://www.rust-lang.org/tools/install)

```bash
git clone https://github.com/AlgoBrother/MayaTok-BPE.git
cd mayatok-bpe
maturin build --release
```

This will generate the optimized binary in target/release/mayatok.

## 📥 Download Pretrained Vocab

Download the trained vocab and merges:

### Using Curl

```bash
curl -O https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json
```

### Using Invoke-WebReques 

```bash
Invoke-WebRequest -Uri https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json -OutFile bpe_tokenizer_py.json
```

## Corpus Used for V1

Cosmopedia

OpenWebText

## Using with Python

To use MayaTok with python. 

```bash
pip install target/wheels/mayatok_bpe-<VERSION>-cp310-cp310-win_amd64.whl
```
Replace VERSION with the file version you have downloaded. Locate under target\wheels\ and copy and paste its path after ```pip install```

Now you can use mayatok in your project :)

```python
import mayatok_bpe as bpe

my_tokenizer = bpe.PyBPETokenizer.load("bpe_tokenizer_py.json")
test = "Hello, world!"
tokens = my_tokenizer.encode(test)
print(tokens)
decoded_text = my_tokenizer.decode(tokens)
print(decoded_text)
```


## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or optimizations.

## 📄 License

Apache-2.0



## Future Targets
Making it pip installable.

More faster merging and better CPU usage

