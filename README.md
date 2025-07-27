# MayaTok
MayaTok is a Byte-Pair Encoding (BPE) tokenizer written in Rust. Built with performance and extensibility in mind. I made this project just because I wanted to study how Byte Pair Encoding Works. 

> Version: **V1** (under development, more optimizations & better token compression coming soon!)

## ⚡️ Features (More optimizations in Progress)
 
- Multithreaded training for fast vocab generation 

- Persistent merges 

- Checkpoint saving

- Focus on raw speed — built for performance benchmarking



Make sure you have Rust installed. If not, [Install Rust](https://www.rust-lang.org/tools/install)

```bash
git clone https://github.com/AlgoBrother/MayaTok-BPE.git
cd mayatok-bpe

```

Use maturin for building wheels.

```bash
pip install maturin
maturin build --release
```

This will generate the optimized binary in target/release/mayatok.

## 📥 Download Pretrained Vocab

Download the trained vocab and merges: 

**Note: If you wish to make your own Vocab please see:** 

> Make sure you have forked/cloned the rust tokenizer code and have built the /target/wheels as mentioned in previous steps

[stream method](examples/train_your_own_vocab.py) - Designed to use streamed data from one local machine and use it on your training machine

[non-stream method](examples/non_stream_train_your_own_vocab.py) - If you have a dataset which your RAM can handle after being loaded, use this. much faster traininhg

### Using Curl

```bash
curl -O https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json
```

### Using Invoke-WebRequest 

```bash
Invoke-WebRequest -Uri https://huggingface.co/datasets/AlgoBrother/mayatok-assets/resolve/main/bpe_tokenizer_py.json -OutFile bpe_tokenizer_py.json
```

## Using with Python

To use MayaTok with python. 

```bash
pip install target/wheels/<PATH>
```
> Replace PATH with the file version you have downloaded. Locate under target\wheels\ and copy and paste its path after ```pip install```

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

Output of the sample code above
```
[732, 21843, 345, 535, 576, 335, 725]
Hello , world !
```

## 📈 Benchmarks

### Batch Encoding

| Tokenizer   | Tokens/sec | Avg Compression Ratio |
| ----------- | ---------- | --------------------- |
| MayaTok-BPE | 59,066     | 2.44                  |
| Falcon-7B   | 784,577    | 3.26                  |
| GPT2        | 1,116,972  | 2.94                  |

### Normal Encoding

| Tokenizer   | Tokens/sec | Compression Ratio |
| ----------- | ---------- | ----------------- |
| MayaTok-BPE | 5,471      | 2.19              |
| Falcon-7B   | 186,368    | 4.38              |
| GPT2        | 266,627    | 4.38              |


## 💽 Corpus Used for V1

[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)

[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)


## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or optimizations.

## 📄 License

Apache-2.0



## Future Targets
- Making it global pip installable.

- the `examples` folder has lot of python implementation. Will experiment to integrate this in rust side of code and make python side of code more smaller and easier for users. Apologies for any current complexities.
  
- Faster merging and better CPU usage

- Better merges. As the creator of this project, I will agree I am not fully satisfied with the current merge file and aim to improve it in future updates.

- bincode support for faster loading.



