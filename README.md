# MayaTok
MayaTok is a blazing-fast, multithreaded Byte-Pair Encoding (BPE) tokenizer written in Rust. Built with performance and extensibility in mind, it’s ideal for anyone working on custom LLM pipelines, research, or production-ready NLP infrastructure.

⚡️ Features

🧠 Custom BPE tokenizer from scratch — no wrappers

🧵 Multithreaded training for fast vocab generation

🗃️ Persistent vocab/merges (just like GPT-style tokenizers)

🏎️ Focus on raw speed — built for performance benchmarking

📦 Minimal dependencies, zero bloat

📦 Installation

Make sure you have Rust installed.

```
git clone let me copy paste
cd mayatok-bpe
maturin build --release
```

This will generate the optimized binary in target/release/mayatok.

## Corpus Used for V1

Cosmopedia

OpenWebText

## 🙌 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or optimizations.

## 📄 License

Apache-2.0



## Future Targets
Making it pip installable.

More faster merging and better CPU usage

