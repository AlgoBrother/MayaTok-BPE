import time
import mayatok as bpe
from transformers import AutoTokenizer
import tiktoken

# ======= Load tokenizers =======
tokenizer = bpe.PyBPETokenizer.load(
    r"path\to\tokenizer.json"
)
=======
# Load tiktoken (cl100k_base is GPT-4's tokenizer, p50k_base is GPT-2/3's)
tiktoken_cl100k = tiktoken.get_encoding("cl100k_base")
tiktoken_p50k = tiktoken.get_encoding("p50k_base")





# Load HF fast tokenizers
hf_falcon = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_fast=True)
hf_gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# Load MayaTok
my_tokenizer = bpe.PyBPETokenizer().load(r"bpe_tokenizer_py.json") 

>>>>>>> bc0711d6f3f244e68d3e58fabbbd55ae72b69b8b
batch_config = bpe.PyBatchEncodingConfig(
    max_length=512, parallel_threshold=32,
    max_threads=16, use_thread_local_cache=True, thread_cache_size=10000,
)

hf_gpt2    = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
hf_falcon  = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_fast=True)
hf_llama   = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)
tiktoken_cl100k = tiktoken.get_encoding("cl100k_base")
tiktoken_p50k   = tiktoken.get_encoding("p50k_base")

# warmup tiktoken
_ = tiktoken_cl100k.encode_batch(["warmup"] * 5)
_ = tiktoken_p50k.encode_batch(["warmup"] * 5)

# === Load corpus =
with open(r"examples\test_corpus.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# ============== Benchmark helpers ===================
def bench_batch(label, fn, pad_id=None):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start

    if pad_id is not None:
        real = [[t for t in seq if t != pad_id] for seq in result]
    else:
        real = result

    total = sum(len(t) for t in real)
    valid = [(len(texts[i]), len(real[i])) for i in range(len(result)) if real[i]]
    ratio = sum(c / t for c, t in valid) / len(valid) if valid else 0
    print(f"  {label:<25} time={elapsed:.4f}s  tok/s={total/elapsed:>12,.0f}  "
          f"avg_tok={total/len(result):>6.2f}  compression={ratio:.2f}")

def bench_single(label, fn, pad_id=None):
    start = time.perf_counter()
    result = [fn(t) for t in texts]
    elapsed = time.perf_counter() - start

    if pad_id is not None:
        real = [[t for t in seq if t != pad_id] for seq in result]
    else:
        real = result

    total = sum(len(t) for t in real)
    valid = [(len(texts[i]), len(real[i])) for i in range(len(result)) if real[i]]
    ratio = sum(c / t for c, t in valid) / len(valid) if valid else 0
    print(f"  {label:<25} time={elapsed:.4f}s  tok/s={total/elapsed:>12,.0f}  "
          f"avg_tok={total/len(result):>6.2f}  compression={ratio:.2f}")

# =========== Batch benchmarks ================
print("\n=============== Batch encode ================")
bench_batch("MayaTok",
    lambda: tokenizer.encode_batch(texts, config=batch_config))
bench_batch("GPT-2 (HF)",
    lambda: hf_gpt2(texts, add_special_tokens=False)["input_ids"])
bench_batch("Falcon-7B (HF)",
    lambda: hf_falcon(texts, add_special_tokens=False)["input_ids"])
bench_batch("LLaMA (HF)",
    lambda: hf_llama(texts, add_special_tokens=False)["input_ids"])
bench_batch("tiktoken-cl100k",
    lambda: tiktoken_cl100k.encode_batch(texts))
bench_batch("tiktoken-p50k",
    lambda: tiktoken_p50k.encode_batch(texts))

# === Single encode benchmarks =================================================
print("\n=============== Single encode ===============")
bench_single("MayaTok",
    lambda t: tokenizer.encode(t))
bench_single("GPT-2 (HF)",
    lambda t: hf_gpt2(t, add_special_tokens=False)["input_ids"])
bench_single("Falcon-7B (HF)",
    lambda t: hf_falcon(t, add_special_tokens=False)["input_ids"])
bench_single("LLaMA (HF)",
    lambda t: hf_llama(t, add_special_tokens=False)["input_ids"])
bench_single("tiktoken-cl100k",
    lambda t: tiktoken_cl100k.encode(t))
bench_single("tiktoken-p50k",
    lambda t: tiktoken_p50k.encode(t))

# =============== Compression (single text) ===============
print("\n=============== Compression (single text) ===============")
TEXT = "I love transformers and tokenizers."

def compress(ids): return f"{len(TEXT)/len(ids):.3f} chars/token"

print(f"  MayaTok:        {compress(tokenizer.encode(TEXT))}")
print(f"  GPT-2:          {compress(hf_gpt2.encode(TEXT))}")
print(f"  Falcon-7B:      {compress(hf_falcon.encode(TEXT))}")
print(f"  tiktoken-cl100k:{compress(tiktoken_cl100k.encode(TEXT))}")
print(f"  tiktoken-p50k:  {compress(tiktoken_p50k.encode(TEXT))}")


print("\n=============== Tokenization Test ===============")
ids = tokenizer.encode(TEXT)
print(f"  input:   {TEXT!r}")
print(f"  ids:     {ids}")
print(f"  decoded: {tokenizer.decode(ids)!r}")

# =============== Stats ===============
print("\n=============== Tokenizer stats ===============")
stats = tokenizer.get_stats()
print(f"  MayaTok vocab:     {stats.vocab_size:,}")
print(f"  MayaTok merges:    {stats.num_merges:,}")
print(f"  GPT-2 vocab:       {hf_gpt2.vocab_size:,}")
print(f"  Falcon vocab:      {hf_falcon.vocab_size:,}")
print(f"  tiktoken-cl100k:  {tiktoken_cl100k.n_vocab:,}")
print(f"  tiktoken-p50k:      {tiktoken_p50k.n_vocab:,}")
