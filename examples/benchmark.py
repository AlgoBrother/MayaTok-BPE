import time
from transformers import AutoTokenizer
import mayatok_bpe as bpe

# Load HF fast tokenizers
hf_falcon = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_fast=True)
hf_gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# Load MayaTok
my_tokenizer = bpe.PyBPETokenizer().load("bpe_tokenizer_py.json")

batch_config = bpe.PyBatchEncodingConfig(
    max_length=512,
    parallel_threshold=32,
    max_threads=16,
    use_thread_local_cache=True,
    thread_cache_size=10000
)

def benchmark_batch(name, encode_fn, texts):
    start = time.time()
    tokens_list = encode_fn()
    total_tokens = sum(len(t) for t in tokens_list)
    elapsed = time.time() - start

    # Compute compression ratios (skip empty tokens)
    valid_pairs = [
        (len(texts[i]), len(tokens_list[i]))
        for i in range(len(texts))
        if len(tokens_list[i]) > 0
    ]

    ratios = [text_len / token_len for text_len, token_len in valid_pairs]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

    print(f"{name}: {{'time_taken': {elapsed:.4f}, 'tokens_per_sec': {total_tokens/elapsed:.0f}, "
        f"'avg_tokens_per_text': {total_tokens/len(tokens_list):.2f}, "
        f"'avg_compression_ratio': {avg_ratio:.2f}}}")


# Load test data (you can use whatever dataset you wish)
with open("test_corpus.txt", "r", encoding="utf-8") as f: 
    texts = f.readlines()

print("Batch Encoding Benchmarks: ")
# Benchmark
benchmark_batch("MayaTok-BPE", lambda: my_tokenizer.encode_batch(texts, batch_config), texts)
benchmark_batch("Falcon-7B", lambda: hf_falcon(texts, add_special_tokens=False)["input_ids"], texts)
benchmark_batch("GPT2", lambda: hf_gpt2(texts, add_special_tokens=False)["input_ids"], texts)


print("Normal Encoding Benchmarks: ")
def compression_ratio(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(text) / len(tokens)

text = "I love transformers and tokenizers."
encoded_texts = my_tokenizer.encode(text) 

print("MayaTok-BPE", encoded_texts)
print("MayaTok-BPE:", my_tokenizer.decode(encoded_texts))
print("Falcon-7B:", hf_falcon.encode(text))
print("Falcon-7B:", hf_falcon.decode(hf_falcon.encode(text)))
print("GPT2:", hf_gpt2.encode(text))
print("GPT2:", hf_gpt2.decode(hf_gpt2.encode(text)))

print("MayaTok-BPE Compression: ", compression_ratio(text, my_tokenizer))
print("Falcon-7B Compression: ", compression_ratio(text, hf_falcon))
print("GPT2 Compression: ", compression_ratio(text, hf_gpt2))
