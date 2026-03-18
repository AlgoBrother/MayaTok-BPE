import os
import requests
import time
import mayatok_bpe as bpe
from urllib.parse import urlparse

# --- Data Streaming Functions ---
# I have added some emojis for easier readability in terminal. You can remove them if you wish to


def is_url(path):
    # Check if the given path is a URL or local file path.
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def stream_local_file(file_path, chunk_size=8192):
    # Streams lines from a local file efficiently to avoid loading entire file into RAM.   
    try:
        print(f"✅ Connected to local file: {file_path}")
        with open(file_path, 'r', encoding='utf-8', buffering=chunk_size) as file:
            line_count = 0
            for line in file:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                if line_count > 0 and line_count % 10000 == 0:
                    print(f"    ... read {line_count} lines from local file ...")
                yield line
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"❌ Error reading local file: {e}")
        raise

def stream_remote_url(url):
    # Streams lines from a remote text corpus.
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        print(f"✅ Connected to remote stream: {url}")
        for i, raw in enumerate(response.iter_lines(decode_unicode=True)):
            if not raw:
                continue
            line = raw.strip()
            if i > 0 and i % 10000 == 0:
                print(f"    ... received {i} lines from remote stream ...")
            yield line

def stream(source):
    # Universal streaming function that handles both local files and remote URLs.
    if is_url(source):
        print(f"🌐 Detected remote URL: {source}")
        yield from stream_remote_url(source)
    else:
        print(f"📁 Detected local file: {source}")
        yield from stream_local_file(source)

def buffered_chunks(source, chunk_size=10000):
    # Yields batches of lines (buffered chunks) from the stream.
    buf = []
    total = 0
    source_type = "remote" if is_url(source) else "local"
    
    for line in stream(source):
        buf.append(line)
        total += 1
        if len(buf) >= chunk_size:
            print(f"📦 Yielding full chunk of {len(buf)} lines from {source_type} source (total seen: {total})")
            yield buf
            buf = []
    if buf:
        print(f"📦 Yielding final chunk of {len(buf)} lines from {source_type} source (total seen: {total})")
        yield buf

# --- Helper Functions for File Management ---

def get_file_size_info(source):
    # Get file size information for progress tracking.
    if not is_url(source) and os.path.exists(source):
        size_bytes = os.path.getsize(source)
        size_mb = size_bytes / (1024 * 1024)
        print(f"📊 Local file size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        return size_bytes, size_mb
    else:
        print("📊 Remote source - size unknown")
        return None, None

# --- Main Training Logic ---

def train_tokenizer(tokenizer, state, config, source):
    # Runs the main training loop, saving checkpoints periodically.
    print("🚀 Starting streaming incremental training...")
    
    # Display source information
    get_file_size_info(source)
    
    start_time = time.time()
    
    # Get current chunk count from the loaded state
    chunks = state.chunks_processed

    for chunk in buffered_chunks(source, chunk_size=config.chunk_size):
        if tokenizer.vocab_size >= config.vocab_size:
            print(f"✅ Target vocab size {config.vocab_size} reached. Stopping training.")
            break

        print(f"▶️ Training on chunk #{chunks + 1}...")
        tokenizer.train_stream(chunk, config, state)
        chunks += 1

        if tokenizer.vocab_size >= config.vocab_size:
            print(f"✅ Target vocab size {config.vocab_size} reached after training chunk #{chunks}.")
            break

        if chunks > 0 and chunks % config.save_frequency == 0:
            print(f"💾 Saving checkpoint after {chunks} chunks...")
            tokenizer.save_checkpoint(config.checkpoint_path, state, "json")
            print("    ... checkpoint saved.")

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.2f}s ({elapsed/60:.2f}m).")
    print(f"📊 Final vocab size: {tokenizer.vocab_size}")

def print_stats(tokenizer):
    # Prints final stats of the trained tokenizer.
    stats = tokenizer.get_stats()
    print(
        "\n📊 BPE Stats:\n"
        f"  vocab_size = {stats.vocab_size}\n"
        f"  num_merges = {stats.num_merges}\n"
        f"  num_special_tokens = {stats.num_special_tokens}\n"
        f"  num_character_tokens = {stats.num_character_tokens}"
    )

def main():
    # --- Configuration ---
    # You can now use either a local file path or a remote URL
    # Examples:
    # source = "large_corpus.txt"  # Local file
    # source = "https://example.com/corpus.txt"  # Remote URL
    # source = "/path/to/very/large/dataset.txt"  # Absolute local path
    
    source = ""  # Change this to your actual file path or URL

    increm_train_config = bpe.PyIncrementalTrainingConfig(
        vocab_size=20000, # Final vocabulary size
        min_frequency=100,  # Increased for higher quality merges
        chunk_size=50000, # Increased for more stable stats
        merge_frequency=800, # Decreased frequency for better performance
        show_progress=True,
        special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"],
        save_frequency=10,  # Checkpoint every 10 chunks
        checkpoint_path="samePC_checkpoint.json" # Path to save checkpoints
    )

    # Uncomment if you want to use batch encoding demo
    # batch_config = bpe.PyBatchEncodingConfig(
    #     max_length=512,     # max number of tokens in a sequence (adjust as needed)
    #     parallel_threshold=32, # if input batch size ≥ 32, use parallel encoding
    #     max_threads=16, # Number of CPU threads for parallel encoding
    #     use_thread_local_cache=True, # Enables per-thread caching for faster repeated tokenization
    #     thread_cache_size=10000 # Size of the thread-local cache (adjust as needed)
    # )

    # --- Checkpoint Loading ---
    if os.path.exists(increm_train_config.checkpoint_path):
        print(f"✅ Found checkpoint at '{increm_train_config.checkpoint_path}'. Resuming training...")
        tokenizer, state = bpe.PyBPETokenizer.load_checkpoint(increm_train_config.checkpoint_path)
    else:
        print("No checkpoint found. Starting a new training run...")
        tokenizer = bpe.PyBPETokenizer()
        state = bpe.IncrementalTrainingState()

    # --- Validate Source ---
    if not is_url(source) and not os.path.exists(source):
        print(f"Error: Local file '{source}' does not exist.")
        print("Please check the file path or provide a valid URL.")
        return

    # --- Run Training and Demo ---
    try:
        train_tokenizer(tokenizer, state, increm_train_config, source)
        print_stats(tokenizer)

        examples = ["newest", "lowest", "brown fox", "Rustaceans", "a high-performance tokenizer"]
        
        # Uncomment to run encoding/decoding demo (Optional)
        # print("\n--- Encoding/Decoding Demo ---")
        # enc = tokenizer.encode_batch(examples, batch_config)
        # for ex, ids in zip(examples, enc):
        #     print(f"Original: {ex}")
        #     print(f"Tokens:   {ids[:10]}{'…' if len(ids)>10 else ''}")
        #     print(f"Decoded:  {tokenizer.decode(ids)}\n")

        out = "bpe_tokenizer_py.json" # Output path for the final tokenizer
        print(f"💾 Saving final tokenizer to '{out}'…")
        tokenizer.save(out)
        print("Tokenizer Saved!!!✅")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
    print("🎉 All done!")
