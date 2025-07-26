import os
import requests
import time
import mayatok_bpe as bpe

# --- Data Streaming Functions ---
# I have added some emojis for easier readability in terminal. You can remove them if you wish to

def stream(url):
    """Streams lines from a remote text corpus."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        print("✅ Connected to stream")
        for i, raw in enumerate(response.iter_lines(decode_unicode=True)):
            if not raw:
                continue
            line = raw.strip()
            if i > 0 and i % 10000 == 0:
                print(f"    ... received {i} lines ...")
            yield line

def buffered_chunks(url, chunk_size=10000):
    """Yields batches of lines (buffered chunks) from the stream."""
    buf = []
    total = 0
    for line in stream(url):
        buf.append(line)
        total += 1
        if len(buf) >= chunk_size:
            print(f"📦 Yielding full chunk of {len(buf)} lines (total seen: {total})")
            yield buf
            buf = []
    if buf:
        print(f"📦 Yielding final chunk of {len(buf)} lines (total seen: {total})")
        yield buf

# --- Main Training Logic ---

def train_tokenizer(tokenizer, state, config, url):
    """Runs the main training loop, saving checkpoints periodically."""
    print("🚀 Starting streaming incremental training...")
    start_time = time.time()
    
    # Get current chunk count from the loaded state
    chunks = state.chunks_processed

    for chunk in buffered_chunks(url, chunk_size=config.chunk_size):
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
    """Prints final stats of the trained tokenizer."""
    stats = tokenizer.get_stats()
    print(
        "\n📊 BPE Stats:\n"
        f"  vocab_size = {stats.vocab_size}\n"
        f"  num_merges = {stats.num_merges}\n"
        f"  num_special_tokens = {stats.num_special_tokens}\n"
        f"  num_character_tokens = {stats.num_character_tokens}"
    )

def main():
    # --- Configuration (Fully adjustible as per your requirements) ---
    url = "your dataset URL here"  # Replace with your actual dataset URL

    increm_train_config = bpe.PyIncrementalTrainingConfig(
      vocab_size=50000,          # Final vocabulary size to train towards
      min_frequency=100,         # Minimum frequency for a token/merge to be kept (filters out rare tokens)
      chunk_size=50000,          # Number of text lines per training chunk (affects how often merges update)
      merge_frequency=800,       # How often merges happen during training (lower = more frequent merges)
      show_progress=True,        # Whether to print training progress in the console
      special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"], # Tokens reserved for NLP tasks
      save_frequency=10,         # Save a checkpoint every 10 chunks
      checkpoint_path="checkpoint.json" # Where to store the checkpoint file (It will automatically create this file if it does not exist.)
    )

  # Uncomment it if you want to do batch encoding demo
   # batch_config = bpe.PyBatchEncodingConfig(
   #    max_length=512,              # Maximum token length per sequence (longer sequences will be truncated)
   #    parallel_threshold=32,       # If batch size >= this value, encoding will run in parallel threads
   #    max_threads=16,              # Maximum number of threads to use for parallel encoding
   #    use_thread_local_cache=True, # Enables caching inside threads to speed up repeated encoding
   #    thread_cache_size=10000      # Number of cache entries per thread for fast token lookup
   # )


    # --- Checkpoint Loading ---
    if os.path.exists(increm_train_config.checkpoint_path):
        print(f"✅ Found checkpoint at '{increm_train_config.checkpoint_path}'. Resuming training...")
        tokenizer, state = bpe.PyBPETokenizer.load_checkpoint(increm_train_config.checkpoint_path)
    else:
        print("🚀 No checkpoint found. Starting a new training run...")
        tokenizer = bpe.PyBPETokenizer()
        state = bpe.IncrementalTrainingState()

    # --- Run Training and Demo ---
    train_tokenizer(tokenizer, state, increm_train_config, url)
    print_stats(tokenizer)


    
    # Uncomment to run encoding/decoding demo (Optional)
    # examples = ["newest", "lowest", "brown fox", "Rustaceans", "a high-performance tokenizer"]
    
    # print("\n--- 🧪 Encoding/Decoding Demo ---")
    # enc = tokenizer.encode_batch(examples, batch_config)
    # for ex, ids in zip(examples, enc):
    #     print(f"Original: {ex}")
    #     print(f"Tokens:   {ids[:10]}{'…' if len(ids)>10 else ''}")
    #     print(f"Decoded:  {tokenizer.decode(ids)}\n")

    out = "bpe_tokenizer_py.json" # Output path for the final tokenizer, you can change this as needed
    print(f"💾 Saving final tokenizer to '{out}'")
    tokenizer.save(out)
    print("✅ Saved.")

if __name__ == "__main__":
    main()
    print("🎉 All done!")
