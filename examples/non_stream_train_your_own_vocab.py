import mayatok_bpe as bpe

# Non-streaming BPE Tokenizer Training Example (Recommended for smaller datasets)

def train_vocab():
    print("\n--- Training (Non Stream) Starts ---")

    # 1. Initialize tokenizer
    tokenizer = bpe.PyBPETokenizer()
    print(f"Initialized tokenizer. Initial vocab size: {tokenizer.vocab_size}")
    print(tokenizer.print_vocab)

    file_path = "your_path.txt"  # Path to your training corpus file
    
    with open(file_path, "r", encoding="utf-8") as f:
        file_lines = [line.strip() for line in f if line.strip()]


    # 2. Define training corpus (this can be empty too. I added some words for demonstration)
    training_corpus = [
        "low", "lower", "newest", "widest", "lowest", "new", "wide", "newly",
        "brown fox jumps", "brown cows", "the quick brown fox",
        "jumps over the lazy dog", "quick brown fox", "the quick brown",
        "lazy dog", "apple pie", "apple tree", "sweet apple", "sweet potato",
        "banana bread", "banana split", "hot dog", "hot potato", "Hello World!",
        "Rustaceans are cool", "Rust is fast", "Rust is safe",
        "tokenization in Rust", "subword tokenization", "performance optimization",
        "memory safety", "concurrency in Rust", "great programming language",
        "artificial intelligence", "machine learning", "natural language processing",
    ]
    training_corpus.extend(file_lines)  # Extend it with your corpus data

    # 3. Define training config
    train_config = bpe.PyTrainConfig(
    vocab_size=10000,                     # Final vocabulary size
    min_frequency=2,                      # Minimum frequency for a token to be included
    n_threads=8,                          # Number of CPU threads to use for training
    special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"],  # Reserved tokens for special purposes
    show_progress=True                    # Whether to display training progress in the console
    )
    
    # 4. Train the tokenizer
    print("\nStarting training...")
    tokenizer.train(training_corpus, train_config)
    print(f"Training complete. Final vocab size: {tokenizer.vocab_size}")
    print(f"Total merges: {tokenizer.num_merges}")

    # 5. Get and print stats
    stats = tokenizer.get_stats()
    print(f"\nBPE Stats:\n{stats.vocab_size=}\n{stats.num_merges=}\n{stats.num_special_tokens=}\n{stats.num_character_tokens=}")


# <----- Optional Code for Encoding/Decoding and Batch Encoding ----->
# Uncomment the following section if you want to run encoding/decoding and batch encoding demos
#     # 6. Encode and Decode 
#     texts_to_encode = [
#         "newest",
#         "lowest",
#         "brown fox",
#         "Rustaceans",
#         "tokenization",
#         "unseenword",
#         "Hello World!",
#         "the quick brown fox jumps",
#         "applepie",
#         "hotdog",
#         "programming language",
#         "artificial intelligence",
#         "Rust is fast and safe",
#         "memory safety and concurrency",
#     ]

#     print("\n--- Encoding/Decoding Demo ---")
#     for text in texts_to_encode:
#         try:
#             encoded_ids = tokenizer.encode(text)
#             print(f"\nText: '{text}'")
#             print(f"  Encoded IDs: {encoded_ids}")
#             decoded_text = tokenizer.decode(encoded_ids)
#             print(f"  Decoded Text: '{decoded_text}'")
#             # Simple check for single words (after normalization)
#             if ' ' not in text:
#                 pass
#         except Exception as e:
#             print(f"Error encoding/decoding '{text}': {e}")

#     # 7. Encode Batch
#     print("\n--- Batch Encoding Demo ---")
#     batch_config = bpe.PyBatchEncodingConfig(
#     max_length=512,                  # max number of tokens in a sequence (adjust as needed)
#     parallel_threshold=32,          # if input batch size ≥ 32, use parallel encoding
#     max_threads=4,                  # max number of threads to use
#     use_thread_local_cache=True,    # enables per-thread caching for faster repeated tokenization
#     thread_cache_size=10000         # max number of cache entries per thread
# )



#     try:
#         batch_encoded_ids = tokenizer.encode_batch(texts_to_encode, batch_config)
#         for i, ids in enumerate(batch_encoded_ids):
#             print(f"Batch Text: '{texts_to_encode[i]}'")
#             print(f"  Batch Encoded IDs: {ids}")
#             decoded_text = tokenizer.decode(ids)
#             print(f"  Batch Decoded Text: '{decoded_text}'")
#     except Exception as e:
#         print(f"Error in batch encoding: {e}")

    # 8. Save and Load
    save_path = "bpe_maya_py.json" # Output path for the final tokenizer, you can change this as needed
    print(f"\nSaving tokenizer to '{save_path}'...")
    try:
        tokenizer.save(save_path)
        print("Tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")

    print(f"Loading tokenizer from '{save_path}'...")
    try:
        loaded_tokenizer = bpe.PyBPETokenizer.load(save_path)
        print(f"Tokenizer loaded successfully. Loaded vocab size: {loaded_tokenizer.vocab_size}")

        test_text = "Hello world! __ -- what @ dicebwiub @3 73 &6 \u205F"

        original_ids = tokenizer.encode(test_text)
        loaded_ids = loaded_tokenizer.encode(test_text)
        
        print(f"\nTest text: '{test_text}'")
        print(f"  Original Tokenizer IDs: {original_ids}")
        print(f"  Loaded Tokenizer IDs: {loaded_ids}")
        
        assert original_ids == loaded_ids, "Encoded IDs should match after load!"
        print("Encoding matches after load!")

        decoded_loaded = loaded_tokenizer.decode(loaded_ids)
        print(f"  Decoded from loaded: '{decoded_loaded}'")

    except Exception as e:
        print(f"Error loading tokenizer: {e}")

if __name__ == "__main__":
    train_vocab()
