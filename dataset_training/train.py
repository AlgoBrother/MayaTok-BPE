import time
import os
import mayatok as bpe
from datasets import load_dataset



# ============================================================
# Dataset mix for MayaTok v1.1
# ============================================================

DATASET_MIX = [
    {
        "name": "cosmopedia-v2",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "subset": "cosmopedia-v2",
        "split": "train",
        "text_field": "text",
        "max_examples": 1_500_000,
        "interleave_weight": 4,
    },
    {
        "name": "c4-english",
        "dataset": "allenai/c4",
        "subset": "en",
        "split": "train",
        "text_field": "text",
        "max_examples": 500_000,
        "interleave_weight": 3,
    },
    {
        "name": "wikipedia",
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_field": "text",
        "max_examples": 200_000,
        "interleave_weight": 2,
    },
    {
        "name": "openwebtext",
        "dataset": "Skylion007/openwebtext",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "max_examples": 200_000,
        "interleave_weight": 2,
    },
    {
        "name": "github-top-code",
        "dataset": "ronantakizawa/github-top-code",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "max_examples": 940_000,
        "interleave_weight": 2,
    },
    {
        "name": "arxiv-papers",
        "dataset": "CShorten/ML-ArXiv-Papers",
        "subset": None,
        "split": "train",
        "text_field": "abstract",
        "max_examples": 100_000,
        "interleave_weight": 1,
    },
]

# ============================================================
# Config
# ============================================================
CHUNK_SIZE = 10_000  # how many examples to process between merges (lower = more merges, slower but better vocab)
SAVE_FREQUENCY = 20 # Save checkpoint every 20 chunks (200k examples)
CHECKPOINT_PATH = "./checkpoints/mayatok_v2_checkpoint.json" # or just choose a directory to store and access your checkpoint
OUTPUT_PATH = "./checkpoints/mayatok_v2_output_2.json"

INCREM_CONFIG = bpe.PyIncrementalTrainingConfig(
    vocab_size=100_000,
    min_frequency=2,
    chunk_size=CHUNK_SIZE,
    merge_frequency=50, 
    show_progress=True,
    special_tokens=[
    "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>",
    "<|endofcode|>",   # marks end of code blocks
    "<|code|>",        # marks start of code blocks  
    "<|system|>",      # for instruction tuning later
    "<|user|>",        # for instruction tuning later
    "<|assistant|>",   # for instruction tuning later
    ],
    save_frequency=SAVE_FREQUENCY,
    checkpoint_path=CHECKPOINT_PATH,
)


TOTAL_EXPECTED = sum(entry["max_examples"] for entry in DATASET_MIX)
# = 1_500_000 + 500_000 + 200_000 + 940_000 + 100_000 = 3_440_000

class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.start_time = time.time()
        self.last_print = 0

    def update(self, seen, vocab_size):
        now = time.time()
        # only print every 10k examples to avoid spam
        if seen - self.last_print < 10_000:
            return
        self.last_print = seen

        elapsed = now - self.start_time
        rate = seen / elapsed if elapsed > 0 else 0
        remaining = self.total - seen
        eta_seconds = remaining / rate if rate > 0 else 0

        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_s = int(eta_seconds % 60)

        pct = (seen / self.total) * 100

        print(
            f"  📈 Progress: {seen:,} / {self.total:,} ({pct:.1f}%) | "
            f"vocab: {vocab_size:,} | "
            f"rate: {rate:.0f} ex/s | "
            f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        )


# ============================================================
# Streaming
# ============================================================

def open_dataset_stream(entry):
    try:
        kwargs = dict(
            split=entry["split"],
            streaming=True,
            trust_remote_code=True,
        )
        if entry.get("subset"):  # only pass subset if not None
            kwargs["name"] = entry["subset"]
            
        ds = load_dataset(entry["dataset"], **kwargs)
        print(f"  ✅ Opened: {entry['name']} (max {entry['max_examples']:,})")
        return iter(ds)
    except Exception as e:
        print(f"  ❌ Failed to open {entry['name']}: {e}")
        return None


def make_dataset_generator(entry, ds_iter):
    """
    Wraps a dataset iterator into a generator that:
    - extracts the text field
    - filters short/empty docs
    - stops at max_examples
    - reports progress
    """
    text_field = entry["text_field"]
    max_ex = entry["max_examples"]
    name = entry["name"]
    count = 0

    for example in ds_iter:
        if count >= max_ex:
            break
        is_code = entry["name"] == "github-top-code"

        text = example.get(text_field, "")
        if is_code:
            text = text.replace('\t', '    ')
        if not text or len(text.strip()) < 20:
            continue
        text = text.strip()
        yield " " + text if text else None  # space-prefix to help BPE learn common word starts

        count += 1
        if count % 50_000 == 0:
            print(f"    [{name}] streamed {count:,} / {max_ex:,}")

    print(f"  ✅ Exhausted: {name} ({count:,} examples)")


def interleaved_stream(dataset_mix):
    """
    Round-robin interleaving respecting interleave_weight.
    
    Each cycle pulls `weight` examples from each dataset in order.
    Datasets that exhaust early are dropped from the cycle.
    Stops when all datasets are exhausted.

    Example with weights [4, 2, 1]:
    cycle: cosmo cosmo cosmo cosmo | c4 c4 | wiki | cosmo cosmo ...
    """
    # Open all streams
    print("\n📂 Opening dataset streams...")
    active = []
    for entry in dataset_mix:
        ds_iter = open_dataset_stream(entry)
        if ds_iter is None:
            continue
        gen = make_dataset_generator(entry, ds_iter)
        active.append({
            "name": entry["name"],
            "gen": gen,
            "weight": entry["interleave_weight"],
            "exhausted": False,
        })

    print(f"\n✅ {len(active)} datasets opened. Starting interleaved stream...\n")

    total_yielded = 0
    while True:
        any_alive = False

        for source in active:
            if source["exhausted"]:
                continue

            # pull `weight` examples from this source per cycle
            pulled = 0
            while pulled < source["weight"]:
                try:
                    text = next(source["gen"])
                    yield text
                    total_yielded += 1
                    pulled += 1
                    any_alive = True
                except StopIteration:
                    source["exhausted"] = True
                    print(f"  🏁 {source['name']} exhausted after {total_yielded:,} total")
                    break

        if not any_alive:
            print(f"\n✅ All datasets exhausted. Total yielded: {total_yielded:,}")
            break


def buffered_chunks(stream, chunk_size):
    buf = []
    total = 0
    for text in stream:
        buf.append(text)
        total += 1
        if len(buf) >= chunk_size:
            yield buf, total
            buf = []
    if buf:
        yield buf, total

# ============================================================
# Training
# ============================================================

def train(tokenizer, state, config):
    print("\n🚀 Starting MayaTok v2 training...")
    print(f"   vocab target:     {config.vocab_size:,}")
    print(f"   chunk size:       {config.chunk_size:,}")
    print(f"   merge frequency:  {config.merge_frequency}")
    print(f"   total expected:   {TOTAL_EXPECTED:,} examples")
    print(f"   checkpoint:       {config.checkpoint_path}")

    start = time.time()
    chunks_done = state.chunks_processed
    tracker = ProgressTracker(TOTAL_EXPECTED)

    # fast-forward tracker if resuming
    already_seen = chunks_done * CHUNK_SIZE
    tracker.start_time = time.time() - 1  # avoid div/0
    tracker.last_print = already_seen

    stream = interleaved_stream(DATASET_MIX)

    for chunk, total_seen in buffered_chunks(stream, CHUNK_SIZE):
        if tokenizer.vocab_size >= config.vocab_size:
            print(f"\n✅ Vocab target reached.")
            break

        chunks_done += 1
        elapsed = time.time() - start

        print(
            f"\n▶️  Chunk #{chunks_done} | "
            f"examples seen: {total_seen:,} | "
            f"vocab: {tokenizer.vocab_size:,} | "
            f"elapsed: {elapsed:.0f}s"
        )

        tokenizer.train_stream(chunk, config, state)
        tracker.update(total_seen, tokenizer.vocab_size)

        if chunks_done % SAVE_FREQUENCY == 0:
            print(f"💾 Saving checkpoint...")
            tokenizer.save_checkpoint(config.checkpoint_path, state, "json")

    print(f"\n💾 Saving final checkpoint...")
    tokenizer.save_checkpoint(config.checkpoint_path, state, "json")

    elapsed = time.time() - start
    print(f"\n✅ Training complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"📊 Final vocab size: {tokenizer.vocab_size:,}")


def print_stats(tokenizer):
    stats = tokenizer.get_stats()
    print(
        f"\n📊 MayaTok v2 Stats:\n"
        f"   vocab_size:           {stats.vocab_size:,}\n"
        f"   num_merges:           {stats.num_merges:,}\n"
        f"   num_special_tokens:   {stats.num_special_tokens}\n"
        f"   num_character_tokens: {stats.num_character_tokens}"
    )

# ============================================================
# Main
# ============================================================

def main():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming from checkpoint: {CHECKPOINT_PATH}")
        tokenizer, state = bpe.PyBPETokenizer.load_checkpoint(CHECKPOINT_PATH)
        print(f"   vocab so far: {tokenizer.vocab_size:,}")
        print(f"   chunks done:  {state.chunks_processed}")
    else:
        print("🆕 Starting fresh training run...")
        tokenizer = bpe.PyBPETokenizer()
        state = bpe.IncrementalTrainingState()

    try:
        train(tokenizer, state, INCREM_CONFIG)
        print_stats(tokenizer)

        print(f"\n💾 Saving final tokenizer to {OUTPUT_PATH}...")
        tokenizer.save(OUTPUT_PATH)
        print("✅ Saved!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — saving checkpoint...")
        tokenizer.save_checkpoint(CHECKPOINT_PATH, state, "json")
        print("✅ Checkpoint saved. Resume by running again.")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
