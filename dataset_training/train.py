import time
import os
import unicodedata
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

# NOTE: dataset proportions (GitHub ~27% of the mix) are left untouched.
# That's a modeling decision for what Maya should be good at, not a bug.
# Revisit if the goal is a general-purpose / STEM-tutoring tokenizer rather
# than a code-heavy one.

# ============================================================
# Config
# ============================================================
CHUNK_SIZE = 5_000  # examples processed between merge rounds
SAVE_FREQUENCY = 20  # checkpoint every 20 chunks (~200k examples)
CHECKPOINT_PATH = "/home/ashwin0010/coding_projects/DATASET/mayatok-checkpoint/M50.json"
OUTPUT_PATH = "/home/ashwin0010/coding_projects/DATASET/mayatok-checkpoint/M50-tokenizer.json"

# FIX #1: min_frequency raised from 2 -> 20. With 2, merges backed by only
# two occurrences across a 3.4M-doc corpus pollute the vocab with one-off
# URLs, hashes, UUID fragments, and typos. 20 is a reasonable floor for a
# 50k-vocab / 3.4M-doc run; tune down only if vocab utilization looks sparse.
INCREM_CONFIG = bpe.PyIncrementalTrainingConfig(
    vocab_size=50_000,
    min_frequency=20,
    chunk_size=CHUNK_SIZE,
    merge_frequency=50,  # TODO: confirm against mayatok's Rust train_stream
    # docstring/source whether this means "every 50 docs" or "50 merges per
    # chunk" -- semantics were unverifiable from this script alone.
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
# = 3,440,000


class ProgressTracker:
    """
    FIX #4: resume-aware ETA. Instead of faking start_time to avoid div/0
    (which made `rate` explode right after a resume), the tracker now tracks
    elapsed time only for the *current session* and reports remaining work
    against TOTAL_EXPECTED. ETA will be a bit optimistic for the first
    ~10-20k examples after a resume (rate hasn't stabilized yet), but it
    will never be nonsense-large like before.
    """

    def __init__(self, total, already_seen=0):
        self.total = total
        self.already_seen = already_seen
        self.start_time = time.time()
        self.last_print = already_seen

    def update(self, seen, vocab_size):
        now = time.time()
        if seen - self.last_print < 10_000:
            return
        self.last_print = seen

        session_seen = max(seen - self.already_seen, 1)
        elapsed = now - self.start_time
        rate = session_seen / elapsed if elapsed > 0 else 0
        remaining = self.total - seen
        eta_seconds = remaining / rate if rate > 0 else 0

        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_s = int(eta_seconds % 60)

        pct = (seen / self.total) * 100

        print(
            f"  \U0001F4C8 Progress: {seen:,} / {self.total:,} ({pct:.1f}%) | "
            f"vocab: {vocab_size:,} | "
            f"rate: {rate:.0f} ex/s | "
            f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        )


# ============================================================
# Text cleaning
# ============================================================

def clean_prose(text):
    # FIX #13: normalize Unicode so visually-identical characters in
    # different forms (NFC vs NFD etc.) don't become distinct tokens.
    return unicodedata.normalize("NFC", text)


def clean_code(text):
    # FIX #8: normalize line endings, strip trailing whitespace per line,
    # collapse runs of blank lines, on top of the existing tab expansion.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned_lines = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    return unicodedata.normalize("NFC", text)


def passes_length_filter(text, is_code):
    # FIX #9: character-length filtering killed valid short code snippets
    # (e.g. `printf("Hello")` is 15 chars). Use a word-count-based filter
    # for code, and keep a slightly more lenient char filter for prose.
    if is_code:
        return len(text.split()) >= 3
    return len(text.strip()) >= 20


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
        if entry.get("subset"):
            kwargs["name"] = entry["subset"]

        ds = load_dataset(entry["dataset"], **kwargs)
        print(f"  \u2705 Opened: {entry['name']} (max {entry['max_examples']:,})")
        return iter(ds)
    except Exception as e:
        print(f"  \u274c Failed to open {entry['name']}: {e}")
        return None


def make_dataset_generator(entry, ds_iter):
    """
    Wraps a dataset iterator into a generator that:
    - extracts the text field
    - cleans it (code vs prose)
    - filters short/empty docs
    - stops at max_examples
    - reports progress
    - FIX #14: wraps each document with <EOS> so the tokenizer never learns
      merges that span two unrelated documents.
    """
    text_field = entry["text_field"]
    max_ex = entry["max_examples"]
    name = entry["name"]
    is_code = entry["name"] == "github-top-code"
    count = 0

    for example in ds_iter:
        if count >= max_ex:
            break

        text = example.get(text_field, "")
        if not text:
            continue

        text = clean_code(text) if is_code else clean_prose(text)

        if not passes_length_filter(text, is_code):
            continue

        # space-prefix (GPT-2 style) + explicit document boundary marker
        yield " " + text + " <EOS>"

        count += 1
        if count % 50_000 == 0:
            print(f"    [{name}] streamed {count:,} / {max_ex:,}")

    print(f"  \u2705 Exhausted: {name} ({count:,} examples)")


def interleaved_stream(dataset_mix, skip=0):
    """
    Round-robin interleaving respecting interleave_weight.

    FIX #6: accepts `skip`, the number of already-processed examples from a
    previous run, and discards exactly that many from the *front* of the
    stream (in the same deterministic order they'd have been produced in)
    before yielding anything. This makes resume actually resume instead of
    silently re-training on the same first N examples every time.

    Caveat: this still has to re-open and re-iterate through the HF
    streaming datasets to reach the skip point -- it's correct, not free.
    For very large skip values this can take a while depending on your
    connection; it's still far cheaper than re-running merges on stale data.
    """
    print("\n\U0001F4C2 Opening dataset streams...")
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

    print(f"\n\u2705 {len(active)} datasets opened. Starting interleaved stream...\n")

    if skip > 0:
        print(f"  \u23e9 Skipping {skip:,} previously-processed examples to resume correctly...")

    total_yielded = 0
    skipped_so_far = 0
    while True:
        any_alive = False

        for source in active:
            if source["exhausted"]:
                continue

            pulled = 0
            while pulled < source["weight"]:
                try:
                    text = next(source["gen"])
                    total_yielded += 1
                    pulled += 1
                    any_alive = True

                    if skipped_so_far < skip:
                        skipped_so_far += 1
                        if skipped_so_far % 200_000 == 0:
                            print(f"    ...skipped {skipped_so_far:,} / {skip:,}")
                        continue

                    yield text
                except StopIteration:
                    source["exhausted"] = True
                    print(f"  \U0001F3C1 {source['name']} exhausted after {total_yielded:,} total")
                    break

        if not any_alive:
            print(f"\n\u2705 All datasets exhausted. Total yielded: {total_yielded:,}")
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
    print("\n\U0001F680 Starting MayaTok v2 training...")
    print(f"   vocab target:     {config.vocab_size:,}")
    print(f"   chunk size:       {config.chunk_size:,}")
    print(f"   merge frequency:  {config.merge_frequency}")
    print(f"   total expected:   {TOTAL_EXPECTED:,} examples")
    print(f"   checkpoint:       {config.checkpoint_path}")

    start = time.time()
    chunks_done = state.chunks_processed
    already_seen = chunks_done * CHUNK_SIZE
    tracker = ProgressTracker(TOTAL_EXPECTED, already_seen=already_seen)

    # FIX #6: pass already_seen through so interleaved_stream skips the
    # examples this run has already trained on, instead of restarting cold.
    stream = interleaved_stream(DATASET_MIX, skip=already_seen)

    for chunk, total_seen in buffered_chunks(stream, CHUNK_SIZE):
        if tokenizer.vocab_size >= config.vocab_size:
            print(f"\n\u2705 Vocab target reached.")
            break

        chunks_done += 1
        elapsed = time.time() - start

        print(
            f"\n\u25b6\ufe0f  Chunk #{chunks_done} | "
            f"examples seen (this session): {total_seen:,} | "
            f"vocab: {tokenizer.vocab_size:,} | "
            f"elapsed: {elapsed:.0f}s"
        )

        tokenizer.train_stream(chunk, config, state)
        # total_seen counts only *this session's* post-skip examples, so
        # add already_seen back to get the true corpus-wide position.
        tracker.update(already_seen + total_seen, tokenizer.vocab_size)

        if chunks_done % SAVE_FREQUENCY == 0:
            print(f"\U0001F4BE Saving checkpoint...")
            tokenizer.save_checkpoint(config.checkpoint_path, state, "json")

    print(f"\n\U0001F4BE Saving final checkpoint...")
    tokenizer.save_checkpoint(config.checkpoint_path, state, "json")

    elapsed = time.time() - start
    print(f"\n\u2705 Training complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"\U0001F4CA Final vocab size: {tokenizer.vocab_size:,}")


def print_stats(tokenizer):
    stats = tokenizer.get_stats()
    print(
        f"\n\U0001F4CA MayaTok v2 Stats:\n"
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
        print(f"\u2705 Resuming from checkpoint: {CHECKPOINT_PATH}")
        tokenizer, state = bpe.PyBPETokenizer.load_checkpoint(CHECKPOINT_PATH)
        print(f"   vocab so far: {tokenizer.vocab_size:,}")
        print(f"   chunks done:  {state.chunks_processed}")
    else:
        print("\U0001F195 Starting fresh training run...")
        tokenizer = bpe.PyBPETokenizer()
        state = bpe.IncrementalTrainingState()

    try:
        train(tokenizer, state, INCREM_CONFIG)
        print_stats(tokenizer)

        print(f"\n\U0001F4BE Saving final tokenizer to {OUTPUT_PATH}...")
        tokenizer.save(OUTPUT_PATH)
        print("\u2705 Saved!")

    except KeyboardInterrupt:
        print("\n\u26a0\ufe0f  Interrupted -- saving checkpoint...")
        tokenizer.save_checkpoint(CHECKPOINT_PATH, state, "json")
        print("\u2705 Checkpoint saved. Resume by running again.")

    except Exception as e:
        print(f"\n\u274c Training failed: {e}")
        raise


if __name__ == "__main__":
    main()