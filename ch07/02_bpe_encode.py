# Concept: Apply the trained BPE rules to web-text files with parallel chunk encoding.
# Goal: Convert train and validation text to ordered binary token-ID arrays for downstream training.
# The encoder uses the merge-rule pickle created by the preceding script.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
# This driver delegates loading, tokenization, chunk multiprocessing, and binary assembly to the shared class.
from storybot.tokenizer import BPETokenizer


# Prevent corpus reads and output writes during import.
if __name__ == '__main__':
    # load_from unpickles the learned pair-to-ID mapping and rebuilds the token-ID-to-bytes vocabulary.
    tokenizer = BPETokenizer.load_from("webbot/merge_rules.pkl")

    # Each call writes one raw uint16 token-ID stream; chunks are encoded in parallel and joined in source order.
    tokenizer.encode_file(
        "webbot/owt_train.txt",
        "webbot/owt_train.bin", num_processes=8)

    tokenizer.encode_file(
        "webbot/owt_valid.txt",
        "webbot/owt_valid.bin", num_processes=8)

# ===== CONCEPT NOTES =====
# CONCEPT: This driver loads a learned BPE tokenizer and calls its parallel `encode_file` method, avoiding one enormous in-memory token list.
# GOAL: Follow vocabulary training by materializing token IDs for the web train and validation sets so downstream model code can consume them.
# INPUT: Loads `webbot/merge_rules.pkl`; reads `webbot/owt_train.txt` and `webbot/owt_valid.txt`; calls `encode_file` with 8 processes. Defaults are 64 chunks, cache directory `bpe_cache`, and end marker `<|endoftext|>`.
# OUTPUT: Writes raw `uint16` arrays to `webbot/owt_train.bin` and `webbot/owt_valid.bin`, not `.npy`-formatted files. Return counts are not printed. Temporary `chunk_*.npy` files are placed in `bpe_cache`, then that directory is removed; output paths are overwritten.
# HOW INPUT BECOMES OUTPUT:
# 1. `BPETokenizer.load_from` unpickles merge rules and reconstructs the vocabulary byte strings.
# 2. For each file, `encode_file` finds end-marker-aligned ranges and sends indexed tasks to a process pool.
# 3. Each `_encode_chunk` decodes its range, calls `encode` to split markers and pretokenize text, and writes token IDs as `uint16`; it returns a cache path and token count.
# 4. The parent sums counts, copies cache arrays in chunk order into a `uint16` memmap, flushes it, removes the cache, and returns total tokens.
# NOTES: The merge pickle and both `owt` text files must exist. Final binary files are overwritten; only temporary cache files are deleted.
