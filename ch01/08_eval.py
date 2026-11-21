# Concept: inspect learned BPE tokens and measure compression on a code sample.
# Goal: evaluate the vocabulary produced by ch01/07_tiny_codes.py.
# It also includes an optional, commented tiktoken comparison.
# This script reads the saved merge rules and corpus but writes no files.
# Resolve and enter the repository root so relative corpus and vocabulary paths work from any launch directory.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Add the root to Python's import search path so the codebot package can be imported.
sys.path.append('.')

# Import the reusable tokenizer and load its serialized vocabulary.
from codebot.tokenizer import BPETokenizer


# load_from reads the pickle and builds the tokenizer's ID-to-bytes mapping.
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

# This Japanese display label means "first 10 learned"; leave the program output unchanged.
print("最初に学習された10個:")
for token_id in range(256, 266):
    # Look up an ID's bytes and decode them to str for display; strict UTF-8 can fail on a partial character.
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")

print("\n最後に学習された10個:")
for token_id in range(990, 1000):
    # Look up an ID's bytes and decode them to str for display; strict UTF-8 can fail on a partial character.
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")


# Measure on the first 10,000 Python characters (not necessarily 10,000 bytes).
# open().read() returns str; slicing selects a prefix without modifying the source file.
sample_text = open("codebot/tiny_codes.txt").read()[:10000]  # first 10,000 characters

# str.encode('utf-8') returns bytes; len counts bytes, so multibyte characters contribute several.
byte_count = len(sample_text.encode("utf-8"))
# encode returns a Python list[int] of token IDs, each representing one or more original bytes.
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
# Input UTF-8 bytes / output tokens; a value above 1 means each token represents multiple bytes on average.
compression_ratio = byte_count / ids_count

# These Japanese display labels mean "compression efficiency," "byte count," "token count," and "compression ratio."
print("\n=== 圧縮効率 ===")
print(f"バイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍（平均 {compression_ratio:.2f} バイト/トークン）")


# Optional comparison against GPT-family tokenizers using the third-party tiktoken package (inactive below).
# Its output labels describe vocabulary size and compression ratio; the Japanese strings remain unchanged as data.
"""
import tiktoken

text = open("codebot/tiny_codes.txt").read()[:10000]
byte_count = len(text.encode("utf-8"))

for name, encoding_name in [('GPT-2', 'gpt2'), ('cl100k_base', 'cl100k_base')]:
    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text, allowed_special={'<|endoftext|>'}))
    ratio = byte_count / token_count
    print(f"{name}: 語彙サイズ {encoding.n_vocab:,}, 圧縮率 {ratio:.2f}倍")
"""

# ===== CONCEPT NOTES =====
# CONCEPT: Evaluate a saved BPE vocabulary by displaying selected token byte sequences and comparing UTF-8 byte count with encoded token count.
# GOAL: Observe what the training script produced and quantify its compression; the later encoding script applies this vocabulary to save the full corpus as token IDs.
# INPUT: Reads codebot/merge_rules.pkl through BPETokenizer.load_from and codebot/tiny_codes.txt. The measurement uses only the first 10,000 characters; the inspection loops cover IDs 256–265 and 990–999.
# OUTPUT: Prints ten early-token lines and ten late-token lines in the form "ID n: 'text'", then reports byte count, token count, and compression ratio to two decimal places. Exact counts and token strings depend on the files on disk. The tiktoken comparison is inside a triple-quoted block and does not run as written.
# HOW INPUT BECOMES OUTPUT:
# 1. BPETokenizer.load_from unpickles the learned rules and builds the ID-to-bytes table.
# 2. For each selected ID, the script looks up tokenizer.id_to_bytes and decodes that byte sequence as UTF-8 for display.
# 3. It reads the first 10,000 text characters, counts their UTF-8 bytes, and encodes them with tokenizer.encode.
# 4. It divides byte_count by ids_count and formats both counts with commas and the ratio with two decimal places.
# NOTES: Requires the pickle from ch01/07_tiny_codes.py. Per-token UTF-8 decoding is strict and can raise UnicodeDecodeError when one token holds only part of a multibyte character. Division assumes the sample encodes to at least one ID. This script reads but does not write files; the tiktoken block is triple-quoted and inactive.
