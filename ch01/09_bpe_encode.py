# Concept: encode the entire tiny-code corpus as BPE token IDs and save the array.
# Goal: create a compact binary dataset for downstream model training.
# The merge rules are loaded from codebot/merge_rules.pkl.
# Running this script replaces codebot/tiny_codes.bin.
# Resolve and enter the repository root from this script's location for stable relative data paths.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Add that root to sys.path, Python's import search list, so codebot can be imported.
sys.path.append('.')

# NumPy is a third-party numerical array library; here it creates a compact fixed-width integer array.
import numpy as np
from codebot.tokenizer import BPETokenizer


# Load the trained tokenizer and its lookup table from token IDs back to byte sequences.
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

# Read the whole corpus as str, then convert it to a Python list[int] of BPE token IDs.
# show_progress=True asks the tokenizer's tqdm wrapper to display encoding progress.
text = open("codebot/tiny_codes.txt").read()
ids = tokenizer.encode(text, show_progress=True)

# np.array converts list[int] to an ndarray; uint16 stores each ID in two bytes.
# ndarray.tofile writes raw values (no shape/dtype header) and replaces the existing binary corpus.
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("codebot/tiny_codes.bin")

# These Japanese output labels mean "number of token IDs" and "first 20 token IDs."
print(f"トークンID数: {len(ids_array)}")
print(f"最初の20個のトークンID: {ids_array[:20]}")

# ===== CONCEPT NOTES =====
# CONCEPT: Apply a persisted pretokenized BPE vocabulary to the full code corpus, then store the resulting IDs as unsigned 16-bit integers.
# GOAL: Turn the text corpus and trained rules into a binary token dataset for later model code; it follows vocabulary creation and inspection in the preceding scripts.
# INPUT: Reads the saved rules and all corpus text. open().read() returns str; encode receives it with show_progress=True.
# OUTPUT: Writes raw NumPy uint16 values to codebot/tiny_codes.bin, then prints total IDs and the first 20; values depend on corpus and rules.
# HOW INPUT BECOMES OUTPUT:
# 1. The script changes the working directory to the repository root and loads the tokenizer from the pickle.
# 2. tokenizer.encode splits special markers and pre-tokens, applies the learned merges, and returns a Python list of IDs while showing a progress bar.
# 3. np.array(list_of_ids, dtype=np.uint16) returns an ndarray; uint16 represents integers from 0 through 65,535.
# 4. ndarray.tofile writes raw values without a file header; len and slicing report the ID count and first 20 values.
# NOTES: Running this script overwrites codebot/tiny_codes.bin and may take time; it was not run during this comment-only edit. uint16 ends at 65,535, so larger IDs would overflow or fail conversion.
