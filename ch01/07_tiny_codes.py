# Concept: train a larger BPE vocabulary on the repository's tiny-code corpus.
# Goal: persist learned merge rules so later scripts can inspect and use them.
# The training input is read from codebot/tiny_codes.txt.
# Running this script replaces codebot/merge_rules.pkl.
# os.path helpers resolve the repository root from this script's __file__, regardless of launch directory.
# chdir changes the process working directory so the relative data paths below resolve there.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# sys.path is Python's import search list; add the root so the codebot package can be found.
sys.path.append('.')

# pickle is Python's standard-library object serializer; its binary format is Python-specific.
import pickle
from codebot.tokenizer import train_bpe

# Target total vocabulary size, including base byte IDs and learned tokens.
vocab_size = 1000  # target number of token IDs
# open() returns a file object; read() returns its entire contents as a Python str.
text = open("codebot/tiny_codes.txt").read()
# The shared trainer returns a dict mapping each adjacent pair to the new ID assigned to its merge.
merge_rules = train_bpe(text, vocab_size)

# Save the learned rules; write-binary mode truncates/replaces the existing file.
with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)

# ===== CONCEPT NOTES =====
# CONCEPT: This script invokes the shared codebot BPE trainer on a code corpus and serializes the learned merge rules with pickle.
# GOAL: Prepare a reusable tokenizer vocabulary for the inspection/compression script and the later corpus-encoding script; it applies the tokenizer concepts from the preceding lessons to code text.
# INPUT: Reads codebot/tiny_codes.txt as text and calls train_bpe(text, vocab_size=1000). open().read() produces str.
# OUTPUT: It writes a binary pickle containing merge_rules to codebot/merge_rules.pkl. The script contains no print calls.
# HOW INPUT BECOMES OUTPUT:
# 1. os.chdir uses __file__ to select the repository root; sys.path.append('.') lets Python import codebot there.
# 2. open(...).read() loads codebot/tiny_codes.txt into text.
# 3. codebot.tokenizer.train_bpe learns up to the requested size and returns a Python dict of pair -> new-ID rules.
# 4. pickle.dump(file, object) serializes the dict into the open binary file for a later loader to reconstruct.
# NOTES: Running this script overwrites codebot/merge_rules.pkl. Training may take time and displays tqdm progress; this script was not run during this comment-only edit.
