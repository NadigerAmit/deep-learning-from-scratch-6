# Concept: format one instruction/response example and inspect its BPE token IDs.
# Goal: make the supervised fine-tuning text layout and tokenization visible.
# This is a preparation/demo step for ch03/03_sft.py, not a training script.
# It intentionally processes only the first JSON example.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
from codebot.tokenizer import BPETokenizer

# Load the BPE tokenizer; merge rules map text pieces to vocabulary IDs.
# BPETokenizer.load_from restores merge rules serialized with Python's pickle format.
tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

# Read instruction/response records from JSON into Python lists and dictionaries.
with open('codebot/tiny_codes_sft.json') as f:
    # json.load parses JSON into a list of dictionaries with instruction/response fields.
    data = json.load(f)

# Inspect only the first example to make the data format concrete.
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# Format the example with section labels separating request from expected answer.
# These literal labels are the functional chat template; <|endoftext|> marks the answer’s end.
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# Encode formatted text as integer IDs for the model.
# BPETokenizer.encode returns a Python list of vocabulary IDs.
ids = tokenizer.encode(text)
print(ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]

# ===== CONCEPT NOTES =====
# CONCEPT: Instruction tuning represents an example as a prompt followed by a
# response and an end-of-text marker, then encodes that string into token IDs.
# GOAL: Preview the exact Alpaca-style sample layout used by ch03/03_sft.py, which
# later uses prompt labels masked out so loss is applied to response tokens.
# INPUT: codebot/tiny_codes_sft.json, codebot/merge_rules.pkl, and data[0] only.
# The prompt uses "### Instruction:" and "### Response:" markers.
# OUTPUT: Prints the first item, its formatted text, and the resulting list of IDs;
# nothing is saved and there are no training hyperparameters in this file.
# HOW INPUT BECOMES OUTPUT:
# 1. BPETokenizer.load_from reads the learned byte-pair merge rules.
# 2. json.load reads the instruction/response examples and data[0] selects one.
# 3. The f-string adds section labels and <|endoftext|> after the response.
# 4. tokenizer.encode splits/merges the text into token IDs, which are printed.
# NOTES: This is only an inspection example; it neither pads/truncates sequences
# nor creates training labels. On CPU it performs a small amount of preprocessing;
# no GPU is used, no model is loaded, and no files are overwritten.
