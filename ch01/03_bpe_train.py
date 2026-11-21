# Concept: learn byte-pair merge rules by repeatedly choosing the most frequent adjacent pair.
# Goal: demonstrate how a byte vocabulary grows through data-driven merges.
# The helpers count pairs and merge matching neighbors left to right.
# The learned rules are consumed by the BPE tokenizer in the next lesson.
# defaultdict comes from the standard-library collections module; int makes missing counts start at zero.
from collections import defaultdict

# Count adjacent token-ID pairs; pairs overlap, so [1, 2, 3] contains (1, 2) and (2, 3).
# Input is a sequence of integers; output maps each pair tuple to its frequency.
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# Example: the repeated adjacent pair (1, 2) occurs twice in this sequence.
ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
print(counts)  # {(1, 2): 2, (2, 3): 1, (3, 1): 1}

# Replace every left-to-right, non-overlapping occurrence of pair with new_id.
# This makes one longer token while preserving the original sequence when expanded.
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    # Scan once; moving by two after a match prevents reusing either ID in the merged pair.
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids

# Example: merge both non-overlapping occurrences of (1, 2) into ID 4.
ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
print(merged)  # [4, 3, 4]

# Train Byte Pair Encoding (BPE): repeatedly replace the most frequent adjacent pair with one ID.
# The 256 base IDs represent all possible byte values; each merge adds a vocabulary entry.
def train_bpe(text, vocab_size):
    # str.encode() returns UTF-8 bytes; list(bytes) converts those bytes into integer IDs 0..255.
    ids = list(text.encode("utf-8"))

    # Reserve one new token ID per requested merge beyond the 256 base byte IDs.
    num_merges = vocab_size - 256  # the initial byte vocabulary has 256 entries
    merge_rules = {}

    for step in range(num_merges):
        # Recount after each merge because replacement changes neighboring IDs and pair frequencies.
        counts = count_pairs(ids)

        # Stop early if fewer than two IDs remain, leaving no adjacent pair to learn.
        if not counts:
            break

        # Select a pair with maximum frequency; ties follow this mapping's iteration order.
        best_pair = max(counts, key=counts.get)
        # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))

        # New IDs start at 256, immediately after the byte-ID range.
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # Apply the replacement to the full sequence before counting pairs again.
        ids = merge(ids, best_pair, new_id)

    return merge_rules

# Example: request four merge slots beyond the 256 byte tokens.
text = "Hello world! This is BPE training."
merge_rules = train_bpe(text, vocab_size=260)
print(merge_rules)  # {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# ===== CONCEPT NOTES =====
# CONCEPT: Byte Pair Encoding (BPE) builds a vocabulary by replacing the most frequent neighboring token pair with one new token ID. A token is a unit the tokenizer emits; an ID is its integer label.
# GOAL: Build a small ordered merge-rule dictionary from UTF-8 bytes; the following lesson applies those learned rules for encoding and decoding.
# INPUT: count_pairs and merge take integer ID sequences, e.g. [1, 2, 3, 1, 2]. train_bpe takes the example text "Hello world! This is BPE training." and vocab_size=260.
# OUTPUT: The helper example counts {(1, 2): 2, (2, 3): 1, (3, 1): 1}; merging pair (1, 2) as ID 4 yields [4, 3, 4]. The training comment gives {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}.
# HOW INPUT BECOMES OUTPUT:
# 0. collections.defaultdict(int) is a mapping whose missing pair counts start at zero.
# 1. train_bpe encodes the text as UTF-8 and initializes its IDs from the byte values.
# 2. Python built-in zip pairs each ID with the next: [1, 2, 3] -> (1, 2), (2, 3); defaultdict counts them.
# 3. max(counts, key=counts.get) selects a most-frequent pair; the code assigns it the next ID, starting at 256.
# 4. merge scans left to right, replaces non-overlapping occurrences with that ID, and the resulting sequence is counted again.
# NOTES: vocab_size=260 requests four merge iterations beyond 256 byte IDs, unless no pair remains. Applying a rule is left-to-right and non-overlapping. Equal-frequency ties follow dictionary order. This example writes no files.
