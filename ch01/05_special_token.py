# Concept: reserve a distinct token ID for a special end-of-text marker.
# Goal: keep document boundaries separate from ordinary byte-pair merges.
# The input is split around the marker during training and encoding.
# This extends the basic BPE tokenizer with a control token.
# defaultdict is from standard-library collections; int makes missing counts start at zero.
# re is Python's standard-library regular-expression module, used to retain the marker while splitting.
from collections import defaultdict
import re

# def count_pairs(ids):
#     counts = defaultdict(int)
#     for pair in zip(ids, ids[1:]):
#         counts[pair] += 1
#     return counts

# Accumulate counts into an optional shared mapping, so multiple segments can be counted together.
def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# Replace neighboring matches left-to-right without overlapping a just-merged pair.
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

# Learn ordinary BPE rules without allowing merges to cross a special document boundary.
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # str.split() returns list[str] and removes the delimiter; each segment is encoded independently.
    texts = input_text.split(end_token)
    ids_list = [list(text.encode("utf-8")) for text in texts]

    # Subtract 256 byte IDs and one dedicated marker ID; remaining vocabulary slots are merge IDs.
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in range(num_merges):
        # Add each segment's pair frequencies to one mapping; segments are never concatenated.
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        # Stop if every segment has no adjacent pair left.
        if not counts:
            break

        # Choose a most frequent pair; equal-frequency ties follow dictionary iteration order.
        best_pair = max(counts, key=counts.get)
        # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # Apply the rule separately to every segment, so no token spans the boundary.
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules

# Train on an in-memory sentence containing an explicit document boundary marker.
sample_text = "Hello world!<|endoftext|>This is BPE training."

merge_rules = train_bpe(sample_text, vocab_size=260)
print(merge_rules)  # {(105, 115): 256, (256, 32): 257, (105, 110): 258}


# Reserve one dedicated token ID for the marker; ordinary BPE rules never consume it.
class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        # Learned IDs occupy 256 through 255 + len(merge_rules); place the marker after them.
        self.end_token_id = 256 + len(merge_rules)

        # Each byte ID maps to its corresponding one-byte bytes object.
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            # Recover a merged token by concatenating its two earlier component byte strings.
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        # Store the marker spelling as UTF-8 bytes so decoding restores the readable marker.
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text):
        # Ordinary text only: Python str -> UTF-8 bytes -> integer byte IDs.
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text):
        # re.escape quotes punctuation for literal matching; parentheses capture the marker.
        # Standard-library re.split then retains each captured marker in the output list.
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        for text in texts:
            # A control token emits its single reserved ID, not IDs for its ordinary spelling.
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                ids = self._encode_text(text)
                all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        # All IDs, including the marker ID, map to bytes; join and decode the full UTF-8 stream.
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


tokenizer = BPETokenizer(merge_rules)

text = "Hello world!<|endoftext|>"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)

# ===== CONCEPT NOTES =====
# CONCEPT: Special-token-aware BPE treats the end marker as a dedicated ID instead of learning it from ordinary UTF-8 bytes.
# GOAL: Show how to train over text segments separated by <|endoftext|> and preserve that marker during round-trip encoding; it builds on basic BPE and leads toward the pretokenized trainer next.
# INPUT: train_bpe receives "Hello world!<|endoftext|>This is BPE training." with vocab_size=260. The tokenizer example encodes "Hello world!<|endoftext|>" using the three rules printed by training.
# OUTPUT: The training print is {(105, 115): 256, (256, 32): 257, (105, 110): 258}. For the tokenizer example, ordinary bytes are [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33], followed by end-token ID 259; decode returns Hello world!<|endoftext|>.
# HOW INPUT BECOMES OUTPUT:
# 1. train_bpe splits on the marker and converts each ordinary segment separately to UTF-8 byte IDs, so no pair crosses the boundary.
# 2. It performs vocab_size - 256 - 1 merge iterations; the final reserved slot is excluded from learned merges.
# 3. BPETokenizer sets end_token_id to 256 + len(merge_rules), which is 259 for the three displayed rules, and maps it to the marker's UTF-8 bytes.
# 4. encode splits while retaining the marker, appends the special ID for that segment, and applies BPE only to ordinary text; decode concatenates the mapped bytes.
# NOTES: Every exact occurrence of the delimiter is special. re.escape protects marker punctuation, and the capturing parentheses preserve the marker in re.split output. This example writes no files.
