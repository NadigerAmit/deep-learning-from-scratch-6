# Concept: apply learned BPE merges to UTF-8 bytes and reverse token IDs back to text.
# Goal: connect a merge-rule dictionary to a usable encode/decode tokenizer.
# Each merged ID is associated with the bytes of its two component IDs.
# The next lesson extends BPE to preserve a special end-of-text token.
# The standard-library collections.defaultdict is used by this local pair-counting helper.
from collections import defaultdict

# Count overlapping neighboring ID pairs; for [1, 2, 3], the result has (1, 2) and (2, 3).
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# Apply one BPE rule from left to right, merging non-overlapping occurrences into new_id.
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


# A vocabulary maps token IDs to byte strings; merge_rules records how new IDs were formed.
# Build the reverse ID-to-bytes table so decoding can reconstruct the original byte stream.
class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # Seed the byte vocabulary: ID i maps to exactly one byte whose numeric value is i.
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # Each merged token is the concatenation of its components; IDs for H and e become bytes b'He'.
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # Count entries in the complete ID-to-bytes vocabulary, including base and merged tokens.
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        # Convert str -> UTF-8 bytes -> list[int]; each starting ID is one byte value.
        ids = list(text.encode("utf-8"))

        # Apply rules in training order because later rules may use IDs made by earlier ones.
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids

    def decode(self, ids):
        # Look up each integer ID and obtain bytes; an unknown ID raises KeyError.
        byte_list = [self.id_to_bytes[i] for i in ids]

        # bytes.join concatenates fragments without separators, restoring the original byte sequence.
        combined_bytes = b"".join(byte_list)

        # bytes.decode converts UTF-8 bytes to str; replace inserts U+FFFD for malformed sequences.
        text = combined_bytes.decode("utf-8", errors="replace")
        return text

# Toy learned rules: each key is an adjacent pair; each value is the new token ID.
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# Create the tokenizer and its reverse lookup table.
tokenizer = BPETokenizer(merge_rules)

# Encode and decode the sample to check the round trip.
text = "Hello世界😁"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)  # [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]
print(decoded)  # Restores "Hello" plus the Japanese word for "world" and the emoji.

# ===== CONCEPT NOTES =====
# CONCEPT: BPE repeatedly replaces common neighboring tokens with one ID. This tokenizer applies ordered rules and stores each ID's original bytes so encoding can be reversed.
# GOAL: Demonstrate a complete BPE round trip; the next lesson reserves an explicit ID for a special end-of-text marker.
# INPUT: The rules are {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}; sample text is "Hello" plus the Japanese word for "world" and an emoji.
# OUTPUT: IDs are [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]; decode restores the original text.
# HOW INPUT BECOMES OUTPUT:
# 0. Python str.encode('utf-8') returns bytes; list(bytes) exposes each byte as an integer.
# 1. __init__ maps base IDs 0–255 to their one-byte sequences, then concatenates component byte sequences for every merged ID.
# 2. encode turns the sample into UTF-8 byte IDs; the first rule merges the bytes for 'H' and 'e' (72, 101) into ID 259.
# 3. The remaining learned rules are applied in training order; unmatched bytes remain unchanged.
# 4. decode looks up each ID's bytes, joins the bytes, and decodes the result as UTF-8.
# NOTES: Rule order matters because later pairs can contain IDs created earlier. bytes.decode(errors='replace') substitutes a replacement character for malformed UTF-8 instead of raising. This script writes no files.
