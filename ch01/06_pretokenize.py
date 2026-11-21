# Concept: constrain BPE merges to regex-defined text pre-tokens.
# Goal: demonstrate GPT-2-style pretokenization before byte-pair learning.
# The trainer also splits around an explicit end-of-text marker.
# This produces more linguistically structured merge units than raw text BPE.
# defaultdict comes from collections; unseen pair counts begin at zero.
# regex is a third-party Unicode-aware regex module supporting properties such as \p{L} and \p{N}.
# tqdm wraps an iterable with a progress display while yielding the same values.
from collections import defaultdict
import regex as re
from tqdm import tqdm


# A pre-token is a chunk BPE may merge internally but must not cross.
def pretokenize(text):
    # The third-party regex pattern groups contractions, Unicode letter runs (\p{L}), number runs (\p{N}), punctuation, and whitespace.
    # A leading optional space stays attached to many words; alternatives preserve useful text boundaries.
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # regex.findall returns list[str] with every non-overlapping match in source order.
    return re.findall(pattern, text)

# Count neighboring IDs; an optional shared defaultdict aggregates counts across pre-token sequences.
def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# Replace non-overlapping occurrences of one learned pair, scanning from left to right.
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

# Learn BPE only within Unicode-aware pre-tokens and never across the document marker.
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # Step 1: split at the marker so a pair cannot bridge document segments.
    texts = input_text.split(end_token)

    # Step 2: split each segment into pre-tokens before converting chunks into byte IDs.
    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text):  # Respect regex-defined text boundaries.
            # UTF-8 encode each chunk separately; each list[int] is an independent BPE sequence.
            ids_list.append(list(pretoken.encode("utf-8")))

    # Leave one vocabulary slot for the end marker; the other new slots are learned merges.
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    # tqdm wraps range, showing progress while still yielding each integer step.
    for step in tqdm(range(num_merges), desc="Training BPE"):
        # Count pairs across all sequences, but never create pairs spanning two sequences.
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        if not counts:
            break

        # max(..., key=counts.get) returns the pair with the greatest frequency.
        best_pair = max(counts, key=counts.get)
        # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # Apply the learned merge inside each sequence, never between neighboring sequences.
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules


# This tokenizer respects the same pre-token boundaries during encoding as during training.
class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text):
        # Convert one ordinary pre-token from str to UTF-8 byte IDs.
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text, show_progress=False):
        # Escape the marker for literal matching; capturing parentheses retain it in regex.split.
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # Optional progress display for segments; False avoids constructing the tqdm wrapper.
        texts = tqdm(texts, desc="Encoding") if show_progress else texts

        for text in texts:
            if text == self.end_token:
                # A special marker becomes its one reserved ID, not IDs of its spelling.
                all_ids.append(self.end_token_id)
            else:
                # Encode each pre-token independently, then append its IDs in original order.
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

# Train the pre-token-aware vocabulary from the in-memory sample.
sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"

merge_rules = train_bpe(sample_text, vocab_size=270)
tokenizer = BPETokenizer(merge_rules)

# Encode the sample and decode it to check the round trip.
text = "Say hello!"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)

# Decode each ID separately to inspect the bytes represented by individual tokens.
for token_id in ids:
    print(f"{token_id} -> '{tokenizer.decode([token_id])}'")

# ===== CONCEPT NOTES =====
# CONCEPT: The GPT-2-style regex first divides text into pre-tokens; BPE training and encoding then merge bytes only within each pre-token.
# GOAL: Extend special-token-aware BPE with lexical/punctuation/whitespace boundaries; this makes tokenization more structured than merging across an entire text segment.
# INPUT: pretokenize accepts text and uses a Unicode-aware regex. The training sample is "Say hello! Why hello? Just hello.<|endoftext|>Good morning!" with vocab_size=270; the encoding example is "Say hello!".
# OUTPUT: Training displays a tqdm progress bar labeled "Training BPE" and produces merge_rules. The final prints show the sample's ID list, its decoded text Say hello!, and one line per ID in the form "<id> -> '<decoded token>'"; actual learned IDs depend on the input and merge counts.
# HOW INPUT BECOMES OUTPUT:
# 1. train_bpe splits the sample at <|endoftext|> so no ordinary pair crosses a document boundary.
# 2. pretokenize uses the Unicode regex to find text pieces; each piece is encoded separately as UTF-8 bytes and stored as one ID sequence.
# 3. Each training iteration counts pairs across sequences, chooses the most frequent pair, assigns an ID beginning at 256, and merges it within every sequence.
# 4. encode splits while retaining the special marker, maps that marker to its dedicated ID, and applies BPE independently to each pre-token.
# 5. decode concatenates each ID's bytes; the final loop decodes each individual ID to illustrate token contents.
# NOTES: Requires third-party regex and tqdm. regex supports Unicode properties \p{L}/\p{N}, unlike standard-library re. Encoding progress defaults off, while training always uses tqdm. One token's bytes may not form valid UTF-8 alone; decode(errors='replace') substitutes malformed parts. This example writes no files.
