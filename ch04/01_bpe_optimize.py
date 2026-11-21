# Concept: Train byte-pair encoding (BPE) by counting adjacent byte-ID pairs.
# Goal: Learn merge rules from the code corpus as the baseline for later speedups.
# Each step rescans every distinct pretoken sequence and weights counts by corpus frequency.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

# `collections.defaultdict(int)` is a standard-library frequency map: missing keys start at 0.
# `collections.Counter` is an alternative frequency map, but this code uses defaultdict; it does not use
# functools.cache/memoisation or import time for timing.
from collections import defaultdict
import regex as re
from tqdm import tqdm


def pretokenize(text):
    # `regex` (the third-party package imported as `re`) recognizes Unicode letter/number classes.
    # `findall` returns a list[str] of matching substrings; pretokens are text pieces, not yet vocabulary IDs.
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

def count_pairs(ids, weight=1, counts=None):
    # Count adjacent integer-ID pairs. For ids=[97, 98, 97], zip(ids, ids[1:]) gives
    # (97,98) and (98,97). `weight` multiplies counts when this distinct sequence repeats in the corpus.
    # The `defaultdict(int)` result maps tuple[int, int] -> int and takes O(len(ids)) time per sequence.
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts

def merge(ids, pair, new_id):
    # Replace all non-overlapping copies of one adjacent pair. Example: [97,98,97] with
    # pair=(97,98) and new_id=256 becomes [256,97]. Advancing by two on a match prevents overlap;
    # one call is O(number of IDs). The returned list is a new sequence; the input is unchanged.
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

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # Split on the special marker first so it is kept separate from ordinary text and is not learned as a byte merge.
    texts = input_text.split(end_token)

    # Count occurrences of each regex-defined pretoken; repeated strings are represented once plus a frequency.
    pretoken_counts = defaultdict(int)
    # `tqdm` wraps an iterable and reports progress without changing the strings it yields.
    for text in tqdm(texts, desc="Pretokenizing"):
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1

    # Encode each pretoken as UTF-8 bytes: each byte value 0..255 is a starting token ID.
    # Tuples are immutable/hashable dictionary keys; their values are the corpus frequencies.
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    # Reserve IDs 0..255 for bytes and one ID for the end marker; each merge adds one vocabulary entry.
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        # Recount all pairs from scratch; this visits every current sequence and costs O(total IDs).
        # Multiplying by `count` makes one scan represent every occurrence of that sequence.
        # A fresh mapping is required because each merge changes every sequence; this simple baseline
        # recounts all neighboring pairs on every round, which can dominate runtime.
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)

        # An empty mapping means no sequence has two neighboring IDs left to merge.
        if not pair_counts:
            break

        # BPE training picks the pair with the largest corpus-wide count; higher numeric pair IDs win ties.
        # Each selected pair gets a fresh token ID, recording the merge order.
        # best_pair = max(pair_counts, key=pair_counts.get)
        # The tuple key makes the tie-break rule explicit and reproducible.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # Apply the rule to every sequence and combine frequencies when merged results are identical.
        # Rebuilding the whole mapping each round is simple but expensive for large corpora.
        new_ids_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            new_ids = merge(ids, best_pair, new_id)  # Replace non-overlapping occurrences in this sequence.
            new_ids_counts[tuple(new_ids)] += count
        ids_counts = new_ids_counts

    return merge_rules


vocab_size = 1000  # Capacity: 256 byte IDs + 1 marker ID + at most 743 learned merge IDs.
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(text, vocab_size)

# ===== CONCEPT NOTES =====
# CONCEPT: Repeatedly replace the most frequent adjacent byte pair with a new token ID.
# GOAL: Establish the weighted full-rescan training approach that the following caching, chunking, and parallel examples optimize.
# INPUT: Reads `codebot/tiny_codes.txt`; calls `train_bpe(text, vocab_size=1000)` with default end marker `<|endoftext|>`. `pretokenize` applies the file's regex after splitting on that marker.
# OUTPUT: Returns a mapping from selected `(id1, id2)` pairs to new IDs, assigned to `merge_rules` but neither printed nor saved. `tqdm` shows preprocessing/training progress. At most 743 merges are requested (`1000 - 256 - 1`); training can stop when no pairs remain.
# HOW INPUT BECOMES OUTPUT:
# 1. `train_bpe` splits the corpus on `end_token`; `pretokenize` counts each piece in `pretoken_counts`.
# 2. Pretokens become UTF-8 byte-ID tuples in `ids_counts`; for example, illustrative `aba` becomes `(97, 98, 97)`.
# 3. Each iteration calls `count_pairs` weighted by sequence counts and chooses the most frequent pair; pair-ID values break ties.
# 4. The pair receives ID `256 + step`; `merge` replaces its non-overlapping occurrences in every sequence, and the next full pass recounts pairs.
# NOTES: The repository-relative corpus file must exist. Recounting and merging every sequence each step can be slow. This script does not persist `merge_rules`.
