# Concept: Speed up BPE training by caching which byte sequences contain each pair.
# Goal: Update counts only for sequences affected by the selected merge.
# The code-corpus setup matches the baseline so the cache optimization is clear.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

# `collections.defaultdict(int)` provides zero-default integer counts; `defaultdict(set)` creates
# membership sets on first use. These are explicit data structures, not functools memoisation.
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
    # tqdm wraps the text list to report progress and yields the original str items.
    for text in tqdm(texts, desc="Pretokenizing"):
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1

    # Encode each pretoken as UTF-8 bytes: each byte value 0..255 is a starting token ID.
    # Tuples are immutable/hashable dictionary keys; their values are the corpus frequencies.
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    # Explicit inverted index: pair -> set of distinct ID sequences containing it. The set prevents
    # duplicate membership if a pair occurs more than once in one sequence; the index uses memory
    # to avoid searching every sequence after every merge (this is not functools memoisation).
# Inverted index: pair -> the distinct tuple sequences containing it. Extra memory lets each merge
    # visit just affected sequences instead of scanning the entire corpus.
    pair_to_ids = defaultdict(set)

    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        # Build pair memberships alongside initial counts; this one-time pass is linear in all IDs.
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)

    for step in tqdm(range(num_merges), desc="Training BPE"):
        # Stop before max() on an empty mapping: no adjacent pairs remain to merge.
        if not pair_counts:
            break

        # BPE training picks the pair with the largest corpus-wide count; numeric pair IDs break ties
        # deterministically. Each selected pair gets a fresh token ID, recording the merge order.
        # best_pair = max(pair_counts, key=pair_counts.get)
        # max scans distinct pair keys; the cache removes the much larger full sequence rescan.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # The index identifies only sequence types that can change; unrelated sequences need no scan.
        affected_ids = pair_to_ids[best_pair]
        # The chosen pair disappears after this rule, so its membership entry is no longer needed.
        del pair_to_ids[best_pair]

        # Update only affected distinct sequences, preserving each sequence frequency. Worst case,
        # a frequent pair occurs in many sequences and the update still approaches a full scan.
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            # Remove the old hashable tuple before replacing it with its merged form.
            del ids_counts[tuple(ids)]
            # Keep the sequence frequency; identical output tuples share one entry in this dictionary.
            ids_counts[tuple(new_ids)] = ids_count

            # Remove old neighboring-pair contributions, weighted by this sequence frequency, and
            # remove the old tuple from each reverse-index set.
            # For each changed sequence, adjust its local pairs; the cost is proportional to its length.
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            # Add the replacement sequence’s weighted pair counts and memberships. This delta update
            # avoids recomputing the whole corpus, though changed sequence lengths still determine cost.
            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return merge_rules


vocab_size = 1000  # Capacity: byte IDs, the marker ID, and at most 743 learned merges.
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(text, vocab_size)

# vocab_size = 10000
# file_path = "storybot/tiny_stories_train.txt"
# text = open(file_path).read()
# merge_rules = train_bpe(text, vocab_size)

# ===== CONCEPT NOTES =====
# CONCEPT: `pair_counts` and `pair_to_ids` support incremental updates to BPE pair frequencies.
# GOAL: Continue the training optimizations by avoiding full-corpus recounting after each merge; merge selection still uses frequency and pair-ID tie-breaking.
# INPUT: Reads `codebot/tiny_codes.txt` and calls `train_bpe` with `vocab_size=1000` and default `<|endoftext|>`. Commented alternative: `storybot/tiny_stories_train.txt` with size 10000.
# OUTPUT: Returns a pair-to-ID mapping assigned to `merge_rules`, not printed or saved. `tqdm` shows progress. The active call requests at most 743 merges, with early stop if there are no pairs.
# HOW INPUT BECOMES OUTPUT:
# 1. Split on the end marker; count `pretokenize` results, then encode each as a weighted UTF-8 byte tuple.
# 2. An initial scan fills pair frequencies and maps pairs to containing sequences. For example, `(97, 98, 97)` contains `(97, 98)` and `(98, 97)`.
# 3. Select the most frequent pair (pair IDs break ties), assign `new_id = 256 + step`, and get its affected sequences from `pair_to_ids`.
# 4. For each affected sequence, `merge` creates its replacement; old pair totals/memberships are removed and new totals/memberships are added.
# NOTES: The active text file must exist. The commented story alternative requires its separate dataset. The returned mapping is not persisted.
