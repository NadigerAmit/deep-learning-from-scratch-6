# Concept: Train BPE from a large file in end-marker-aligned byte chunks.
# Goal: Limit preprocessing memory while reusing the cached pair-update strategy.
# This version processes chunks sequentially; the next example parallelizes preprocessing.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
# `collections.defaultdict(int)` provides zero-default integer counts; `defaultdict(set)` creates
# membership sets on first use. These are explicit data structures, not functools memoisation.
from collections import defaultdict
import regex as re
from tqdm import tqdm


# Unlike findall, regex.finditer yields matches lazily; group(0) returns each matched string.
def pretokenize(text):
    # `regex` (the third-party package imported as `re`) recognizes Unicode letter/number classes.
    # `finditer` yields match objects; group(0) is each matching string, not yet a vocabulary ID.
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

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

# Read a large file as bytes so seek positions are exact byte offsets; align chunk starts to markers.
def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:  # Open the file in binary mode.
        # Get the file size.
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # Integer division gives approximate equal-sized chunks; marker alignment can shift the boundaries.
        chunk_size = file_size // num_chunks

        # Calculate evenly spaced chunk boundary estimates.
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # Append the end-of-file offset.

        # Searching in bounded buffers avoids reading the whole corpus into memory.
        buffer_size = 4096  # Number of bytes to read ahead while searching for a boundary.

        # Adjust boundaries by searching for the end marker.
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)  # Start at the estimated boundary position.

            while True:
                buffer = file.read(buffer_size)  # Read one buffer-sized block.

                # Handle reaching the end of the file.
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Search the bytes just read for the end marker.
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # If found, use its byte position as the new boundary.
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                # If not found, continue at the next buffer position.
                chunk_position += buffer_size

    # Remove duplicates, sort the offsets, and return them.
    return sorted(set(chunk_boundaries))

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>"):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks=64)

    # Split on the special marker first so it is kept separate from ordinary text and is not learned as a byte merge.
    # Raw text is bounded by the current chunk, but this aggregate of unique pretokens remains in memory.
    pretoken_counts = defaultdict(int)
    with open(file_path, "rb") as f:
        total_chunks = len(chunk_boundaries) - 1

        # tqdm reports chunk progress; chunks are still processed one after another in this example.
        for i in tqdm(range(total_chunks), desc="Pretokenizing"):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i+1]

            f.seek(start)
            chunk_byte = f.read(end - start)
            # Decode each bytes chunk to str; errors="ignore" skips any incomplete UTF-8 bytes at its edge.
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # Split the chunk at the special token.
            texts = chunk_text.split(end_token)
            # Count occurrences of each regex-defined pretoken; repeated strings are represented once plus a frequency.
            for text in texts:
                for pretoken in pretokenize(text):
                    pretoken_counts[pretoken] += 1


    # Encode each pretoken as UTF-8 bytes: each byte value 0..255 is a starting token ID.
    # Tuples are immutable/hashable dictionary keys; their values are the corpus frequencies.
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    # Explicit inverted index: pair -> set of distinct ID sequences containing it. The set avoids
    # duplicate membership for repeated occurrences; extra memory saves rescanning unrelated sequences.
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


file_path = "storybot/tiny_stories_train.txt"
vocab_size = 10000  # Capacity: byte IDs, the marker ID, and at most 9743 learned merges.
merge_rules = train_bpe(file_path, vocab_size)

# ===== CONCEPT NOTES =====
# CONCEPT: Chunked file reading avoids loading the entire corpus into one text string; cached memberships limit work during the subsequent BPE merges.
# GOAL: Extend the in-memory trainers with bounded-size preprocessing. Unlike the next parallel example, this file reads and counts chunks sequentially.
# INPUT: Reads `storybot/tiny_stories_train.txt`; calls `train_bpe(file_path, vocab_size=10000)` and requests 64 chunks. `find_chunk_boundaries` scans binary data; chunks decode as UTF-8 with invalid bytes ignored.
# OUTPUT: Returns pair-to-new-ID `merge_rules`, assigned but not printed or saved. `tqdm` shows chunk and merge progress. At most 9743 merges are requested (`10000 - 256 - 1`); training may stop earlier.
# HOW INPUT BECOMES OUTPUT:
# 1. `find_chunk_boundaries` estimates 64 ranges and moves interior boundaries forward to the next byte occurrence of `<|endoftext|>`; duplicate boundaries are removed.
# 2. `train_bpe` reads each range, decodes it, splits on the marker, and counts `pretokenize` results. For example, `aba` becomes byte tuple `(97, 98, 97)`.
# 3. Weighted byte sequences initialize `ids_counts`, `pair_counts`, and `pair_to_ids`. The loop chooses a frequent pair and assigns `256 + step`.
# 4. `merge` updates affected sequences; cached frequencies and memberships are adjusted, and the completed `merge_rules` mapping is returned.
# NOTES: `storybot/tiny_stories_train.txt` is required (it is not included in the repository's tiny/OWT training-file listing). Fewer than 64 chunks may result after boundary deduplication; no output file is written.
