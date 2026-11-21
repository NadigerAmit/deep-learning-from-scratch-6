# Concept: Parallelize BPE corpus preprocessing and cache pair-to-sequence memberships.
# Goal: Count pretokens across end-marker-aligned chunks with worker processes, then train merge rules.
# The script persists the rules for a later tokenizer to load.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
# Pool distributes independent chunk tasks to child processes; parallelism can use multiple CPU cores.
from multiprocessing import Pool
# `collections.defaultdict(int)` provides zero-default integer counts; `defaultdict(set)` creates
# membership sets on first use. These are explicit data structures, not functools memoisation.
from collections import defaultdict
import regex as re
# tqdm can wrap the Pool iterator to show progress without changing each result.
from tqdm import tqdm
# `pickle.dump` serializes the learned merge-rules dictionary to a binary stream for later encoding.
import pickle


# `regex.finditer` is lazy: each match object is read and converted to its matching string on demand.
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

# Return sorted list[int] byte offsets, with interior offsets moved to the next marker so chunks
# are independently processable without splitting a document.
def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks


        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # Append the end-of-file offset.

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

# Pool worker input is (path: str, start: int, end: int, marker: str); it returns a local
# defaultdict[str, int]. Each worker reopens the file because open handles are not passed between processes.
def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    # Open the file and read this chunk.
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # Split on the special marker first so it is kept separate from ordinary text and is not learned as a byte merge.
        texts = chunk_text.split(end_token)

        # Pretokenize each text segment.
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=64):
    # Step 1: prepare the chunks.
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    # Step 2: process chunks in parallel.
    # The context manager closes/joins workers on exit. imap yields results in task order; each result
    # is a dict of local pretoken counts. Starting workers and transferring dictionaries also costs time.
    with Pool(processes=num_processes) as pool:
        # imap yields task results in order; tqdm shows completed tasks while list collects them.
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list), desc="Pretokenizing"))

    # Step 3: combine worker results.
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    # Encode each pretoken as UTF-8 bytes: each byte value 0..255 is a starting token ID.
    # Tuples are immutable/hashable dictionary keys; their values are the corpus frequencies.
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}


    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    # Explicit inverted index: pair -> set of distinct ID sequences containing it. Extra memory
    # lets each merge visit just affected sequences instead of scanning the whole corpus.
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


if __name__ == '__main__':
    vocab_size = 10000
    file_path = "storybot/tiny_stories_train.txt"
    # Run the work in parallel using eight processes.
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)

    with open("storybot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)

# ===== CONCEPT NOTES =====
# CONCEPT: `multiprocessing.Pool` parallelizes chunk reading and pretoken counting; the combined counts feed cached incremental BPE training in the parent process.
# GOAL: Build on the sequential chunked trainer by distributing preprocessing and saving the result for encoding. Merge iterations themselves are not parallelized.
# INPUT: Main block reads `storybot/tiny_stories_train.txt`, uses `vocab_size=10000`, 8 processes, 64 chunks, and default marker `<|endoftext|>`.
# OUTPUT: Returns merge rules and pickles them to `storybot/merge_rules.pkl`, overwriting that path. `tqdm` shows chunk/training progress. At most 9743 merges are requested; no-pair termination can reduce that number.
# HOW INPUT BECOMES OUTPUT:
# 1. `find_chunk_boundaries` computes byte ranges; `chunk_info_list` packages each range for a worker.
# 2. `Pool.imap(pretoken_chunk, ...)` reads and decodes each range, splits on the end marker, and counts pretokens locally. Illustrative `aba` becomes `(97, 98, 97)`.
# 3. The parent combines worker counts, builds weighted byte sequences, and initializes `pair_counts` and `pair_to_ids`.
# 4. Each iteration selects the most frequent pair (IDs break ties), assigns `256 + step`, updates affected sequences/caches, and returns the mapping for `pickle.dump`.
# NOTES: The story training dataset must exist. Chunk decoding ignores invalid UTF-8; duplicate boundaries can reduce the chunk count. Importing the module does not run the main block.
