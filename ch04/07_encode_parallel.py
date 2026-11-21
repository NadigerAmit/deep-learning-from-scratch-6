# Concept: Encode a large file in parallel chunks and assemble token IDs in a memory-mapped array.
# Goal: Scale BPE encoding while keeping chunk order in the final binary stream.
# Temporary per-chunk caches are removed after each file has been encoded.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import pretokenize, count_pairs, merge, find_chunk_boundaries

import os
import pickle
from multiprocessing import Pool
import shutil
import regex as re
from tqdm import tqdm
import numpy as np


class BPETokenizer:
    # The trained merge table defines the vocabulary beyond the 256 possible byte values.
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)  # Reserve a distinct ID immediately after learned merge IDs.

        # Start with all 256 one-byte tokens; learned merge IDs map to concatenated byte strings.
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)  # Number of known IDs, including base bytes and the marker.

    @staticmethod
    def load_from(filepath):
        # Open the pickle as binary and deserialize the saved dict[tuple[int, int], int].
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    # Encoding is the reverse direction of training: take text bytes and apply already-learned rules.
    def _encode_text(self, text):
        # str.encode gives bytes; list exposes each byte as an integer in 0..255, the base token IDs.
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf'))  # An unlearned pair has the lowest priority.

        while len(ids) > 1:
            # Get the pairs currently present (step 1).
            counts = count_pairs(ids)

            # Find the highest-priority pair (step 2).
            # min chooses the earliest learned applicable rule, not the pair with the most occurrences.
            best_pair = min(counts, key=get_merge_priority)

            # Check whether the pair can be merged (step 3).
            if best_pair not in self.merge_rules:
                break

            # Apply the merge (step 4).
            new_id = self.merge_rules[best_pair]
            # Example: [97,98,97], rule (97,98)->256 becomes [256,97]; repeat because neighbors change.
            ids = merge(ids, best_pair, new_id)

        return ids

    # Pool workers receive a byte-range task and return (cache path, number of encoded IDs).
    def _encode_chunk(self, args):
        """Process one chunk and cache its token IDs on disk."""
        file_path, start, end, cache_dir, chunk_idx = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # Encode the chunk.
            ids = self.encode(chunk_text)

        # Save the encoded chunk to a cache file.
        cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
        # uint16 stores IDs in two bytes each; tofile writes raw array bytes, not a .npy header.
        np.array(ids, dtype=np.uint16).tofile(cache_file)

        return cache_file, len(ids)


    def encode_file(self, file_path, output_file,
                                    num_processes=8, num_chunks=64,
                                   cache_dir="bpe_cache"):

        # Prepare the cache directory.
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Tokenize chunks in parallel and cache their outputs.
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1

            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))

            # Parallelize only independent chunks; imap keeps their results in submission order.
            with Pool(processes=num_processes) as pool:
                cache_results = list(tqdm(
                    pool.imap(self._encode_chunk, chunk_info_list),
                    total=len(chunk_info_list),
                    desc="Encoding chunks"
                ))

            # Calculate the total token count.
            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            # Counts determine the exact memmap shape before writing, so no resize/copy pass is needed.
            total_tokens = sum(token_counts)

            # Create the memory-mapped output file.
            dtype = np.uint16
            # A memory map is file-backed: the final array need not fit wholly in Python-managed RAM.
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            # Copy cached chunks to the memory-mapped output in batches.
            idx = 0
            # Append chunk arrays in the original range order to reconstruct the corpus token stream.
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()
            del arr

        finally:
            # Delete intermediate chunks after success or failure; the final binary output remains.
            shutil.rmtree(cache_dir)

        return total_tokens

    def encode(self, input_text, show_progress=False):
        # regex.escape makes the marker literal; re.split returns list[str] and preserves the captured marker.
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # Show a tqdm progress bar when show_progress is True.
        # `tqdm` adds progress reporting while yielding the same text segments in order.
        texts = tqdm(texts) if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # BPE-encode each pretoken.
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        # Each token ID maps back to bytes; concatenating bytes before UTF-8 decoding handles multibyte text.
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin", num_processes=8)

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin", num_processes=8)

# ===== CONCEPT NOTES =====
# CONCEPT: Parallel chunk workers write temporary token-ID arrays; `encode_file` assembles them into a `uint16` memmap instead of one large Python list.
# GOAL: Extend the encoding lesson to large train/validation files using parallel processing and disk-backed intermediate data.
# INPUT: Loads `storybot/merge_rules.pkl`, then reads `storybot/tiny_stories_train.txt` and `storybot/tiny_stories_valid.txt`. Each call uses 8 processes, 64 requested chunks, default `bpe_cache` and `<|endoftext|>`. Shared helpers are imported from `storybot.tokenizer`.
# OUTPUT: Writes raw `uint16` token-ID arrays to `storybot/tiny_stories_train.bin` and `storybot/tiny_stories_valid.bin` (not NumPy `.npy` containers). Returned token counts are ignored. Temporary `chunk_00000.npy`-style files are written under `bpe_cache`; `finally` removes that directory. Memmap mode `w+` overwrites output files.
# HOW INPUT BECOMES OUTPUT:
# 1. `encode_file` creates the cache and asks `find_chunk_boundaries` for byte ranges, then builds indexed chunk tasks.
# 2. Pool workers invoke `_encode_chunk`: read/decode a range (invalid UTF-8 is ignored), call `encode`, and save IDs with `np.uint16` and `tofile`; each returns path and count.
# 3. The parent sums counts, creates a memmap of exactly that size, and copies cached chunk arrays into consecutive slices in chunk order.
# 4. The array is flushed and closed; `finally` deletes the cache directory, and `encode_file` returns total tokens.
# NOTES: The pickle and both story datasets are required. Token IDs must fit in `uint16`. Cache files are temporary and output `.bin` paths are replaced, not appended.
