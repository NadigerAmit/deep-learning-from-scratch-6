# Concept: Encode text by applying the earliest-ranked currently available BPE merge.
# Goal: Use merge-rule priority to skip rules that cannot apply to the current token sequence.
# The example loads code-corpus rules and prints the encoded token count.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import pretokenize, count_pairs, merge

import pickle
import regex as re
from tqdm import tqdm


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

    def encode(self, input_text, show_progress=False):
        # regex.escape makes the marker literal; re.split returns list[str] and preserves the captured marker.
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # `tqdm` adds progress reporting while yielding the same text segments in order.
        texts = tqdm(texts) if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
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


if __name__ == "__main__":
    tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

    file_path = "codebot/tiny_codes.txt"
    text = open(file_path).read()
    ids = tokenizer.encode(text, show_progress=True)
    print(len(ids))

# ===== CONCEPT NOTES =====
# CONCEPT: `BPETokenizer` reconstructs token bytes and encodes each pretoken by choosing the available pair with the smallest learned merge ID.
# GOAL: Shift from training optimizations to efficient encoding; merge IDs represent rule order, so lowest-priority-number-first reproduces that order without testing every rule.
# INPUT: Loads `codebot/merge_rules.pkl`, reads `codebot/tiny_codes.txt`, and calls `encode(text, show_progress=True)`. The default special marker is `<|endoftext|>`; `pretokenize`, `count_pairs`, and `merge` come from `storybot.tokenizer`.
# OUTPUT: `encode` returns token IDs and the script prints only `len(ids)`; it writes no encoded file. Each marker occurrence becomes ID `256 + len(merge_rules)`.
# HOW INPUT BECOMES OUTPUT:
# 1. `load_from` unpickles rules; `__init__` starts `id_to_bytes` with all 256 byte tokens and combines child bytes for learned IDs.
# 2. `encode` splits while preserving the special marker, optionally shows `tqdm`, and sends ordinary segments through `pretokenize`.
# 3. `_encode_text` converts a pretoken to UTF-8 byte IDs; `count_pairs` finds adjacent pairs and `get_merge_priority` assigns learned IDs (or infinity for unknown pairs).
# 4. It repeatedly selects the minimum-priority pair; if unlearned, it stops, otherwise `merge` replaces its occurrences. The caller collects IDs and prints their count.
# NOTES: Both the merge-rules pickle and code text file must exist and match. This example only prints a count; it does not save tokens.
