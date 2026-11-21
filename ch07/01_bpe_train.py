# Concept: Train BPE merge rules for the web-text corpus using the shared parallel trainer.
# Goal: Produce a larger vocabulary for the chapter's web-text tokenizer and following encode step.
# This driver saves the returned merge rules as a pickle.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
import pickle
# Training details live in the shared tokenizer module; this file is the web-corpus configuration driver.
from storybot.tokenizer import train_bpe


# The guard avoids training or creating files when the module is imported.
if __name__ == '__main__':
    vocab_size = 50000
    file_path = "webbot/owt_train.txt"
    # The trainer parallelizes pretoken counting but merge choices remain sequential and dependent.
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8, num_chunks=64)

    with open("webbot/merge_rules.pkl", "wb") as f:
        # Store the pair-to-new-ID dictionary as binary pickle data for the encoder.
        pickle.dump(merge_rules, f)

# ===== CONCEPT NOTES =====
# CONCEPT: This driver delegates chunking, parallel pretoken counting, cached pair updates, and BPE merge selection to `storybot.tokenizer.train_bpe`.
# GOAL: Train on web text, extending the earlier chapter's BPE training/parallelization examples; the saved rules are consumed by the next chapter script.
# INPUT: Main block passes `webbot/owt_train.txt`, `vocab_size=50000`, `num_processes=8`, and `num_chunks=64`. The shared trainer's default end marker is `<|endoftext|>`.
# OUTPUT: Pickles the returned pair-to-ID dictionary to `webbot/merge_rules.pkl`, overwriting any existing file. At most 49743 merges are requested (`50000 - 256 - 1`); the shared trainer may stop if no pairs remain. Its `tqdm` bars report preprocessing/training.
# HOW INPUT BECOMES OUTPUT:
# 1. The main guard calls shared `train_bpe` with the corpus and explicit vocabulary/process/chunk settings.
# 2. The shared function finds marker-aligned ranges and has `pretoken_chunk` workers count pretokens in each range.
# 3. Worker counts are combined into weighted UTF-8 byte-ID sequences; the trainer initializes pair frequencies and sequence-membership caches.
# 4. It repeatedly chooses the most frequent pair (pair IDs break ties), applies `merge` to affected sequences, and returns rules for `pickle.dump`.
# NOTES: `webbot/owt_train.txt` must exist; training on this corpus may take substantial time and memory. Importing the file does not run training or write the pickle.
