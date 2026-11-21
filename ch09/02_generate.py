# Concept: generate continuations from a pretrained web-text GPT using KV caching.
# Goal: compare model continuations for five fixed natural-language prompts.
# This follows ch09/01_pretrain.py and reads its saved webbot checkpoint.
# The tokenizer is loaded from storybot's shared BPE merge-rules file.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from webbot.model import GPT
from storybot.tokenizer import BPETokenizer
# generate performs inference with a KV cache; get_device chooses CUDA, MPS, or CPU.
from webbot.utils import generate, get_device

# Runtime setup: select device and paths for the checkpoint and matching tokenizer.
device = get_device()
model_path = 'webbot/model_pretrain.pt'
tokenizer_path = 'webbot/merge_rules.pkl'
max_new_tokens = 100
temperature = 0.5  # Lower temperature makes token sampling less random.
# Top-k filtering is not used here; it would sample only among the k highest-probability IDs.

# Fixed prompts make generated continuations easy to compare.
prompts = [
    "In 1991, Linus Torvalds created",
    "Monday, Tuesday, Wednesday,",
    "Python was created by",
    "Machine learning is defined as",
    "The capital of Japan is",
]

# Load model weights and matching tokenizer rules before inference.
# Japanese status means “Loading model and tokenizer”; preserve output.
print("モデルとトークナイザを読み込み中...")
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)
# Japanese status means “Loading complete”; preserve output.
print(f"読み込み完了！\n")

# Generate and print a continuation for every fixed test prompt.
for i, prompt in enumerate(prompts, 1):
    print(f"{'=' * 70}")
    # Japanese label means “Prompt”; preserve output.
    print(f"プロンプト {i}: {prompt}")
    print(f"{'=' * 70}")

    # Shared generator encodes the prompt, samples next-token IDs with KV caching, then decodes text.
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(response)
    print()

# ===== CONCEPT NOTES =====
# CONCEPT: Autoregressive language generation extends each prompt by sampling
# next-token IDs; webbot.utils uses a key/value cache to avoid recomputing old states.
# GOAL: Inspect the pretrained webbot GPT from ch09/01_pretrain.py on five prompts.
# INPUT: webbot/model_pretrain.pt, storybot/merge_rules.pkl, and five hard-coded text
# prompts. The cap is 100 new tokens per prompt and temperature is 0.5.
# OUTPUT: Prints a loading status, then each numbered prompt and generated text;
# nothing is saved.
# HOW INPUT BECOMES OUTPUT:
# 1. BPETokenizer.load_from reads BPE merges; GPT.load_from loads the checkpoint.
# 2. For each prompt, webbot.utils.generate clears the model's KV cache and encodes
#    the prompt into an input tensor on the model's device.
# 3. The model predicts one next-token distribution at a time with use_cache=True;
#    temperature-scaled softmax is sampled until the end token or 100-token limit.
# 4. The utility removes any end-token IDs, decodes the accumulated IDs, and this
#    script prints the returned text under the corresponding prompt.
# NOTES: The webbot GPT is configured with a large context/model in its training
# script, so CPU loading and autoregressive inference may be slow; this inference
# path does not require a GPU. The expected checkpoint and merge-rules file must
# already exist. No model or data file is modified.
