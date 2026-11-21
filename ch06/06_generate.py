# Concept: autoregressive text generation from a pretrained Storybot GPT.
# Goal: turn a prompt into sampled text using the shared BPE tokenizer and KV cache.
# The script loads the final pretraining checkpoint and generates three samples.
# It prints text to the console and writes no output files.

import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# Paths and runtime device configuration.
device = get_device()
model_path = 'storybot/model_pretrain.pt'
tokenizer_path = 'storybot/merge_rules.pkl'

# Sampling controls: temperature changes distribution sharpness; sample count repeats generation.
# Alternative English story starter kept as a code example: “Once upon a time.”
# <|endoftext|> is a functional special-token prompt marking a story boundary.
prompt = "<|endoftext|>"
max_new_tokens = 300  # Stop after at most this many newly sampled token IDs.
temperature = 1.0  # Divide logits by temperature; higher values flatten sampling probabilities.
num_samples = 3  # Generate and print three independent sampled stories.

# BPETokenizer.load_from restores merge rules; GPT.load_from loads weights onto the selected device.
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# storybot.utils.generate encodes the prompt, uses cached inference, then decodes IDs to text.
for i in range(num_samples):
    # Japanese console label means “Sample N.”
    print(f"--- サンプル {i+1} ---")
    # The shared utility returns decoded text; it does not train or write a file. get_device chooses CUDA, then MPS, else CPU.
    story = generate(
        model, tokenizer, prompt, max_new_tokens, temperature
    )
    print(story)

# ===== CONCEPT NOTES =====
# CONCEPT: Autoregressive generation repeatedly selects the next token from model logits. Here temperature-scaled softmax followed by multinomial sampling is provided by storybot.utils.generate, which also uses the model's KV cache and stops at the tokenizer's end token.
# GOAL: Inspect text produced by the pretrained language model before preference tuning. It consumes the final checkpoint written by ch06/05_pretrain.py.
# INPUT: storybot/model_pretrain.pt and storybot/merge_rules.pkl; prompt is `<|endoftext|>`, max_new_tokens=300, temperature=1.0, and num_samples=3. Device is selected by get_device (CUDA, then MPS, else CPU).
# OUTPUT: Prints a Japanese sample label, translated here as “Sample N,” and decoded generated text for each of three attempts. No model or generated text file is saved.
# HOW INPUT BECOMES OUTPUT: 1) Load the BPE merge rules and model checkpoint. 2) Encode the prompt into IDs and pass them to the shared generator. 3) At each step, run the current token through cached GPT inference, divide logits by temperature 1.0, softmax, and sample one token. 4) Stop on end_token_id or after 300 new-token attempts; decode and print the generated IDs (the utility removes the end token before decoding).
# NOTES: Requires both files produced/downloaded for Storybot; absent model_pretrain.pt or merge_rules.pkl prevents execution. Three generations are stochastic at temperature 1.0. CPU works but generation can be slow; this script does not train a model.
