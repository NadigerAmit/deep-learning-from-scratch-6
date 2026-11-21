# Concept: autoregressively sample code tokens from a pretrained GPT checkpoint.
# Goal: turn a short text prompt into decoded text without updating model weights.
# This follows ch03/01_pretrain.py and demonstrates inference after pretraining.
# Sampling stops at the tokenizer's end token or the configured generation limit.

import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import torch
import torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device


# Runtime setup: select a device and identify the pretrained model and matching tokenizer.
device = get_device()
model_path = 'codebot/model_pretrain.pt'
tokenizer_path = 'codebot/merge_rules.pkl'

# Generation settings: set the output token limit and sampling temperature.
prompt = "def"  # Starting prompt: generation continues after this text.
max_new_tokens = 200  # Hard cap on the number of new token IDs to sample.
temperature = 1.0  # Higher temperature flattens probabilities and increases randomness.
# Top-k sampling (not implemented here) would restrict each choice to the k highest-scoring tokens.

@torch.no_grad()
# Inference helper: encode text, sample one next ID repeatedly, append it, then decode the full sequence.
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # Disable training-time dropout; @torch.no_grad also avoids gradient storage.

    # BPE encoding maps text to IDs; add a batch axis so ids has shape (1,T).
    device = next(model.parameters()).device  # Keep inputs on the same device as model weights.
    ids = tokenizer.encode(prompt)
    # One prompt forms a batch of one, so input IDs have shape (1,T).
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Keep the full sequence for decoding; ids may be cropped to the context window.
    generated_ids = ids.clone()

    # Autoregressive loop: predict one next token, append it, then use the longer prefix.
    for _ in range(max_new_tokens):
        # Keep only the newest context once the model’s maximum length is exceeded.
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # GPT returns (B,T,V) logits; final-position scores predict the next ID.
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            # softmax converts scores to probabilities; lower temperature sharpens, higher temperature flattens.
            probs = F.softmax(logits / temperature, dim=-1)
            # torch.multinomial samples one vocabulary ID (B,1); unlike argmax, it may choose a lower-ranked token.
            next_id = torch.multinomial(probs, num_samples=1)

        # The tokenizer end marker signals that the response is complete.
        if next_id.item() == tokenizer.end_token_id:
            break

        # Append the sampled ID along the time axis for the next prediction.
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # Decode accumulated IDs back to text with the same BPE vocabulary.
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# Generate five independent continuations; model weights are not updated.
for i in range(5):
    # Japanese label means “Sample”; preserve output.
    print(f"--- サンプル {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(generated_text)
    print()

# ===== CONCEPT NOTES =====
# CONCEPT: Autoregressive generation repeatedly predicts and samples the next token.
# GOAL: Demonstrate what the checkpoint from ch03/01_pretrain.py has learned; the
# next lesson ch03/03_sft.py adapts a pretrained model to instruction/response data.
# INPUT: codebot/model_pretrain.pt and codebot/merge_rules.pkl; prompt is "def",
# max_new_tokens is 200, temperature is 1.0, and the script requests 5 samples.
# Each forward call receives at most model.max_context_len tokens.
# OUTPUT: Five labeled generated-text samples are printed; no file is saved.
# HOW INPUT BECOMES OUTPUT:
# 1. BPETokenizer.load_from loads merge rules and encode converts the prompt to IDs.
# 2. generate creates a one-row ID tensor on the model device and switches to eval.
# 3. Each loop calls GPT on the current context, takes the final-position logits,
#    and either chooses argmax (temperature 0) or samples from softmax(logits/temp).
# 4. The sampled ID is appended; an end-token stops generation, and excess context
#    is cropped before the next prediction. tokenizer.decode turns accumulated IDs
#    back into text, which the outer loop prints five times.
# NOTES: Temperature controls sampling randomness; 1.0 is not greedy. This script
# only reads the checkpoint and tokenizer. Generation can be slow on CPU, but it
# does not require a GPU and does not overwrite the model.
