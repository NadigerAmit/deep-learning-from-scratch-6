# Concept: run an interactive instruction-following chat loop with a fine-tuned GPT.
# Goal: format each user message as an instruction and print the generated reply.
# This uses the SFT checkpoint from ch03/03_sft.py (or the optional GRPO checkpoint).
# The shared generator performs inference only; this script does not train or save.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device

# Select a device and checkpoint; switch to the commented GRPO checkpoint to try that stage.
device = get_device()
model_path = 'codebot/model_sft.pt'
# model_path = 'codebot/model_grpo.pt'
tokenizer_path = 'codebot/merge_rules.pkl'
max_new_tokens = 200
temperature = 1.0

# A chat template is the agreed input structure; keep it consistent with the SFT examples.
def format_prompt(user_message):
    # Section labels are literal template text consumed by the model, not comments.
    return f"### Instruction:\n{user_message}\n\n### Response:\n"

# Load tokenizer and model so user text and weights use the same vocabulary.
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# Interactive inference loop: each nonempty message is answered; this script performs no training.
while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    # Apply the training-time template and generate a reply without changing weights.
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    # If the model repeats the header, discard everything through its final occurrence.
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # Display multiline answers on separate lines; keep short answers on one line.
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")

# ===== CONCEPT NOTES =====
# CONCEPT: A prompt template connects a user's message to a model trained on
# instruction/response examples, and autoregressive decoding supplies the reply.
# GOAL: Try the checkpoint produced by ch03/03_sft.py; ch03/09_grpo.py is an
# optional later alignment stage whose checkpoint can be selected in the comment.
# INPUT: codebot/model_sft.pt by default (the model_grpo.pt assignment is commented
# out), codebot/merge_rules.pkl, max_new_tokens 200, and temperature 1.0. User input
# is read interactively and formatted by format_prompt.
# OUTPUT: Prints a "Bot:" reply for each nonempty input; no file is saved.
# HOW INPUT BECOMES OUTPUT:
# 1. BPETokenizer and GPT load the merge rules and selected model checkpoint.
# 2. The loop reads and strips a line; empty messages are skipped.
# 3. format_prompt wraps the message with the Instruction/Response labels, then
#    codebot.utils.generate tokenizes it and samples response tokens from the model.
# 4. The script removes the prefix through the last "### Response:" marker when
#    present, then prints the result on one line or multiple lines as appropriate.
# NOTES: The loop continues until input is interrupted/closed. Temperature 1.0 is
# stochastic. Checkpoint loading and inference run on the device returned by
# get_device; CPU-only use is supported but generation may be slow. No files change.
