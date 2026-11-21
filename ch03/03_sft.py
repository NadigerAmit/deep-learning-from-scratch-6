# Concept: supervised fine-tuning (SFT) adapts a pretrained GPT to responses.
# Goal: train on instruction/response pairs while masking prompt tokens from loss.
# This follows pretraining and prepares the model used by ch03/04_chat.py.
# The response includes the end-of-text token so training can learn when to stop.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device

# Runtime setup: paths identify SFT data, tokenizer, pretrained weights, and output checkpoint.
device = get_device()
data_path = 'codebot/tiny_codes_sft.json'
tokenizer_path = 'codebot/merge_rules.pkl'
pretrain_model_path = 'codebot/model_pretrain.pt'
sft_model_save_path = 'codebot/model_sft.pt'

# Hyperparameters: T is tokens per sample, B is examples per batch, and learning rate sets update scale.
context_len = 256
batch_size = 32
learning_rate = 3e-4  # Scale of each AdamW update while adapting pretrained weights.
max_iters = 500

# SFT (supervised fine-tuning) starts from pretraining and learns instruction following from labeled answers.
# Prompt IDs stay as context, but their labels are masked out of the loss.
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        # json.load (Python standard library) parses the JSON list of instruction/response dictionaries.
        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            ids, labels = self._create_sample(item['instruction'], item['response'])
            self.samples.append((ids, labels))

    def _create_sample(self, instruction, response):
        # Chat template: section labels distinguish request from answer, matching training format.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        # Encode parts separately to retain the prompt/answer boundary after tokenization.
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # Prompt labels are -100, cross_entropy’s ignore_index, so only answer targets train.
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # Shift by one: at t the causal model uses tokens through t to predict token t+1.
        ids = ids[:-1]
        labels = labels[1:]

        # Pad/truncate to common length so DataLoader can stack examples into a batch.
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len  # ID 0 is filler; its target is -100, so padding is ignored.
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]

        return ids, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long)


# Load tokenizer/data; DataLoader collates samples into tensors shaped (B,T).
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = SFTDataset(data_path, tokenizer, context_len)
# DataLoader stacks samples into input/target batches shaped (B,T).
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Restore pretrained weights and create the optimizer that adapts them to instructions.
# GPT.load_from restores checkpoint weights and architecture metadata on the chosen device.
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Predict next tokens, compute cross-entropy, backpropagate, and update weights.
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))  # tqdm renders a live progress bar; it does not change training.

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # Forward pass maps IDs (B,T) to vocabulary scores (B,T,V).
    logits = model(batch_x)
    # Cross-entropy scores correct next-token predictions; ignore_index skips prompt and padding positions.
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100
    )

    # Start a fresh gradient accumulation cycle before backward.
    optimizer.zero_grad()
    # Backpropagation computes how each trainable weight contributed to prediction error.
    loss.backward()
    # AdamW applies one update; repeated steps adapt pretrained weights to examples.
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# Plot training losses and save the SFT checkpoint.
# matplotlib.pyplot plots the recorded scalar losses; the graph is a diagnostic, not model state.
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_sft.png')

# Save weights and architecture metadata for future loading.
model.save(sft_model_save_path)

# ===== CONCEPT NOTES =====
# CONCEPT: Supervised fine-tuning teaches a pretrained causal LM to respond to
# instructions; only response-token positions contribute to the cross-entropy loss.
# GOAL: Adapt codebot/model_pretrain.pt using instruction/response examples. This
# follows ch03/01_pretrain.py and prepares a checkpoint for ch03/04_chat.py.
# INPUT: codebot/tiny_codes_sft.json and codebot/merge_rules.pkl; context_len 256,
# batch size 32, learning rate 3e-4, and 500 iterations. Each sample is formatted
# with instruction/response markers, receives a terminal <|endoftext|> token, and
# is padded with ID 0 or truncated to 256 positions. Prompt labels are -100.
# OUTPUT: tqdm reports loss; loss_sft.png is written and codebot/model_sft.pt saved.
# HOW INPUT BECOMES OUTPUT:
# 1. SFTDataset loads JSON and tokenizes prompt and response separately.
# 2. It concatenates IDs, masks prompt labels with -100, shifts IDs/labels by one,
#    then pads/truncates each sample; DataLoader shuffles batches of 32.
# 3. GPT.load_from initializes from the pretrained checkpoint; forward produces
#    logits, and cross_entropy(ignore_index=-100) scores response targets only.
# 4. zero_grad, backward, and AdamW optimizer.step update the model; losses are
#    recorded, plotted, and model.save writes the fine-tuned checkpoint.
# NOTES: ID 0 is used for padding and ignored labels keep padded positions out of
# the loss. The loss plot and model_sft.pt are overwritten on rerun. CPU-only
# execution is supported by get_device but 500 updates may take time; no GPU needed.
