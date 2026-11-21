# Concept: train a small GPT as a next-token language model on tokenized code.
# Goal: learn to predict each next token from a fixed-length preceding context.
# This is chapter 3's pretraining stage; its checkpoint feeds the SFT lesson.
# The code uses the codebot GPT, whose logits are trained against shifted token IDs.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.utils import get_device

# Runtime setup: choose a device and identify tokenized input and checkpoint paths.
device = get_device()
data_path = 'codebot/tiny_codes.bin'
tokenizer_path = 'codebot/merge_rules.pkl'
model_save_path = 'codebot/model_pretrain.pt'

# Hyperparameters set context length, model size, batch size, learning rate, and training duration.
context_len = 256  # Context/block size T: number of preceding tokens the model can use per example.
vocab_size = 1000
batch_size = 32  # B examples per update; the loader stacks B independent token windows.
learning_rate = 3e-4  # The learning rate scales each optimizer update.
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1

# Dataset: PyTorch asks for the number of examples and one input/target pair by index.
# For [a,b,c,d], x=[a,b,c] and y=[b,c,d]: every position learns next-token prediction.
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        # torch.tensor converts NumPy IDs to torch.long, the integer type nn.Embedding requires.
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y

# Data preparation: read IDs and use DataLoader to create shuffled mini-batches.
# numpy.fromfile reads binary token IDs; it does not tokenize raw text.
ids = np.fromfile(data_path, dtype=np.uint16)
dataset = TokenDataset(ids, context_len)
# PyTorch DataLoader batches windows as (B,T); shuffling changes examples, not token order inside a window.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# GPT predicts vocabulary scores; AdamW applies parameter updates.
# codebot.GPT maps input IDs (B,T) through transformer layers to vocabulary logits (B,T,V).
model = GPT(
    vocab_size=vocab_size,
    max_context_len=context_len,
    embed_dim=embed_dim,
    n_head=n_head,
    n_layer=n_layer,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate
).to(device)
# torch.optim.AdamW applies adaptive gradient updates with decoupled weight decay.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
# Japanese label means “Parameter count”; preserve output.
print(f"パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

losses = []
data_iter = cycle(dataloader)  # itertools.cycle repeats the loader after an epoch is exhausted.
pbar = tqdm(range(max_iters))  # tqdm displays progress and the latest loss.

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # Each output position scores all possible next-token IDs.
    logits = model(batch_x)
    # Cross-entropy penalizes low probability on the correct next ID; torch.nn.functional.cross_entropy compares BT rows (BT,V) with BT target IDs.
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    # Clear prior gradients because PyTorch accumulates .grad values by default.
    optimizer.zero_grad()
    # Autograd calculates gradients: derivatives showing how each parameter affects loss.
    loss.backward()
    # One optimizer step applies AdamW updates using the gradients just computed.
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# Save results: plot training losses and write the learned checkpoint.
# matplotlib.pyplot creates the learning-curve figure; plotting records does not affect training.
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_pretrain.png')

# GPT.save stores model weights and architecture metadata for GPT.load_from.
model.save(model_save_path)

# ===== CONCEPT NOTES =====
# CONCEPT: Causal language-model pretraining learns next-token prediction from code.
# GOAL: Fit codebot.GPT on the token stream; this starts the chapter's training flow
# and produces the pretrained checkpoint consumed by ch03/03_sft.py.
# INPUT: codebot/tiny_codes.bin is read as uint16 token IDs. Each example uses a
# context of 256 IDs and its one-token-shifted targets. The model uses vocabulary
# size 1,000, embedding width 384, 6 attention heads, 6 layers, FFN width 1,536,
# dropout 0.1, batch size 32, and AdamW learning rate 3e-4 for 20,000 iterations.
# OUTPUT: tqdm prints the current loss and the script prints the parameter count;
# loss_pretrain.png is written, then codebot/model_pretrain.pt is saved.
# HOW INPUT BECOMES OUTPUT:
# 1. np.fromfile loads the pre-tokenized uint16 stream; TokenDataset stores it as
#    long tensors and slices overlapping context/next-token pairs.
# 2. DataLoader shuffles those windows; cycle lets the loop request 20,000 batches.
# 3. batch_x and batch_y move to the selected device; GPT(batch_x) returns logits
#    shaped (batch, context, vocabulary), compared with batch_y by cross entropy.
# 4. zero_grad, backward, and optimizer.step update GPT with AdamW; each loss is
#    recorded, plotted, and the final model is saved with its architecture settings.
# NOTES: The file already contains token IDs, so this script does not invoke the
# tokenizer (tokenizer_path is declared but unused). The loss plot and checkpoint
# are overwritten when run again. On a CPU-only machine, all 20,000 updates train
# on CPU and can be very slow; no GPU is required by this script.
