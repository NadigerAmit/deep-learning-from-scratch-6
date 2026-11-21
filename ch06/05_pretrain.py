# Concept: causal-language-model pretraining with batches, validation, and LR warmup/decay.
# Goal: train a GPT on tokenized TinyStories data and save intermediate/final weights.
# The script also plots validation loss and uses BF16 autocast where supported.
# Paths are resolved from the repository root before data/model files are opened.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device


def get_lr(it, max_lr, warmup_iters, max_iters):
    # Linear warmup gradually raises the learning rate to max_lr.
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # Linear annealing reduces the rate to zero by max_iters; this is not cosine decay.
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0


# data is a 1-D NumPy token stream; each x row has C IDs and y is the next-token shift.
# torch.stack builds (B,C) long tensors; .to(device) moves them to CUDA/MPS/CPU.
def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)

        ix = ix[ix + context_len + 1 < len(data)]
        if len(ix) == 0:
            return None, None

    # Stack equal-length input and one-token-shifted target slices into a batch.
    # torch.from_numpy wraps int64 slices; x,y are (B,C), and y[t] is the next token after x[t].
    x = torch.stack([torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)

# model.eval() and torch.no_grad disable training behavior and gradient storage for validation.
def evaluate(model, val_data, context_len, batch_size, device):
    """Evaluate sequential validation batches without updating model weights."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_start = len(val_data) - context_len - 1
    num_batches = (max_start // context_len) // batch_size + 1

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Validation"):
            offset = batch_idx * batch_size * context_len

            x, y = get_batch(val_data, context_len, batch_size, device,
                        random=False, offset=offset)

            if x is None:
                break

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    y.view(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens

# Device and file-path configuration.
device = get_device()
data_path = 'storybot/tiny_stories_train.bin'
val_data_path = 'storybot/tiny_stories_valid.bin'
tokenizer_path = 'storybot/merge_rules.pkl'
model_save_path = 'storybot/model_pretrain.pt'

# Training/model settings: C=context length, V=vocabulary size, B=batch size.
context_len = 256
vocab_size = 10000
batch_size = 32
learning_rate = 0.001  # max_lr
warmup_iters = 200  # Number of initial linear warmup iterations.
max_iters = 40000
embed_dim = 512
n_head = 16
n_layer = 4
ff_dim = 1344
theta = 10000
eval_iters = 500
grad_clip = 1.0
save_iters = [500, 5000]  # Iterations at which intermediate checkpoints are saved.

# NumPy memmap reads uint16 token IDs on demand without copying the whole file into RAM.
train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

# Load BPE rules, construct GPT, and create AdamW to update trainable parameters.
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT(
    vocab_size, context_len, embed_dim, n_head, n_layer, ff_dim, theta
).to(device)
# torch.optim.AdamW uses gradient moments and decoupled weight decay; this script does not use W&B.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
# Japanese output means “parameter count.”
print(f"パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

pbar = tqdm(range(max_iters))

val_loss = float('inf')
val_losses = []
val_iters = []

for i in pbar:
    # Update AdamW’s LR using the scalar warmup/decay schedule.
    lr = get_lr(i, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_x, batch_y = get_batch(train_data, context_len, batch_size, device)

    # Clear stale gradients before backward; PyTorch otherwise accumulates them.
    optimizer.zero_grad()

    # Forward logits and cross-entropy under BF16 autocast; loss averages B*C targets.
    # torch.amp.autocast lowers eligible math precision on the chosen device; logits are (B,C,V).
    # BF16 usually needs no GradScaler; this loop has no scaler because it uses BF16, not FP16.
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    # Backpropagate scalar loss; clip the global gradient norm before AdamW updates parameters.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    # Save weights and architecture metadata at configured checkpoints.
    if i in save_iters:
        save_path = f'storybot/model_iter_{i}.pt'
        model.save(save_path)
        # Japanese output says that the model was saved at this iteration.
        print(f"\nモデルを保存しました（イテレーション {i}）: {save_path}")

    # Evaluate periodically on sequential validation batches.
    if (i % eval_iters) == 0 or i == max_iters - 1:
        val_loss = evaluate(model, val_data, context_len, batch_size, device)
        val_losses.append(val_loss)
        val_iters.append(i)
    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val_loss': f'{val_loss:.6f}'})


# Matplotlib plots recorded scalar validation losses against iteration numbers.
plt.figure(figsize=(10, 6))
plt.plot(val_iters, val_losses)
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.savefig('loss_val.png')

model.save(model_save_path)

# ===== CONCEPT NOTES =====
# CONCEPT: This script performs next-token language-model pretraining with cross-entropy, AdamW, gradient clipping, a linear warmup/decay schedule, periodic validation, and BF16 autocast. Inputs are sequences of token IDs; each target is the same sequence shifted left by one token.
# GOAL: Train the Storybot GPT on TinyStories after the architecture/tokenizer work in earlier chapters. The resulting checkpoint and validation curve are inputs to the generation and later preference-tuning/judging scripts.
# INPUT: From the repository root it reads storybot/tiny_stories_train.bin and storybot/tiny_stories_valid.bin as uint16 memmaps and loads storybot/merge_rules.pkl. Settings: context_len=256, vocab_size=10000, batch_size=32, max learning rate=0.001, warmup_iters=200, max_iters=40000, embed_dim=512, n_head=16 (head_dim=32), n_layer=4, ff_dim=1344, theta=10000, eval_iters=500, grad_clip=1.0; save iterations are [500,5000].
# OUTPUT: Prints parameter count and a tqdm loss/validation-loss status. Saves checkpoints storybot/model_iter_500.pt and storybot/model_iter_5000.pt when those loop indices are reached, writes loss_val.png, and finally saves storybot/model_pretrain.pt. Validation returns mean token cross-entropy over sequential batches.
# HOW INPUT BECOMES OUTPUT: 1) Read each uint16 token stream and load the BPE tokenizer. 2) Randomly choose 32 starting offsets; build x with shape (32,256) and next-token y with the same shape. 3) Update LR; run GPT to logits (32,256,10000) under BF16 autocast, flatten positions for cross-entropy, backpropagate, clip global gradient norm to 1.0, and step AdamW. 4) Every 500 loop indices (also at index 0), evaluate sequential validation batches and accumulate summed loss divided by token count; save at indices 500 and 5000. 5) Plot recorded validation losses and save the final checkpoint.
# NOTES: This is a long training job (up to 40000 iterations) and must not be run for this documentation task; CPU-only execution can take a very long time, while autocast BF16 support depends on PyTorch/device. All listed data files and tokenizer pickle must exist. The script overwrites its checkpoint outputs and loss_val.png. model_save_path is storybot/model_pretrain.pt; it is written only after training completes. The batch sampler requires each stream to exceed context_len+1 tokens.
