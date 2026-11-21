# Concept: pretrain a larger GPT on OpenWebText-style train/validation token streams.
# Goal: learn next-token prediction with RoPE, grouped-query attention, and SwiGLU.
# Optional distributed training uses DDP/NCCL; otherwise get_device selects a device.
# This chapter's pretraining checkpoint is the input to ch09/02_generate.py.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from webbot.model import GPT
from webbot.utils import get_device


# --- Distributed setup: DDP coordinates multiple GPUs; otherwise use one local device. ---
# torchrun sets RANK for distributed jobs; without it this script runs as one process. The output guard below uses local rank.
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    seed_offset = int(os.environ['RANK'])
else:
    device = get_device()
    local_rank = 0
    seed_offset = 0

is_main = (not ddp) or (local_rank == 0)
torch.manual_seed(42 + seed_offset)

# --- Initialize W&B on local rank 0; this is intended for single-node DDP to avoid duplicate runs. ---
if is_main:
    # wandb.init starts an experiment-tracking run; hyperparameters are added below.
    wandb.init(project='webbot-pretrain', config={})  # Start a W&B run; hyperparameters are added below.


# Warmup avoids a large abrupt initial update; linear decay reduces step sizes near the end.
def get_lr(it, max_lr, warmup_iters, max_iters):
    # Warmup gradually increases learning rate from zero to max_lr.
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # Linear decay gradually reduces learning rate from max_lr to zero.
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0


# Return x,y tensors shaped (B,T); y is shifted one token to teach next-token prediction.
def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)
        ix = ix[ix + context_len + 1 < len(data)]
        if len(ix) == 0:
            return None, None

    # Convert memmap slices to int64 tensors because embedding lookups require integer IDs.
    x = torch.stack([torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)


# Validation measures per-token loss on held-out IDs without updating weights.
def evaluate(model, val_data, context_len, batch_size, device):
    """Evaluate every validation window in order without changing model weights."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_start = len(val_data) - context_len - 1
    num_batches = (max_start // context_len) // batch_size + 1

    # No backward pass is needed for evaluation, saving memory.
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating", leave=False):
            offset = batch_idx * batch_size * context_len

            x, y = get_batch(val_data, context_len, batch_size, device,
                        random=False, offset=offset)

            if x is None:
                break

            # torch.amp.autocast uses lower-precision CUDA math where supported to reduce memory or use faster kernels.
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass maps IDs (B,T) to scores (B,T,V).
                logits = model(x)
                # torch.nn.functional.cross_entropy compares flattened (B*T,V) scores with (B*T,) targets; sum for token-weighted averaging.
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    y.view(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += x.numel()

    model.train()
    return total_loss / total_tokens

# --- Context length is T; micro-batch is per pass; accumulation sets effective batch size. ---
vocab_size = 50000
context_len = 1024  # Block size T: number of preceding token positions processed per example.
embed_dim = 768
n_head = 12
n_kv_head = 4
n_layer = 12
ff_dim = 2048
theta = 10000

micro_batch_size = 32  # Examples in one forward/backward pass on each process.
accumulation_steps = 4
learning_rate = 6e-4  # Initial/max scale of each AdamW update before the schedule modifies it.
warmup_iters = 500
max_iters = 100000
grad_clip = 1.0
eval_interval = 1000

# --- Record model and optimizer settings in W&B for later interpretation. ---
if is_main:
    wandb.config.update({
        'vocab_size': vocab_size, 'context_len': context_len,
        'embed_dim': embed_dim, 'n_head': n_head, 'n_kv_head': n_kv_head,
        'n_layer': n_layer, 'ff_dim': ff_dim, 'theta': theta,
        'micro_batch_size': micro_batch_size,
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': micro_batch_size * accumulation_steps,
        'learning_rate': learning_rate, 'warmup_iters': warmup_iters,
        'max_iters': max_iters, 'grad_clip': grad_clip,
    })

# --- NumPy memory maps expose token streams without eagerly copying them into RAM. ---
train_data_path = 'webbot/owt_train.bin'
val_data_path = 'webbot/owt_valid.bin'
model_save_path = 'webbot/model_pretrain.pt'

# numpy.memmap is disk-backed; only selected batch slices are copied into memory.
train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

if is_main:
    # Japanese label means “Training data”; preserve output.
    print(f"学習データ: {len(train_data):,} tokens")
    # Japanese label means “Validation data”; preserve output.
    print(f"検証データ: {len(val_data):,} tokens")

# --- Configure web-text GPT, which maps IDs to vocabulary logits. ---
model = GPT(
    vocab_size, context_len, embed_dim, n_head,
    n_kv_head, n_layer, ff_dim, theta
).to(device)

if is_main:
    num_params = sum(p.numel() for p in model.parameters())
    # Japanese label means “Parameter count”; preserve output.
    print(f"パラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")

# torch.compile optimizes graph execution; it changes performance, not the objective.
model = torch.compile(model)

if ddp:
    # DistributedDataParallel synchronizes gradients across GPU processes during backward.
    model = DDP(model, device_ids=[local_rank])

# --- AdamW maintains gradient statistics and updates trainable parameters. ---
# AdamW maintains adaptive gradient statistics; scheduled LR is set on each iteration.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Accumulate micro-batch gradients before each optimizer step. ---
pbar = tqdm(range(max_iters), disable=not is_main)  # tqdm provides a progress bar on rank 0 only.
val_loss = float('inf')
val_losses = []
val_iters = []

for step in pbar:
    # Set this iteration’s learning rate from the warmup/decay schedule.
    lr = get_lr(step, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Clear once per optimizer update; gradients from all micro-batches must accumulate.
    optimizer.zero_grad()

    # Accumulation emulates a larger batch without storing all examples at once.
    for micro_step in range(accumulation_steps):
        batch_x, batch_y = get_batch(train_data, context_len,
                                     micro_batch_size, device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Model output shape is (B,T,V), one vocabulary score vector per position.
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   batch_y.view(-1))
            # Scale loss so accumulated gradients average micro-batches rather than sum them.
            loss = loss / accumulation_steps

        # DDP skips intermediate communication; the final backward syncs accumulated gradients.
        if ddp and micro_step < accumulation_steps - 1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    # Clip global gradient norm to prevent unusually large updates from destabilizing training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # One AdamW step applies accumulated gradients to trainable parameters.
    optimizer.step()

    # W&B logs training metrics; logging does not affect gradients.
    train_loss = loss.item() * accumulation_steps
    if is_main:
        wandb.log({'train/loss': train_loss, 'train/lr': lr}, step=step)

    # Evaluate held-out data periodically without an optimizer step.
    if is_main and ((step % eval_interval) == 0 or step == max_iters - 1):
        raw_model = model.module if ddp else model
        val_loss = evaluate(raw_model, val_data, context_len,
                           micro_batch_size, device)
        val_losses.append(val_loss)
        val_iters.append(step)
        print(f"\nstep {step}: val_loss = {val_loss:.4f}")

        wandb.log({'val/loss': val_loss}, step=step)

    if is_main:
        pbar.set_postfix({'loss': f'{train_loss:.4f}',
                          'val_loss': f'{val_loss:.4f}', 'lr': f'{lr:.2e}'})

# --- Save the validation-loss curve and trained model checkpoint. ---
if is_main:
    # matplotlib.pyplot creates the validation-loss plot; these metrics are diagnostic, not model parameters.
    plt.figure(figsize=(10, 6))
    plt.plot(val_iters, val_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig('webbot_val_loss.png')

    raw_model = model.module if ddp else model
    raw_model.save(model_save_path)
    # Japanese status means “Model saved”; preserve output.
    print(f"\nモデル保存: {model_save_path}")

if is_main:
    wandb.finish()

if ddp:
    dist.destroy_process_group()

# ===== CONCEPT NOTES =====
# CONCEPT: Causal language-model pretraining optimizes a GPT on packed web-text IDs;
# this model uses rotary positions, grouped-query attention, RMSNorm, and SwiGLU.
# GOAL: Train the webbot model from scratch and save a checkpoint for
# ch09/02_generate.py, which demonstrates generation from the trained model.
# INPUT: webbot/owt_train.bin and webbot/owt_valid.bin are uint16 token streams read
# as memory maps. Model settings: vocabulary 50,000; context 1,024; embedding 768;
# 12 query heads, 4 key/value heads, 12 layers, FFN width 2,048, and theta 10,000.
# Training uses micro-batches of 32 and accumulates 4 steps (128 examples per process;
# multiply by DDP process count for the global batch), LR 6e-4, 500 warmup iterations,
# and 100,000 total iterations,
# gradient clipping at 1.0, and validation every 1,000 iterations.
# OUTPUT: In single-node mode, local rank 0 prints dataset token counts, parameter count, and validation loss;
# tqdm/W&B log training metrics to project webbot-pretrain. webbot_val_loss.png and
# webbot/model_pretrain.pt are written by the main process.
# HOW INPUT BECOMES OUTPUT:
# 1. DDP is initialized only when RANK is set; otherwise get_device chooses the
#    available device. In single-node DDP, local rank 0 initializes W&B; all ranks set rank-adjusted seeds.
# 2. np.memmap exposes train/validation token IDs. get_batch selects random windows
#    for training or ordered windows for validation, returning x and shifted y.
# 3. GPT maps IDs to hidden states, applies causal attention and transformer blocks,
#    and returns next-token logits. Cross entropy computes summed validation loss or
#    mean micro-batch training loss; autocast requests CUDA bfloat16.
# 4. Four micro-batches accumulate gradients (DDP sync is skipped until the last);
#    gradients are clipped and AdamW steps after a linear warmup/decay LR schedule.
# 5. The is_main process evaluates, logs, and saves periodically (local rank 0 in a single-node run); it writes the validation-loss plot and
#    model checkpoint; W&B is finished and the DDP process group is destroyed.
# NOTES: On a CPU-only machine, this script uses the non-DDP device path, but its
# autocast calls explicitly target CUDA and may be disabled with a PyTorch warning;
# torch.compile and the 100,000-step, 768-wide model can make CPU training extremely
# slow. DDP mode requires CUDA/NCCL and is not suitable for a CPU-only host. Both
# token files must exist locally; this script does not download them. W&B logging
# requires its normal local account/configuration. Plot and checkpoint are overwritten.
