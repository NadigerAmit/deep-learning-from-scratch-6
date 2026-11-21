# Concept: multi-head attention runs several independent attention calculations in parallel.
# Each head can learn a different feature subspace or relationship; their outputs are concatenated
# and projected back to the model width. PyTorch nn.Linear combines head projections efficiently.
# This file shows manual tensor steps, then a reusable torch.nn.Module with dropout regularization.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Name the axes: B=batch items, C=token positions, E=embedding features, H=heads, D=features per head.
B = 2   # Two independent sequences are processed together.
C = 4   # Each sequence has four token positions.
E = 16  # Each token is represented by 16 input features.
H = 3   # Three attention heads operate in parallel.
D = 8   # Each head projects to eight query/key/value features.

# torch.randn (PyTorch) draws a random float tensor; shape is (batch, positions, features).
x = torch.randn(B, C, E)

# Project once to H*D=24 channels, then reshape channels into H separate D-wide heads.
# One larger nn.Linear is equivalent to concatenating per-head linear projections.
W_q = nn.Linear(E, H*D, bias=False)
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)  # (B, C, H*D)
V = W_v(x)  # (B, C, H*D)

# view splits the last H*D feature axis into head and per-head feature axes; transpose
# places H before C so batched matmul treats each head as an independent attention batch.
Q = Q.view(B, C, H, D).transpose(1, 2)  # (B,C,H*D)->(B,C,H,D)->(B,H,C,D).
K = K.view(B, C, H, D).transpose(1, 2)  # Same rearrangement for keys.
V = V.view(B, C, H, D).transpose(1, 2)  # Same rearrangement for values.

# PyTorch matmul broadcasts over B and H: each query token compares with each key token.
scores = torch.matmul(Q, K.transpose(-2, -1))  # (B,H,C,D) @ (B,H,D,C) -> (B,H,C,C).
scores = scores / (D ** 0.5)  # Scale each head's logits by sqrt(head width), here sqrt(8).

# Causal mask: for each query row, allow only current and earlier key positions.
mask = torch.tril(torch.ones(C, C, device=scores.device))  # (C,C) lower triangle, shared across batches and heads.
scores = scores.masked_fill(mask == 0, float('-inf'))  # Replace future score entries by negative infinity.

# F.softmax is torch.nn.functional.softmax; dim=-1 means normalize over the C key positions.
# Every allowed row's attention weights sum to 1; masked future positions have zero weight.
weights = F.softmax(scores, dim=-1)  # (B,H,C,C), a key-position distribution per query/head.
hidden = torch.matmul(weights, V)   # (B,H,C,C) @ (B,H,C,D) -> (B,H,C,D).

# Restore head next to the feature axis so all H*D features can be concatenated.
hidden = hidden.transpose(1, 2)  # (B,H,C,D) -> (B,C,H,D), reversing the earlier axis move.
hidden = hidden.contiguous().view(B, C, H*D)  # Merge H,D -> H*D; contiguous makes memory layout safe for view.

# Output nn.Linear mixes concatenated head features and maps width H*D back to E.
W_o = nn.Linear(H*D, E, bias=False)
output = W_o(hidden)  # (B, C, E)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # nn.Dropout randomly zeros a fraction of values during training as regularization;
        # attention dropout is applied to weights and output dropout to projected features.
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, E = x.shape  # Read the runtime input axes: batch, positions, and input features.
        H, D = self.n_head, self.head_dim  # Number of heads and features in each head.

        # Three learned PyTorch projections create Q, K, and V; each maps E -> H*D per token.
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split the concatenated channels into H heads and move the head axis before positions.
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # torch.matmul: (B,H,C,D) @ (B,H,D,C) -> (B,H,C,C).
        scores = scores / (D ** 0.5)  # Scale each head before softmax.

        # The lower triangle allows a position to attend only to itself and earlier positions.
        mask = torch.tril(torch.ones(C, C, device=scores.device))  # Shape (C,C); broadcasts to (B,H,C,C).
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Block future-key entries.

        # Softmax converts scores to per-query distributions; dropout regularizes these weights.
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)  # Dropout regularizes attention weights during training.
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # Rejoin the heads and map their concatenated features back to the embedding width.
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H*D)
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)  # Dropout regularizes the projected output during training.

        return output

# Example module setup. embed_dim is 512, and eight heads of width 64 yield 512 projected channels.
embed_dim = 512
n_head = 8
head_dim = 64

mha = MultiHeadAttention(embed_dim, n_head, head_dim)  # Instantiate trainable PyTorch projections and dropout layers.

# Create random test input; shape is (batch_size, context_len, embed_dim).
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# Run the module on the sample batch; output keeps batch, position, and embedding axes.
output = mha(x)  # Module call delegates to forward(); result shape is (2,10,512).
print(f"入力形状: {x.shape}")       # (2, 10, 512)
print(f"出力形状: {output.shape}") # (2, 10, 512)

# ===== CONCEPT NOTES =====
# CONCEPT: Each of H heads computes its own query-key scores and value mixture in a D-wide
# subspace. Concatenating all head outputs gives H*D features; W_o mixes them and maps to E.
# GOAL: Extend causal single-head attention (06_attn_mask.py, 07_attn_value.py) with parallel
# heads. The reusable module adds dropout regularization to weights and projected outputs.
# INPUT: Manual demo: B=2,C=4,E=16,H=3,D=8, so projections map 16 -> 24 and input is (2,4,16).
# Module demo: E=512,H=8,D=64, dropout=0.1 and x=(2,10,512). No external data is used.
# OUTPUT: Manual demo computes (2,4,16) without printing. Module demo prints input/output
# shapes torch.Size([2,10,512]); random tensors/parameters make values differ by run.
# HOW INPUT BECOMES OUTPUT:
# 1. nn.Linear maps (B,C,E) -> (B,C,H*D); view/transpose rearranges to (B,H,C,D).
# 2. torch.matmul(Q,K.transpose(-2,-1)) gives (B,H,C,C); scale by sqrt(D), mask, then softmax.
# 3. Multiply weights by V to get (B,H,C,D); module dropout may zero some probabilities.
# 4. Transpose back to (B,C,H,D), merge H and D, then W_o maps H*D -> E.
# NOTES: torch.nn supplies Linear, Module, and Dropout; torch.nn.functional supplies softmax.
# Dropout is active in training mode and inactive after eval(); weight dropout doesn't renormalize.
# E need not equal H*D because W_o explicitly maps between those widths.
