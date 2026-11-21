# Concept: after attention weights blend the value vectors, a learned output projection
# maps their combined features back to the model's embedding width. This keeps the result
# compatible with later residual additions to the original token representation.
# PyTorch nn.Linear projects feature axes; torch.matmul performs the batched attention products.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)  # Learn the output feature projection D -> E.
        self.key_dim = key_dim

    def forward(self, x):  # x: (B,C,E); all projections act on the final feature dimension.
        Q = self.W_q(x)  # (B,C,E) -> (B,C,D): the query at each token.
        K = self.W_k(x)  # (B,C,E) -> (B,C,D): the key compared to every query.
        V = self.W_v(x)  # (B,C,E) -> (B,C,D): the content to retrieve.

        # Transpose only sequence/feature axes so each query can dot with all keys.
        K_t = K.transpose(-2, -1)  # (B,C,D) -> (B,D,C); PyTorch Tensor.transpose.
        scores = torch.matmul(Q, K_t)  # (B,C,D) @ (B,D,C) -> (B,C,C) similarity scores.
        scores = scores / (self.key_dim ** 0.5)  # Scale logits to avoid overly sharp softmax.

        # Build a shared causal mask: the strict upper triangle contains future positions.
        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))  # torch.tril: shape (C,C).
        scores = scores.masked_fill(mask == 0, float('-inf'))  # masked future weights become zero.

        weights = F.softmax(scores, dim=-1)  # Normalize each query across C allowed keys.
        hidden = torch.matmul(weights, V)  # (B,C,C) @ (B,C,D) -> (B,C,D) weighted values.

        # W_o maps the aggregated D features to E output features at each sequence position.
        output = self.W_o(hidden)  # nn.Linear: (B,C,D) -> (B,C,E).

        return output

attention = Attention(embed_dim=256, key_dim=64)  # Create the layer and its trainable projection matrices.
x = torch.randn(2, 5, 256)  # torch.randn sample input: batch 2, five positions, 256 features.
y = attention(x)  # Invoke forward; output shape remains (2,5,256) after W_o.

print("入力形状:", x.shape)
print("出力形状:", y.shape)

# ===== CONCEPT NOTES =====
# CONCEPT: Q and K determine how much each position attends to each earlier position; the
# resulting weights sum to 1 per query and mix V vectors. W_o then projects that mixture.
# GOAL: Continue causal attention from 06_attn_mask.py. Projecting the result back to E
# features allows it to join an embedding-width residual stream in a Transformer block.
# INPUT: Random x has shape (2,5,256); Q, K, and V have width D=64; W_o maps D=64 to E=256.
# OUTPUT: Prints input and output shape torch.Size([2,5,256]); values vary by run. Nothing is saved.
# HOW INPUT BECOMES OUTPUT:
# 1. Three torch.nn.Linear maps turn (2,5,256) into Q,K,V each shaped (2,5,64).
# 2. K transpose and batched matmul yield (2,5,5) scores; scale, mask, then softmax over keys.
# 3. Multiply weights (2,5,5) by V (2,5,64) to make hidden states (2,5,64).
# 4. W_o applies a 64-to-256 learned map per token, yielding output (2,5,256).
# NOTES: All four projections have bias=False. torch.tril constructs a (5,5) causal mask
# that broadcasts over batches; F.softmax is torch.nn.functional.softmax. Attention weights aren't returned.
