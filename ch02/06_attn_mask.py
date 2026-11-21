# Concept: self-attention lets every token retrieve information from tokens in the same sequence.
# A causal (look-ahead) mask restricts token i to itself and earlier tokens, which is essential
# when predicting the next token: future text must not leak into the prediction.
# Q asks, K is matched against Q, and V carries the information to combine; torch.nn.Linear
# learns projections, while torch.matmul and torch.nn.functional.softmax perform attention.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        # nn.Linear creates trainable maps along the last dimension: input (...,E) -> (...,D).
        # Separate maps let the same token embedding play three different roles: query, key, value.
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_dim = key_dim

    def forward(self, x):  # Input layout: (B=batch, C=positions, E=embedding features).
        # PyTorch nn.Linear applies its learned matrix to the last axis independently at each token.
        Q = self.W_q(x)  # Query tensor (B,C,E) -> (B,C,D): what each position seeks.
        K = self.W_k(x)  # Key tensor (B,C,E) -> (B,C,D): what each position can match.
        V = self.W_v(x)  # Value tensor (B,C,E) -> (B,C,E): information returned after matching.

        # Compare every query position with every key position. The last two dimensions of
        # K are swapped so batched matrix multiplication can produce one C-by-C score matrix.
        K_t = K.transpose(-2, -1)  # PyTorch transpose: (B,C,D) -> (B,D,C), swap sequence/feature axes.
        scores = torch.matmul(Q, K_t)  # torch.matmul: (B,C,D) @ (B,D,C) -> (B,C,C).
        # Divide by sqrt(D) to keep dot-product magnitudes moderate before softmax.
        scores = scores / (self.key_dim ** 0.5)

        # torch.tril makes a lower triangle of ones: row i permits key positions <= i.
        # The (C,C) mask broadcasts across B batches; zero entries mark forbidden future tokens.
        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))  # torch.ones creates (C,C); tril retains its lower triangle.
        # -inf is used because exp(-inf)=0: masked positions get exactly zero softmax weight.
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Tensor.masked_fill replaces disallowed (B,C,C) scores.
        # F.softmax is torch.nn.functional.softmax; dim=-1 normalizes over the key-position axis.
        weights = F.softmax(scores, dim=-1)  # Shape (B,C,C); each query row sums to 1.

        # Mix value vectors: (B,C,C) @ (B,C,E) -> one output feature vector per query position.
        output = torch.matmul(weights, V)  # torch.matmul output shape (B,C,E).
        return output

# Construct the nn.Module: it registers three learned projections, mapping E->D, E->D, and E->E.
attention = Attention(embed_dim=256, key_dim=64)  # PyTorch module with trainable weights.
x = torch.randn(2, 5, 256)  # torch.randn creates random input (B,C,E)=(2,5,256).
y = attention(x)  # Calls Module.forward; output has the same (2,5,256) layout.

print("入力形状:", x.shape)
print("出力形状:", y.shape)

# ===== CONCEPT NOTES =====
# CONCEPT: Causal self-attention builds a C-by-C distribution for each batch item. At row i,
# a lower-triangular mask allows keys 0 through i and excludes every later (future) token.
# Replacing forbidden logits with -inf makes their softmax weights exactly zero.
# GOAL: Add learned Q/K/V projections, scaling by sqrt(D), and causal masking to earlier
# attention math. 07_attn_value.py adds the output projection after value aggregation.
# INPUT: x is random with shape (B,C,E)=(2,5,256). nn.Linear maps its final feature axis so
# Q,K have D=64 and V has E=256. No external data or files are read.
# OUTPUT: Prints torch.Size([2,5,256]) for input and output. Values vary with random input
# and initialization; this demonstration trains nothing and writes no file.
# HOW INPUT BECOMES OUTPUT:
# 1. Q=W_q(x), K=W_k(x) give (2,5,64); V=W_v(x) gives (2,5,256).
# 2. K.transpose(-2,-1) is (2,64,5); torch.matmul(Q,K_t) gives scores (2,5,5).
# 3. Divide by sqrt(64)=8. For five positions the mask rows permit [0], [0..1], ..., [0..4].
# 4. F.softmax(scores,dim=-1) normalizes each row over keys, then weights@V gives (2,5,256).
# NOTES: torch.nn provides Module/Linear; torch.nn.functional provides softmax. The mask
# is on the score tensor's device and broadcasts from (5,5) over two batches. Only output is returned.
