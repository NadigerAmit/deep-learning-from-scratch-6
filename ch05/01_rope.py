# Concept: rotary positional embeddings (RoPE) inside causal multi-head attention.
# Goal: rotate query/key feature pairs by token position so attention can use relative position.
# This standalone example builds a small attention layer and runs one random batch.
# The output keeps the input batch, sequence, and embedding dimensions.

import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE encodes position by rotating each adjacent Q/K pair. For pair j at position p, angle[p,j]=p/theta**(2*j/D);
# (u,v) becomes (u*cos(angle)-v*sin(angle),u*sin(angle)+v*cos(angle)).
class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0  # Each 2-D rotation requires an even feature width.
        half = key_dim // 2

        # torch.arange supplies pair indices; frequencies decrease across feature pairs.
        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ( (2.0 * half_ids) / key_dim ))  # (half,)

        positions = torch.arange(max_context_len)  # (max_context_len,)
        angles = positions[:, None] * inv_freq[None, :]  # (max_context_len, half)

        cos = torch.cos(angles)  # (max_context_len, half)
        sin = torch.sin(angles)  # (max_context_len, half)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x):
        # x is (B,H,C,D); cached trigonometric rows broadcast over batch and heads.
        batch_size, num_head, context_len, key_dim = x.shape

        # Preserve input dtype, but use float32 for stable rotation arithmetic.
        input_dtype = x.dtype
        x = x.float()

        cos = self.cos_cache[:context_len]
        sin = self.sin_cache[:context_len]

        # Pair adjacent even/odd features as 2-D rotation coordinates.
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # Rotate each pair: (u*cos-v*sin, u*sin+v*cos).
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        # Re-interleave rotated pairs to restore the original feature order.
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (batch_size, num_head, context_len, key_dim/2, 2)
        out = out.reshape(batch_size, num_head, context_len, key_dim)

        return out.to(input_dtype)  # Return in the caller’s original dtype.

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, rope=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.rope = rope

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # Rotate Q and K only; V carries content and receives no positional rotation.
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        # torch.matmul: (B,H,C,D) @ (B,H,D,C) -> scores (B,H,C,C).
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D ** 0.5)

        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # torch.nn.functional.softmax normalizes each query over permitted key positions.
        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

# Architecture widths: E=embedding size, H=head count, D=features per head.
embed_dim = 512
n_head = 8
head_dim = 64
theta = 10000
max_context_len = 1024

# Build the rotation table and attention layer; torch.nn initializes projections.
rope = RoPE(theta, head_dim, max_context_len)
mha = MultiHeadAttention(embed_dim, n_head, head_dim, rope=rope)

# torch.randn creates random floating-point activations shaped (B,C,E).
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# One forward pass returns the same outer shape, (B,C,E).
output = mha(x)
print(output.shape)  # (2, 10, 512)

# ===== CONCEPT NOTES =====
# CONCEPT: RoPE applies a position-dependent 2-D rotation to adjacent even/odd feature pairs in each query and key. For pair index j and position p, the angle is p / theta**(2j/key_dim); values use the cosine/sine rotation. Causal self-attention computes softmax(QK^T / sqrt(head_dim)) V.
# GOAL: Demonstrate rotary positions in multi-head attention, extending basic attention with position information. This is a standalone illustration rather than a training or checkpoint-loading script.
# INPUT: Module settings are embed_dim=512, n_head=8, head_dim=64, theta=10000, and max_context_len=1024. The example creates random x with shape (batch=2, context=10, embedding=512); RoPE requires an even key dimension.
# OUTPUT: Prints torch.Size([2, 10, 512]). No model or image is saved.
# HOW INPUT BECOMES OUTPUT: 1) For 64 features, build 32 inverse frequencies and cosine/sine tables for 1024 positions. 2) Project x (2,10,512) into Q/K/V with 8 heads, reshaping each to (2,8,10,64). 3) Rotate Q and K using position rows 0..9; split each head's 64 features into 32 even/odd pairs. 4) Compute attention scores of shape (2,8,10,10), scale by sqrt(64)=8, mask future positions, and softmax over keys. 5) Multiply weights by V, join heads back to (2,10,512), and apply the output projection.
# NOTES: The random sample and weights vary by run. A sequence longer than the configured context will not have enough cached RoPE rows. CPU execution is supported; no GPU or external files are required.
