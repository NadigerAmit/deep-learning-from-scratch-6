# Concept: Transformer blocks stabilize feature scales with LayerNorm, mix token context
# with attention, then transform each token through a nonlinear feed-forward network (FFN).
# Residual additions preserve an information path around both sublayers. GELU is a smooth activation.
# PyTorch nn provides parameters, layers, and Sequential; attention is imported from codebot.model.
# This file defines components only and does not execute a forward pass.

# Make the repository root importable so this lesson can load its shared codebot.model module.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # Resolve and enter the directory above ch02.
sys.path.append('.')  # Python import search path; append the current repository directory.
import torch
import torch.nn as nn
from codebot.model import MultiHeadAttention


# nn.Module is PyTorch's base class, allowing components and trainable parameters to be registered.
class LayerNorm(nn.Module):
    # LayerNorm normalizes each token's feature vector independently, not across the batch.
    # Learned gamma (scale) and beta (shift) let training restore useful feature distributions.
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))  # Trainable per-feature scale, shape (E,).
        self.beta = nn.Parameter(torch.zeros(embed_dim))  # Trainable per-feature offset, shape (E,).
        self.eps = 1e-5  # Small positive constant prevents division by zero for constant inputs.

    def forward(self, x):  # Any leading axes are kept; the final axis is the feature axis E.
        mean = x.mean(dim=-1, keepdim=True)  # PyTorch mean over E; keepdim makes broadcasting work.
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Population variance over E for each vector.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # Center and scale to roughly zero mean/unit variance.
        return self.gamma * norm_x + self.beta  # Broadcast learned scale/shift; output shape equals x.


class GELU(nn.Module):
    # GELU smoothly gates inputs using x times the standard-normal CDF, unlike ReLU's hard cutoff.
    def forward(self, x):  # PyTorch elementwise operations preserve x's shape.
        # Tanh approximation: GELU(x) ~= 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
        # For a small negative input this gives a small negative output; large positive x passes through.
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # Scalar constant for the tanh approximation.
            (x + 0.044715 * torch.pow(x, 3))  # torch.pow cubes each feature value.
        ))

class FFN(nn.Module):
    # The FFN processes each token's features independently; attention is what mixes information across tokens.
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)  # Common Transformer choice: expand features fourfold before compressing.

        # nn.Sequential runs modules in order on the last dimension of input (...,x_dim).
        # The expansion creates capacity; GELU adds nonlinearity; the final layer returns to x_dim.
        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),  # (...,x_dim) -> (...,hidden_dim).
            GELU(),  # Elementwise smooth activation; same shape.
            nn.Linear(hidden_dim, x_dim),  # (...,hidden_dim) -> (...,x_dim).
            nn.Dropout(dropout_rate)  # Regularize the output during training; shape unchanged.
        )

    def forward(self, x):
        return self.layers(x)  # Applies the Sequential FFN at every batch/position independently.


class Block(nn.Module):
    # A pre-norm residual block normalizes before each sublayer and adds its output to the input.
    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):  # Input/output shape is (B,C,E).
        # LayerNorm stabilizes features before attention; attention mixes positions. The residual
        # x + ... preserves the original representation and provides a direct gradient path.
        x = x + self.attn(self.norm1(x))
        # The second pre-norm FFN transforms each token's features; another residual keeps shape (B,C,E).
        x = x + self.ffn(self.norm2(x))
        return x

# ===== CONCEPT NOTES =====
# CONCEPT: A pre-norm Transformer block applies LayerNorm, causal multi-head attention, a
# residual addition, then another LayerNorm, token-wise FFN, and residual addition.
# GELU is the smooth elementwise activation between the FFN's expanding/compressing linears.
# GOAL: Add normalization, nonlinear per-token processing, and residual paths to attention
# from 08_multi_head.py. 10_gpt2.py imports a separate Block from codebot.model.
# INPUT: Definitions accept feature vectors on the final axis. FFN maps x_dim -> hidden_dim
# (default 4*x_dim) -> x_dim; Block input is (B,C,E). Attention comes from codebot.model.
# OUTPUT: No forward pass, print, or file write happens here. LayerNorm, GELU, FFN, and Block
# preserve the input's leading dimensions; Block output shape is (B,C,E).
# HOW INPUT BECOMES OUTPUT:
# 1. LayerNorm computes mean and population variance over only the final feature axis, then
#    divides by sqrt(var+1e-5) and applies learned per-feature gamma and beta.
# 2. GELU approximates x*Phi(x) with 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
# 3. FFN uses torch.nn.Linear to expand/contract each token, GELU between, then dropout.
# 4. Block adds attention(LayerNorm(x)) to x, then adds FFN(LayerNorm(x)) to that result.
# NOTES: PyTorch nn.Parameter marks gamma/beta trainable; unbiased=False chooses population
# variance. The GELU constant tensor is created on PyTorch's default device. The imported
# codebot.model.MultiHeadAttention supplies causal attention and dropout.
