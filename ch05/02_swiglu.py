# Concept: the SwiGLU gated feed-forward network used in transformer blocks.
# Goal: show how two learned projections form a SiLU-gated hidden representation.
# The module transforms the final feature dimension and preserves leading dimensions.
# This file defines the layer only; it does not run a training example.

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU(x)=x*sigmoid(x), a smooth activation that can retain small negative outputs.
# torch.sigmoid is elementwise and preserves x’s shape.
def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)

        # torch.nn.Linear applies a learned matrix to the last axis: (...,x_dim)->(...,hidden_dim).
        # W and V form content/gate projections; O maps hidden width back to x_dim.
        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)

    def forward(self, x):
        # For x=(B,C,x_dim), a and b each have shape (B,C,hidden_dim).
        a = self.W(x)
        b = self.V(x)

        # torch.nn.functional.silu gates b elementwise; product remains (B,C,hidden_dim).
        gated = F.silu(a) * b
        out = self.O(gated)
        return out

# ===== CONCEPT NOTES =====
# CONCEPT: SwiGLU is a gated feed-forward transformation: SiLU(Wx) is multiplied elementwise by Vx, then projected back with O. SiLU(x)=x*sigmoid(x); the learned gate lets the network control which hidden features pass through.
# GOAL: Introduce the LLaMA-style feed-forward component used in the transformer block in the next model example. This file defines a reusable module but does not instantiate a GPT or train it.
# INPUT: A floating-point tensor x with shape (..., x_dim), two bias-free input linear layers, and optional hidden_dim. If omitted, hidden_dim=int(x_dim*8/3), e.g. x_dim=384 gives 1024.
# OUTPUT: A tensor with shape (..., x_dim). There is no print, plot, or saved checkpoint in this file.
# HOW INPUT BECOMES OUTPUT: 1) Compute a=W(x) and b=V(x), each shaped (..., hidden_dim). 2) Apply SiLU elementwise to a and multiply by b, retaining (..., hidden_dim). 3) Apply O to produce (..., x_dim).
# NOTES: The class has trainable parameters in W, V, and O; the standalone silu helper is also provided, although forward uses torch.nn.functional.silu. CPU is sufficient for the module; callers must supply tensors whose final dimension matches x_dim.
