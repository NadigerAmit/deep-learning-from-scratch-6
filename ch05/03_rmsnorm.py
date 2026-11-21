# Concept: root-mean-square layer normalization (RMSNorm).
# Goal: rescale each feature vector using its RMS without subtracting its mean.
# A learned per-feature gain restores the scale the model needs.
# This file contains a reusable module and no runnable demonstration.

import torch
import torch.nn as nn


# RMSNorm divides each last-axis vector by sqrt(mean(x^2)+eps), then applies learned gain gamma.
# Unlike LayerNorm, it does not subtract the mean (center the vector) and has no learned bias here.
# For [1,2,2], RMS=sqrt(3), giving about [0.577,1.155,1.155] before gamma.
class RMSNorm(nn.Module):
    def __init__(self, x):
        super().__init__()
        # torch.nn.Parameter registers one trainable scale per feature; ones begin as identity gain.
        self.gamma = nn.Parameter(torch.ones(x))
        self.eps = 1e-5

    def forward(self, x):
        # For input (...,D), mean/sqrt yield (...,1), which broadcasts over D features.
        x2 = x**2
        ms = x2.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        return self.gamma * x / rms

# ===== CONCEPT NOTES =====
# CONCEPT: RMSNorm divides each vector by its root mean square and multiplies by a learned gain gamma: y_i = gamma_i*x_i/sqrt(mean_j(x_j^2)+eps). Unlike LayerNorm, it does not center by subtracting the mean and has no learned bias here.
# GOAL: Explain the lightweight normalization used within modern transformer blocks. The following GPT implementation uses PyTorch's nn.RMSNorm for its blocks and final normalization.
# INPUT: A tensor x of shape (..., x), where x is the normalized last dimension; gamma is initialized to ones with shape (x,), and eps is fixed at 1e-5.
# OUTPUT: A tensor with the same shape as x. The module defines no top-level print, plot, or file output.
# HOW INPUT BECOMES OUTPUT: 1) Square each value, keeping shape (..., x). 2) Average along the last axis with keepdim=True, giving (...,1). 3) Add 1e-5 and take the square root to obtain one RMS denominator per vector. 4) Divide each input feature by that scalar and multiply by its corresponding learned gamma feature.
# NOTES: The epsilon prevents division by zero. The parameter is initialized to one and is learned during optimization. This example runs on CPU and requires no dataset or checkpoint.
