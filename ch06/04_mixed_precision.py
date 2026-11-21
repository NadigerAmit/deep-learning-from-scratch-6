# Concept: FP16 and BF16 trade numeric range and precision differently.
# Goal: demonstrate rounding, underflow, overflow, and autocast tensor dtypes.
# Scalar examples are followed by a 1000-by-1000 matrix multiplication.
# The device is selected automatically as CUDA when available, otherwise CPU.

import torch

# FP16 has a narrow exponent range; tiny gradients can vanish and large values overflow.
# BF16 has fewer fraction bits but an FP32-like exponent range, often useful for training.
print("----- FP16 -----")

large = torch.tensor(1000.0, dtype=torch.float16)
small = torch.tensor(0.01, dtype=torch.float16)
print(large + small)  # tensor(1000., dtype=torch.float16)

tiny = torch.tensor(1e-8, dtype=torch.float16)
print(tiny)  # tensor(0., dtype=torch.float16)

huge = torch.tensor(70000.0, dtype=torch.float16)
print(huge)  # tensor(inf, dtype=torch.float16)

print("----- BF16 -----")

# FP16 underflows this tiny value because its exponent range is limited.
tiny_fp16 = torch.tensor(1e-8, dtype=torch.float16)
print(tiny_fp16)  # tensor(0., dtype=torch.float16)

# BF16 has a wider exponent range, so this value remains representable (with rounding).
tiny_bf16 = torch.tensor(1e-8, dtype=torch.bfloat16)
print(tiny_bf16)  # tensor(1.0012e-08, dtype=torch.bfloat16)

# FP16 overflows 70000 beyond its finite range, producing infinity.
huge_fp16 = torch.tensor(70000.0, dtype=torch.float16)
print(huge_fp16)  # tensor(inf, dtype=torch.float16)

# BF16 represents this value (with rounding).
huge_bf16 = torch.tensor(70000.0, dtype=torch.bfloat16)
print(huge_bf16)  # tensor(70144., dtype=torch.bfloat16)


# The original Japanese section label means “automatic mixed precision.”
print("----- 自動混合精度 -----")

# CUDA is chosen when available; otherwise this file uses CPU (no MPS branch). CPU BF16 autocast support
# depends on hardware and the PyTorch build; unsupported configurations can raise an error.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(1000, 1000, device=device)

# torch.autocast selects lower precision for supported ops and keeps sensitive ops in safer types.
# BF16 generally needs no GradScaler; FP16 training often uses torch.amp.GradScaler to prevent underflow.
# This example has no scaler or optimizer step: it only demonstrates result dtypes.
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    b = a @ a   # Autocast chooses BF16 for this matrix multiplication.
    c = a.sum() # Reduction accumulates/returns FP32 for numerical stability.
    print(b.dtype)  # torch.bfloat16
    print(c.dtype)  # torch.float32

# ===== CONCEPT NOTES =====
# CONCEPT: Mixed precision performs selected operations in lower-precision formats. FP16 has a narrower exponent range than BF16, so FP16 can underflow small magnitudes and overflow large ones; autocast chooses operation dtypes (the matrix multiply here is BF16 while the reduction accumulates in FP32).
# GOAL: Make numeric range and autocast behavior concrete before applying mixed precision in model training. The pretraining example later wraps forward/loss calculations in BF16 autocast.
# INPUT: Scalar tensors 1000.0, 0.01, 1e-8, and 70000.0 are constructed in FP16/BF16 as labelled; the autocast section creates random a of shape (1000,1000) on CUDA if available, otherwise CPU, and uses dtype=torch.bfloat16.
# OUTPUT: Prints section labels and scalar representations: FP16 1000+0.01 rounds to 1000, FP16 1e-8 underflows to 0, FP16 70000 overflows to inf; BF16 represents 1e-8 approximately as 1.0012e-08 and 70000 approximately as 70144. Then it prints b.dtype (bfloat16) and c.dtype (float32). No files are saved.
# HOW INPUT BECOMES OUTPUT: 1) Convert each scalar to the stated 16-bit format, exposing rounding, underflow, or overflow. 2) Enter autocast for BF16 on the selected device. 3) Compute b=a@a with shape (1000,1000), which autocast executes in BF16. 4) Compute c=a.sum(), whose accumulation/result remains FP32, and print both dtypes.
# NOTES: Printed low-precision representations can vary slightly across PyTorch versions/devices. A CPU-only machine takes the CPU branch; BF16 autocast support depends on the installed PyTorch build and CPU. CUDA is not required. This is an active numeric demo, but do not run it for this comment-only task.
