# Concept: a dot product sums d coordinate-wise products, so its variance tends to grow with width d.
# Dividing attention scores by sqrt(d) keeps their typical scale near 1 before softmax.
# This matters because very large score differences make softmax nearly one-hot and gradients weak.
# PyTorch demonstrates softmax; NumPy samples scores; Matplotlib displays histograms (no file saved).

import torch
import torch.nn.functional as F

# torch.tensor creates a length-3 tensor. These logits differ by 100; softmax assigns
# nearly all probability to 300. The formula is exp(x_i)/sum_j exp(x_j), with PyTorch
# using a numerically stable implementation; dim=0 selects this vector's only axis.
x = torch.tensor([100.0, 200.0, 300.0])
y = F.softmax(x, dim=0)  # torch.nn.functional: shape (3,) -> normalized probabilities (3,).
print(y)


import numpy as np
import matplotlib.pyplot as plt

# NumPy draws two random vectors q,k in R^10, each shape (10,) with independent
# standard-normal components. Their dot product sums ten products; its variance grows with d.
d = 10

q = np.random.randn(d)  # NumPy: draw d standard-normal values, shape (10,).
k = np.random.randn(d)  # A separate random key vector, also shape (10,).

dot_product = np.dot(q, k)  # NumPy dot: two shape-(10,) vectors -> scalar similarity.
scaled_dot_product = dot_product / np.sqrt(d)  # Divide by sqrt(10), not by 10.

print('dot product:', dot_product)
print('scaled dot product:', scaled_dot_product)


# Repeat the random experiment many times to compare distributions rather than one draw.
d = 10
num_samples = 10000  # Number of independent query/key pairs to sample.

dot_products = []
scaled_dot_products = []

for _ in range(num_samples):
    q = np.random.randn(d)  # New standard-normal query, shape (10,).
    k = np.random.randn(d)  # New standard-normal key, shape (10,).

    dot_product = np.dot(q, k)  # Scalar sum of ten pairwise products.
    scaled_dot_product = dot_product / np.sqrt(d)  # Scaled scalar; expected variance near 1.

    dot_products.append(dot_product)
    scaled_dot_products.append(scaled_dot_product)


# Matplotlib overlays the two score distributions so their spreads can be compared visually.
plt.figure(figsize=(10, 6))  # Matplotlib pyplot: create a 10-by-6-inch plotting canvas.
plt.hist(dot_products, bins=50, alpha=0.5, label='Without scaling')  # Plot 10,000 raw scalar scores.
plt.hist(scaled_dot_products, bins=50, alpha=0.5, label='With scaling')  # Overlay the scaled-score distribution.
plt.legend()  # Display labels for the two histogram series.
plt.show()  # Open/render the figure; this script does not save it.

print("Variances without scaling:", np.var(dot_products))  # NumPy np.var reduces the list to one scalar.
print("Variances with scaling:", np.var(scaled_dot_products))  # Expected near 1 for scaled standard-normal dot products.

# ===== CONCEPT NOTES =====
# CONCEPT: If q_i and k_i are independent, zero-mean, unit-variance values, the dot product
# sum_i(q_i*k_i) has variance d. Dividing by sqrt(d) makes its variance approximately 1.
# This moderates the logits so softmax is less likely to collapse onto one key.
# GOAL: Motivate scaled dot-product attention after the raw QK scores in 02_attn_math.py.
# INPUT: PyTorch softmax receives [100,200,300]. NumPy samples 10,000 pairs of shape-(10,)
# standard-normal vectors; randomness is unseeded, so sampled values vary between runs.
# OUTPUT: Prints one three-value probability tensor, one raw/scaled score pair, then two
# estimated variances; Matplotlib opens an overlapping histogram with 50 bins per series.
# No image file is saved by this script.
# HOW INPUT BECOMES OUTPUT:
# 1. Softmax normalizes [100,200,300]; the probability for 300 is essentially 1 because
#    its logit is much larger than the other two.
# 2. For each sample, np.dot sums ten coordinate products; divide that scalar by sqrt(10).
# 3. Store 10,000 raw and scaled scalars, plot both distributions, and compute np.var.
# NOTES: F.softmax is from torch.nn.functional; np.random.randn, np.dot, np.sqrt, and np.var
# come from NumPy; plt.figure/hist/legend/show come from Matplotlib. Scaling is by sqrt(d),
# not by d. The random sample means estimated variance will be near, not exactly, 10 and 1.
