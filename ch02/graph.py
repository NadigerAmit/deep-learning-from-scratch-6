# Concept: ReLU clamps every negative input to zero; GELU smoothly gates inputs using their magnitude.
# Comparing the curves shows why GELU can preserve some negative information instead of a hard cutoff.
# NumPy calculates sample values and Matplotlib draws and saves a PNG; this is a plotting-only example.

import matplotlib.pyplot as plt
import numpy as np

# NumPy linspace makes 500 evenly spaced x values from -3 to 3, inclusive; shape is (500,).
x = np.linspace(-3, 3, 500)

# ReLU(x)=max(0,x): NumPy maximum applies elementwise, yielding zero for x<0 and x for x>=0.
relu = np.maximum(0, x)

# Tanh GELU approximation, applied elementwise to the same 500 input values.
# The smooth transition gives small negative inputs a small negative output.
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Matplotlib creates a figure and axes; plot both (500,) outputs against the (500,) x array.
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, relu, 'b-', linewidth=2, label='ReLU')  # Matplotlib Axes.plot: x and ReLU arrays each have shape (500,).
ax.plot(x, gelu, 'r-', linewidth=2, label='GELU')  # Draw the second shape-(500,) curve on the same axes.

# Set visible coordinate range and human-readable axis labels.
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize=12)  # Set horizontal axis label and font size.
ax.set_ylabel('f(x)', fontsize=12)  # Set vertical axis label and font size.

# Add a dashed grid and horizontal/vertical zero reference lines to compare curves.
ax.grid(True, linestyle='--', alpha=0.7)  # Draw a semi-transparent dashed grid.
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# Identify the blue and red curves by their labels.
ax.legend(loc='upper left', fontsize=12)

# Matplotlib tight_layout reduces clipping and unnecessary whitespace around the chart.
plt.tight_layout()

# Save a PNG image in the current working directory, then close the figure to release resources.
plt.savefig('relu_gelu.png', format='png', bbox_inches='tight')  # pyplot saves the current figure, tightly cropped.
plt.close()

# ===== CONCEPT NOTES =====
# CONCEPT: ReLU(x)=max(0,x) sharply discards all negative values. GELU approximates x*Phi(x),
# where Phi is the standard-normal cumulative distribution; its smooth gate retains some negatives.
# GOAL: Visualize the GELU activation used by the FFN in 09_norm_gelu.py and contrast it with ReLU.
# INPUT: NumPy linspace samples 500 x values in [-3,3]; ReLU and tanh-approximated GELU each
# produce shape-(500,) arrays. No user or external data is read.
# OUTPUT: Saves relu_gelu.png in the current working directory: 10x6 inches, blue ReLU/red GELU,
# x limits [-3,3], y limits [-0.5,3], grid, zero axes, and legend. It prints nothing.
# HOW INPUT BECOMES OUTPUT:
# 1. np.linspace creates sample x values; np.maximum computes ReLU element by element.
# 2. NumPy arithmetic, tanh, and sqrt evaluate 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
# 3. Matplotlib plots each output against x, labels axes, and adds grid, reference axes, and legend.
# 4. plt.tight_layout adjusts spacing; plt.savefig writes the PNG; plt.close releases figure memory.
# NOTES: NumPy handles arrays and math; Matplotlib pyplot handles plotting/file output. The path
# is relative to the process working directory, and an existing relu_gelu.png is overwritten.
# This plotting script is intentionally not run during validation because it writes a PNG.
