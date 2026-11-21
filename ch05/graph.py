# Concept: compare ReLU, tanh-approximate GELU, and Swish/SiLU activation curves.
# Goal: visualize how common nonlinearities transform scalar inputs.
# The script samples 500 points from -3 through 3 and writes a PNG figure.
# It is a plotting example, not a neural-network training script.

import matplotlib.pyplot as plt
import numpy as np

# NumPy np.linspace creates 500 evenly spaced scalar samples over [-3,3].
# NumPy arrays allow vectorized formulas: each following expression evaluates all 500 points.
x = np.linspace(-3, 3, 500)

# ReLU clips negative values to zero; np.maximum broadcasts elementwise.
relu = np.maximum(0, x)

# GELU uses the common tanh approximation, retaining a smooth transition around zero.
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Swish/SiLU is x*sigmoid(x), attenuating rather than hard-clipping negative inputs.
swish = x * (1 / (1 + np.exp(-x)))  # x * sigmoid(x)

# Matplotlib pyplot.subplots creates a Figure and Axes for drawing.
fig, ax = plt.subplots(figsize=(10, 6))

# Axes.plot receives same-length x and y arrays to draw a 500-point curve.
ax.plot(x, relu, 'b-', linewidth=2, label='ReLU')
ax.plot(x, gelu, 'g-', linewidth=2, label='GELU')
ax.plot(x, swish, 'r-', linewidth=2, label='Swish')

# Set visible ranges and labels; each plotted series has 500 x/y points.
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)

# Grid and reference lines help compare each activation around zero.
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# Matplotlib displays series labels in a legend.
ax.legend(loc='upper left', fontsize=12)

# pyplot.tight_layout reduces overlap between labels and figure edge.
plt.tight_layout()

# pyplot.savefig writes the image; pyplot.close releases plotting state.
# Running writes activation_comparison.png in the current directory; do not run plotting scripts for validation.
plt.savefig('activation_comparison.png', format='png', bbox_inches='tight')
plt.close()

# ===== CONCEPT NOTES =====
# CONCEPT: ReLU(x)=max(0,x), GELU is plotted using the tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), and Swish/SiLU is x*sigmoid(x). These curves illustrate how activation functions pass or reshape positive and negative values.
# GOAL: Provide a visual comparison of three nonlinearities, connecting the SwiGLU lesson to its SiLU gate. The script only makes a figure; it does not instantiate or train a model.
# INPUT: NumPy creates 500 evenly spaced scalar values from -3 to 3 with np.linspace(-3,3,500). No external data or command-line arguments are read.
# OUTPUT: Saves activation_comparison.png in the current working directory (10-by-6 inch figure, PNG format), then closes the figure. The plot has x limits [-3,3] and y limits [-0.5,3].
# HOW INPUT BECOMES OUTPUT: 1) Evaluate ReLU elementwise on the 500 x values. 2) Evaluate the stated tanh approximation for GELU and x/(1+exp(-x)) for Swish. 3) Draw the three 500-point series with labels, axes, grid, and legend. 4) Tighten layout, save the PNG, and close the figure.
# NOTES: Running the file overwrites activation_comparison.png in the current directory. Matplotlib and NumPy are required. The request forbids running plotting scripts during this task; syntax validation alone does not create the image. CPU is sufficient.
