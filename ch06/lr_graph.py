# Concept: visualize cosine decay and linear D2Z learning-rate schedules with warmup.
# Goal: compare a nonzero-minimum cosine curve with linear decay to zero.
# Both curves are normalized to a maximum learning rate of 1 over progress [0,1].
# The plotting script saves one PNG and does not train a model.

import matplotlib.pyplot as plt
import numpy as np

# Optional font preference for Japanese chart annotations; Linux may not have this macOS font.
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS
# plt.rcParams['font.family'] = 'Yu Gothic'  # Windows

# Schedule parameters.
warmup_ratio = 0.05  # Use the first 5% of normalized training progress to warm up.
eta_min_ratio = 0.1  # Cosine schedule ends at 10% of its peak rate.

# NumPy creates 1000 evenly spaced progress values t in [0,1].
t = np.linspace(0, 1, 1000)

# Cosine annealing schedule with an initial warmup.
# After warmup, cosine LR = eta_min+(1-eta_min)*(1+cos(pi*p))/2.
# It starts decay at 1 and ends at eta_min_ratio (0.1 here), unlike linear decay to zero.
def cosine_annealing(t, warmup_ratio, eta_min_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # Linear warmup raises normalized LR from 0 to 1.
            lr[i] = ti / warmup_ratio
        else:
            # Cosine decay from 1 down to eta_min.
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + np.cos(np.pi * progress))
    return lr

# D2Z here means warmup followed by linear decay-to-zero.
# Both curves are normalized to peak 1; input and output are NumPy arrays of matching shape.
def d2z(t, warmup_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # Linear warmup from 0 to 1.
            lr[i] = ti / warmup_ratio
        else:
            # Linear decay takes normalized LR from 1 down to zero.
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = 1 - progress
    return lr

# NumPy stores the sampled curves; Matplotlib plots them without training a model.
cosine_lr = cosine_annealing(t, warmup_ratio, eta_min_ratio)
d2z_lr = d2z(t, warmup_ratio)

# Create the Matplotlib figure and axes.
fig, ax = plt.subplots(figsize=(10, 6))

# Shade warmup; adjacent Japanese annotation literally means “warmup”.
ax.axvspan(0, warmup_ratio, color='lightgray', alpha=0.5)
# Japanese annotation means “warmup”; retained as functional chart text.
ax.text(0.01, 1.02, 'ウォームアップ', fontsize=10, va='bottom')

# Plot 1000-point arrays: cosine ends at eta_min_ratio, D2Z ends at zero.
# Japanese label means “cosine annealing”; retained as functional chart text.
ax.plot(t, cosine_lr, 'b-', linewidth=2, label='コサインアニーリング')
ax.plot(t, d2z_lr, color='orange', linestyle='--', linewidth=2, label='D2Z')

# Axes show normalized training progress and learning rate, not absolute units.
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
# Japanese axis label means “training progress”; retained as functional chart text.
ax.set_xlabel('学習の進行度', fontsize=12)
# Japanese axis label means “learning rate”; retained as functional plot text.
ax.set_ylabel('学習率', fontsize=12)

# Grid lines help compare warmup and decay slopes.
ax.grid(True, linestyle='--', alpha=0.7)

# Legend names schedules; Japanese labels are plot data.
ax.legend(loc='upper right', fontsize=12)

# pyplot.tight_layout reduces overlap between labels and figure edge.
plt.tight_layout()

# pyplot.savefig writes the image and pyplot.close releases figure state.
# Running writes lr_schedule.png in the current directory; do not run plotting scripts for validation.
plt.savefig('lr_schedule.png', format='png', bbox_inches='tight')
plt.close()

# ===== CONCEPT NOTES =====
# CONCEPT: The plot compares warmup followed by cosine annealing to a minimum ratio and D2Z-style linear decay to zero. Both schedules start at 0, rise linearly to 1 over the warmup interval, then decay; cosine ends at eta_min_ratio while D2Z reaches 0.
# GOAL: Visually compare learning-rate schedules used to control optimization over training progress. This plot complements the executable linear warmup/decay function in ch06/03_lr_scheduler.py.
# INPUT: t=np.linspace(0,1,1000); warmup_ratio=0.05; eta_min_ratio=0.1. Curves are normalized (peak=1), so no absolute learning rate or model/data input is involved.
# OUTPUT: Saves lr_schedule.png in the current working directory as a 10-by-6 inch PNG. The chart plots cosine and D2Z curves, shades the warmup span [0,0.05], and uses axes spanning x=[0,1], y=[0,1.05].
# HOW INPUT BECOMES OUTPUT: 1) Create 1000 progress values. 2) For t<0.05, both curves use t/0.05. 3) After warmup, cosine uses progress=(t-0.05)/(1-0.05) and eta_min+0.5*(1-eta_min)*(1+cos(pi*progress)); D2Z uses 1-progress. 4) Plot both arrays with labels, grid, and warmup annotation. 5) Tighten layout, write the PNG, and close the figure.
# NOTES: Running overwrites lr_schedule.png in the current working directory. The script sets the font family to Hiragino Sans (a macOS font); on Linux it may fall back or emit a font warning, but this is not a model or training dependency. NumPy and Matplotlib are required. CPU is sufficient; do not run the plotting script for this task.
