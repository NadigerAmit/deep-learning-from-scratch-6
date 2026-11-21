# Concept: a warmup followed by linear learning-rate decay to zero.
# Goal: provide a small step-indexed schedule function for an optimizer.
# The function expects integer iteration boundaries and a maximum learning rate.
# No model, data, training loop, or plot is run in this file.

# Training loops assign this Python float to optimizer.param_groups[*]['lr'].
# This is linear warmup + linear decay, not cosine; see ch06/lr_graph.py for cosine comparison.
def get_lr(it, max_lr, warmup_iters, max_iters):
    # Linear warmup rises from zero to max_lr, avoiding a large first update.
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # Linear decay reduces the rate from max_lr to zero over remaining iterations.
    if it < max_iters:
        # progress moves from 0 to 1; halfway through decay, LR is half max_lr.
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0

# ===== CONCEPT NOTES =====
# CONCEPT: get_lr implements linear warmup from 0 to max_lr, then linear annealing from max_lr to 0, and returns 0 after max_iters. It is a simple two-stage schedule, not cosine annealing.
# GOAL: Make the optimizer's learning rate increase gradually before decreasing across training. The chapter's pretraining script contains a copy of this schedule and calls it at each iteration.
# INPUT: get_lr(it, max_lr, warmup_iters, max_iters). For example, with max_lr=0.001, warmup_iters=200, max_iters=40000, it=100 returns 0.0005. The schedule assumes sensible positive boundaries and max_iters greater than warmup_iters for the annealing branch.
# OUTPUT: One Python float learning rate; this module has no top-level print, plot, or file output.
# HOW INPUT BECOMES OUTPUT: 1) If it < warmup_iters, calculate max_lr*(it/warmup_iters). 2) Otherwise, if it < max_iters, set progress=(it-warmup_iters)/(max_iters-warmup_iters) and return max_lr*(1-progress). 3) At it>=max_iters, return 0.0.
# NOTES: At it=warmup_iters the function enters the second branch and returns max_lr; at it=max_iters it returns zero. This file only defines a function and can be imported/run on CPU without datasets or checkpoints.
