# Concept: parameter updates with SGD and AdamW, including decoupled weight decay.
# Goal: compare the optimizer formulas in code and show a tiny linear-regression loop.
# The runnable loop deliberately selects PyTorch's AdamW, not the custom class above.
# Loss is printed before each update for a single fixed example.

import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                p.data = p.data - lr * p.grad.data


# AdamW combines momentum (first moment m) and squared-gradient scale (second moment v), with bias correction.
# Decoupled weight decay separately shrinks parameters by lr*weight_decay, not by adding L2 to the gradient.
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                # grad, m and v have the same shape as parameter p; state tracks each parameter separately.
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['t'] += 1
                t = state['t']

                # Update momentum m (average gradient) and v (average squared gradient), elementwise per tensor.
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                state['m'], state['v'] = m, v

                # Correct initially zero moments, biased toward zero during early steps.
                # Bias-corrected m_hat/v_hat set an adaptive update scale ~lr*m_hat/(sqrt(v_hat)+eps).
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
                # Apply the adaptive gradient step, then decoupled weight decay toward zero.
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * p.data


torch.manual_seed(0)

# torch.nn.Linear maps two features to one prediction: (1,2)->(1,1).
model = torch.nn.Linear(2, 1)
# optimizer = AdamW(model.parameters(), lr=0.1)
# Active demo uses maintained torch.optim.AdamW, not the educational custom class above.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
# One illustrative batch: x shape (1,2), target y shape (1,1).
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[3.0]])

# MSE is scalar; backward fills gradients, step updates weights, zero_grad clears them.
for step in range(5):
    # Forward predicts one scalar; mean squared difference from y is scalar loss.
    output = model(x)
    loss = (output - y).pow(2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Step {step}: loss = {loss.item():.4f}")

# ===== CONCEPT NOTES =====
# CONCEPT: SGD updates parameters by subtracting lr*gradient. The custom AdamW class tracks first and second gradient moments, applies bias correction, divides by sqrt(v_hat)+eps, then applies decoupled decay as a separate -lr*weight_decay*parameter term.
# GOAL: Illustrate optimizer mechanics and a short training loop. The demo uses torch.optim.AdamW; the custom SGD and AdamW implementations are definitions for comparison and are not selected by the active code.
# INPUT: torch.manual_seed(0); a torch.nn.Linear(2,1), one x=[[1.0,2.0]] batch of shape (1,2), target y=[[3.0]] of shape (1,1), PyTorch AdamW with lr=0.1, and 5 update iterations. The custom AdamW defaults are lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01.
# OUTPUT: Prints five lines, `Step 0` through `Step 4`, each with the scalar MSE loss computed before that iteration's optimizer update. No parameters are saved.
# HOW INPUT BECOMES OUTPUT: 1) The linear layer maps the single two-feature input to one prediction. 2) Compute mean((prediction-target)^2), a scalar. 3) Backpropagate that loss, step AdamW at lr=0.1, and clear gradients. 4) Repeat five times; each displayed loss reflects the model state before that step's update.
# NOTES: Exact loss values depend on the PyTorch optimizer implementation/version, though seeding makes initialization repeatable in a fixed environment. This example is small and CPU-compatible. The custom optimizer uses p.data assignments and is educational; the active demonstration uses the maintained PyTorch optimizer.
