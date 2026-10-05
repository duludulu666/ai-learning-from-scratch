"""Optimizer update rules from scratch (PyTorch) with official optimizer check.

Prescribed gradients isolate the update rule: SGD, Momentum, and Adam.
Manual updates must match torch.optim after every step to 1e-12.
"""

import torch

# Fixed gradient sequence: excludes model/data effects, checks the rule only
gradients = [torch.tensor(g, dtype=torch.float64)
             for g in ([2., -0.5], [-1., 0.25], [0., 3.], [0.5, -2.])]

lr = 0.03
momentum = 0.9

# ---------- 1. Manual SGD: w = w - lr * g ----------
param_sgd = torch.tensor([1., -2.], dtype=torch.float64)
for g in gradients:
    param_sgd = param_sgd - lr * g
print("manual SGD:", param_sgd)

# ---------- 2. Manual Momentum: velocity accumulates history with decay ----------
param_mom = torch.tensor([1., -2.], dtype=torch.float64)
velocity = torch.zeros(2, dtype=torch.float64)
for g in gradients:
    velocity = momentum * velocity + g
    param_mom = param_mom - lr * velocity
print("manual Momentum:", param_mom)

# ---------- 3. Official SGD check ----------
ref_sgd = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
opt_sgd = torch.optim.SGD([ref_sgd], lr=lr, momentum=0)
for g in gradients:
    ref_sgd.grad = g.clone()
    opt_sgd.step()
torch.testing.assert_close(param_sgd, ref_sgd.detach(), atol=1e-12, rtol=1e-12)

# ---------- 4. Official Momentum check ----------
ref_mom = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
opt_mom = torch.optim.SGD([ref_mom], lr=lr, momentum=momentum)
for g in gradients:
    ref_mom.grad = g.clone()
    opt_mom.step()
torch.testing.assert_close(param_mom, ref_mom.detach(), atol=1e-12, rtol=1e-12)

# ---------- 5. Manual Adam: first/second moments with bias correction ----------
param_adam = torch.tensor([1., -2.], dtype=torch.float64)
first = torch.zeros(2, dtype=torch.float64)
second = torch.zeros(2, dtype=torch.float64)
beta1, beta2, eps = 0.9, 0.999, 1e-8
for t, g in enumerate(gradients, 1):
    first = beta1 * first + (1 - beta1) * g
    second = beta2 * second + (1 - beta2) * g.square()
    first_hat = first / (1 - beta1 ** t)
    second_hat = second / (1 - beta2 ** t)
    direction = first_hat / (second_hat.sqrt() + eps)
    param_adam = param_adam - lr * direction
print("manual Adam:", param_adam)

# ---------- 6. Official Adam check ----------
ref_adam = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
opt_adam = torch.optim.Adam([ref_adam], lr=lr, betas=(0.9, 0.999), eps=1e-8)
for g in gradients:
    ref_adam.grad = g.clone()
    opt_adam.step()
torch.testing.assert_close(param_adam, ref_adam.detach(), atol=1e-12, rtol=1e-12)

print("manual updates match torch")
