"""Learning-rate stability demo for lesson 04.3 (PyTorch).

Part 1: exact scalar quadratic L = 1/2*(theta-3)^2 under four learning
        rates: stable convergence, one-step landing, oscillation, divergence.
Part 2: linear model trained with three learning rates from the same init:
        the first gradient must be identical across runs (learning rate only
        controls how far params move, not where the gradient points);
        validation loss exposes too-slow vs well-tuned vs exploding rates.
Part 3: exact loss ratio after one GD step: L(theta')/L(theta) = (1-eta*a)^2.
Part 4: warmup + cosine schedule endpoints and geometric range test.
"""

import copy
import math
import torch
from torch import nn


# ---------- 1. Exact scalar quadratic: four learning-rate fates ----------
for rate in [0.1, 1.0, 2.0, 2.2]:
    theta = 0.0
    losses = []
    for _ in range(8):
        losses.append(0.5 * (theta - 3.0) ** 2)   # record current loss
        theta -= rate * (theta - 3.0)             # one GD step
    print("rate:", rate, "losses:", losses)

# ---------- 2. Linear model: same init, three rates, first gradient equal ----------
torch.manual_seed(6)
X = torch.randn(200, 4)
y = X @ torch.tensor([[2.], [-1.], [0.5], [1.]]) + 0.1 * torch.randn(200, 1)
train_x, val_x = X[:150], X[150:]
train_y, val_y = y[:150], y[150:]
initial = nn.Linear(4, 1).state_dict()   # snapshot: every run starts identical

first_grads = []
for rate in [1e-4, 0.05, 2.0]:
    model = nn.Linear(4, 1)
    model.load_state_dict(copy.deepcopy(initial))
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    first_grad = None
    last_val = float("inf")

    for step in range(100):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(train_x), train_y)
        if not math.isfinite(loss.item()):
            break
        loss.backward()
        grad_norm = math.sqrt(sum(p.grad.square().sum().item() for p in model.parameters()))
        if first_grad is None:
            first_grad = grad_norm
        optimizer.step()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(val_x), val_y)
            last_val = val_loss.item()
        if last_val > 1e10 or not math.isfinite(last_val):   # divergence guard
            break
    print("rate:", rate, "first gradient:", first_grad, "validation:", last_val)
    first_grads.append(first_grad)

# learning rate does not affect the first gradient (same init, same data)
assert max(first_grads) - min(first_grads) < 1e-10
print("OK")

# ---------- 3. Exact loss ratio: L(theta')/L(theta) = (1 - eta*a)^2 ----------
for curvature in (1., 100.):
    for rate in (.1 / curvature, 1 / curvature, 2 / curvature, 2.2 / curvature):
        error = -3.0
        old_loss = curvature * error ** 2 / 2
        new_error = error - rate * curvature * error   # error updates by eta*a*error
        new_loss = curvature * new_error ** 2 / 2
        actual_ratio = new_loss / old_loss
        assert math.isclose(actual_ratio, (1 - rate * curvature) ** 2, abs_tol=1e-12)
print("exact loss ratios verified")

# ---------- 4. Warmup + cosine schedule and geometric range test ----------
def warmup_cosine(t, updates=6, warmup=2, peak=.1, floor=.01):
    if not 0 <= t < updates or not 1 <= warmup < updates - 1:
        raise ValueError("invalid schedule indices")
    if t < warmup:                       # linear warmup: peak*(t+1)/warmup
        return peak * (t + 1) / warmup
    progress = (t - warmup) / (updates - warmup - 1)   # cosine decay to floor
    return floor + .5 * (peak - floor) * (1 + math.cos(math.pi * progress))

rates = [warmup_cosine(t) for t in range(6)]
torch.testing.assert_close(torch.tensor(rates), torch.tensor([.05, .1, .1, .0775, .0325, .01]))

geometric = [1e-5 * (1e-1 / 1e-5) ** (t / 4) for t in range(5)]   # range test: 1e-5 -> 0.1
assert math.isclose(geometric[0], 1e-5) and math.isclose(geometric[-1], .1)
assert all(math.isclose(geometric[i + 1] / geometric[i], 10.) for i in range(4))
print("exact loss ratios and schedule endpoints verified:", rates)
