"""Executable examples for lesson 04.3; run from any working directory."""

import copy
import math
import torch
from torch import nn

torch.manual_seed(6)
torch.set_num_threads(1)
for rate in [0.1, 1.0, 2.0, 2.2]:
    theta = 0.0
    losses = []
    for _ in range(8):
        losses.append(0.5 * (theta - 3.0)**2)
        theta -= rate * (theta - 3.0)
    print("quadratic rate/losses:", rate, losses)

X = torch.randn(200, 4)
y = X @ torch.tensor([[2.], [-1.], [0.5], [1.]]) + 0.1 * torch.randn(200, 1)
train_x, val_x = X[:150], X[150:]
train_y, val_y = y[:150], y[150:]
initial = nn.Linear(4, 1).state_dict()

initial_gradients = []
for rate in [1e-4, 0.05, 2.0]:
    model = nn.Linear(4, 1)
    model.load_state_dict(copy.deepcopy(initial))
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    first_grad, last_val = None, float("inf")

    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(train_x) - train_y).square().mean()
        if not torch.isfinite(loss):
            print("non-finite loss at step", step)
            break
        loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
        if first_grad is None:
            first_grad = grad_norm.item()
        if not torch.isfinite(grad_norm):
            break
        optimizer.step()
        with torch.no_grad():
            last_val = (model(val_x) - val_y).square().mean().item()
        if not math.isfinite(last_val) or last_val > 1e10:
            break

    initial_gradients.append(first_grad)
    print("linear run:", rate, "first gradient:", first_grad, "validation:", last_val)

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(__file__).resolve().parent / "figures"
output.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
for rate in [0.1, 1.0, 2.0, 2.2]:
    theta, losses = 0.0, []
    for _ in range(16):
        losses.append(max(0.5 * (theta - 3.0)**2, 1e-12))
        theta -= rate * (theta - 3.0)
    ax.semilogy(range(16), losses, marker=".", label=f"learning rate = {rate}")
ax.set(xlabel="Update", ylabel="Loss (log scale; zero floored for display)",
       title="Exact scalar quadratic: stable, oscillating, and divergent")
ax.legend()
fig.tight_layout()
fig.savefig(output / "04_3_learning_rates.png", dpi=150)
plt.close(fig)


assert max(initial_gradients) - min(initial_gradients) < 1e-10
for curvature in (1., 100.):
    for rate in (.1/curvature, 1/curvature, 2/curvature, 2.2/curvature):
        error = -3.
        old_loss = curvature*error**2/2
        new_error = error-rate*curvature*error
        actual_ratio = (curvature*new_error**2/2)/old_loss
        assert math.isclose(actual_ratio, (1-rate*curvature)**2, abs_tol=1e-12)

def warmup_cosine(t, updates=6, warmup=2, peak=.1, floor=.01):
    if not 0 <= t < updates or not 1 <= warmup < updates-1:
        raise ValueError("invalid schedule indices")
    if t < warmup:
        return peak*(t+1)/warmup
    progress = (t-warmup)/(updates-warmup-1)
    return floor + .5*(peak-floor)*(1+math.cos(math.pi*progress))

rates = [warmup_cosine(t) for t in range(6)]
torch.testing.assert_close(torch.tensor(rates), torch.tensor([.05,.1,.1,.0775,.0325,.01]))
geometric = [1e-5*(1e-1/1e-5)**(t/4) for t in range(5)]
assert math.isclose(geometric[0], 1e-5) and math.isclose(geometric[-1], .1)
assert all(math.isclose(geometric[i+1]/geometric[i], 10.) for i in range(4))
print("exact loss ratios and schedule endpoints verified:", rates)
