"""Executable examples for lesson 01.4; run from any working directory."""

import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.set_num_threads(1)

X = torch.randn(320, 4)
truth = torch.tensor([[2.], [-1.], [0.5], [3.]])
y = X @ truth + 0.7 + 0.1 * torch.randn(320, 1)
split = torch.randperm(len(X))
train_idx, val_idx = split[:240], split[240:]

train_set = TensorDataset(X[train_idx], y[train_idx])
val_set = TensorDataset(X[val_idx], y[val_idx])
shuffle_rng = torch.Generator().manual_seed(11)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                          generator=shuffle_rng, num_workers=0)
train_eval_loader = DataLoader(train_set, batch_size=64)
val_loader = DataLoader(val_set, batch_size=64)

model = nn.Linear(4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss_sum, count = 0.0, 0
    for inputs, targets in loader:
        predictions = model(inputs)
        assert predictions.shape == targets.shape
        loss_sum += criterion(predictions, targets).item() * len(inputs)
        count += len(inputs)
    return loss_sum / count

history = []
best_val, best_state = float("inf"), None
for epoch in range(30):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    train_loss, val_loss = evaluate(train_eval_loader), evaluate(val_loader)
    history.append((train_loss, val_loss))
    if val_loss < best_val:
        best_val = val_loss
        best_state = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "epoch": epoch,
            "torch_rng": torch.get_rng_state(),
            "loader_rng": shuffle_rng.get_state(),
        }

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "checkpoint.pt"
    torch.save(best_state, path)
    saved = torch.load(path, weights_only=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    shuffle_rng.set_state(saved["loader_rng"])
    print("restored epoch:", saved["epoch"], "validation MSE:", evaluate(val_loader))

print("first/final evaluation losses:", history[0], history[-1])
print("learned weights:", model.weight.detach())
assert history[-1][1] < history[0][1]


# A short final evaluation batch must not change the sample-weighted mean.
uneven_loader = DataLoader(val_set, batch_size=37)
with torch.no_grad():
    full_val = criterion(model(X[val_idx]), y[val_idx]).item()
assert abs(evaluate(uneven_loader) - full_val) < 1e-7
assert abs(full_val - best_val) < 1e-7

# Replaying one epoch from the same snapshot checks more than file readability.
def replay_next_epoch(snapshot):
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    torch.set_rng_state(snapshot["torch_rng"])
    shuffle_rng.set_state(snapshot["loader_rng"])
    model.train()
    observed_inputs, losses = [], []
    for inputs, targets in train_loader:
        observed_inputs.append(inputs.clone())
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return (copy.deepcopy(model.state_dict()), torch.cat(observed_inputs),
            torch.stack(losses), torch.get_rng_state(), shuffle_rng.get_state())

run_a = replay_next_epoch(saved)
run_b = replay_next_epoch(saved)
for name in run_a[0]:
    torch.testing.assert_close(run_a[0][name], run_b[0][name], rtol=0, atol=0)
for left, right in zip(run_a[1:], run_b[1:]):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
model.load_state_dict(saved["model"])
optimizer.load_state_dict(saved["optimizer"])
print("uneven evaluation and checkpoint next-epoch replay verified")
