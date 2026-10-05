"""04.2 Regularization and Generalization

Verifies three regularization mechanisms numerically:
1. L2-in-the-loss equals SGD weight decay (one step).
2. AdamW decoupled decay matches the hand-derived first step.
3. Dropout preserves activation expectation in train mode, identity in eval.
Plus a ridge ablation on an underdetermined problem (25 samples, 40 features).
"""

import numpy as np
import torch

torch.manual_seed(4)
torch.set_num_threads(1)

# ---------- 1. L2 in the loss == SGD weight decay ----------
w1 = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
w2 = torch.nn.Parameter(w1.detach().clone())
g = torch.tensor([0.4, -0.2], dtype=torch.float64)
lr, decay = 0.1, 0.05

# Route A: L2 penalty written into the loss, no weight_decay
opt1 = torch.optim.SGD([w1], lr=lr, weight_decay=0)
loss1 = (w1 * g).sum() + 0.5 * decay * w1.square().sum()
loss1.backward()
opt1.step()

# Route B: pure data loss, weight_decay performs the shrinkage
opt2 = torch.optim.SGD([w2], lr=lr, weight_decay=decay)
(w2 * g).sum().backward()
opt2.step()

torch.testing.assert_close(w1, w2)
print("L2 == decay | w1:", w1.detach(), "| w2:", w2.detach())

# ---------- 2. AdamW first step: decoupled decay + Adam direction ----------
w3 = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
old = w3.detach().clone()
adamw = torch.optim.AdamW([w3], lr=lr, weight_decay=decay, eps=1e-8)
(w3 * g).sum().backward()
adamw.step()

# First step has no momentum history: direction = g / (|g| + eps)
expected = (1 - lr * decay) * old - lr * g / (g.abs() + 1e-8)
torch.testing.assert_close(w3, expected)
print("AdamW first step:", w3.detach())

# ---------- 3. Dropout: preserve expectation, identity at eval ----------
dropout = torch.nn.Dropout(p=0.5)
values = torch.full((20000,), 2.0)

dropout.train()
out = dropout(values)
print(f"train mean: {out.mean().item():.4f} (expect 2), "
      f"var: {out.var(unbiased=False).item():.4f} (expect 4)")

dropout.eval()
torch.testing.assert_close(dropout(values), values)
print("eval identity OK")

# ---------- 4. Ridge regression on an underdetermined problem ----------
rng = np.random.default_rng(8)
X = rng.normal(size=(180, 40))
truth = np.zeros(40)
truth[:3] = [2., -1., 0.5]
y = X @ truth + rng.normal(scale=1.0, size=180)
A, V = X[:25], X[25:]            # 25 training samples, 155 validation
a, v = y[:25], y[25:]

print("\nlam    train_mse  val_mse   ||w||")
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    # Augment the system: rows sqrt(25*lam)*I pull weights toward zero.
    A_aug = np.vstack([A, np.sqrt(25 * lam) * np.eye(40)])
    a_aug = np.concatenate([a, np.zeros(40)])
    w, *_ = np.linalg.lstsq(A_aug, a_aug, rcond=None)
    train_mse = np.mean((A @ w - a) ** 2)
    val_mse = np.mean((V @ w - v) ** 2)
    print(f"{lam:5.2f}  {train_mse:.4f}  {val_mse:.4f}  {np.linalg.norm(w):.4f}")
