"""Executable examples for lesson 04.2; run from any working directory."""

import torch

torch.manual_seed(4)
torch.set_num_threads(1)

# L2-in-the-loss equals decay for one plain-SGD step.
w1 = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
w2 = torch.nn.Parameter(w1.detach().clone())
g = torch.tensor([0.4, -0.2], dtype=torch.float64)
lr, decay = 0.1, 0.05
opt1 = torch.optim.SGD([w1], lr=lr, weight_decay=0)
opt2 = torch.optim.SGD([w2], lr=lr, weight_decay=decay)
loss1 = (w1 * g).sum() + 0.5 * decay * w1.square().sum()
loss1.backward()
(w2 * g).sum().backward()
opt1.step()
opt2.step()
torch.testing.assert_close(w1, w2)

# AdamW: verify the first step with known data gradients.
w3 = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
old = w3.detach().clone()
adamw = torch.optim.AdamW([w3], lr=lr, weight_decay=decay, eps=1e-8)
(w3 * g).sum().backward()
adamw.step()
expected = old * (1 - lr * decay) - lr * g / (g.abs() + 1e-8)
torch.testing.assert_close(w3, expected)

# Dropout preserves activation expectation, approximately over many samples.
dropout = torch.nn.Dropout(p=0.5)
values = torch.full((20000,), 2.0)
dropout.train()
print("training mean:", dropout(values).mean().item())
dropout.eval()
torch.testing.assert_close(dropout(values), values)
print("plain SGD result:", w1.detach())
print("AdamW first step:", w3.detach())

import numpy as np

rng = np.random.default_rng(8)
X = rng.normal(size=(180, 40))
truth = np.zeros(40)
truth[:3] = [2., -1., 0.5]
y = X @ truth + rng.normal(scale=1.0, size=180)
A, V = X[:25], X[25:]
a, v = y[:25], y[25:]

# Objective: mean squared error / 2 + lambda * ||w||^2 / 2.
# Augmenting the system gives the same ridge objective.
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    augmented_X = np.vstack([A, np.sqrt(len(A) * lam) * np.eye(40)])
    augmented_y = np.r_[a, np.zeros(40)]
    weights, *_ = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)
    train_mse = np.mean((A @ weights - a)**2)
    val_mse = np.mean((V @ weights - v)**2)
    print(lam, train_mse, val_mse, np.linalg.norm(weights))


# Explicit L2 and coupled Adam decay agree; decoupled AdamW does not.
coupled = torch.nn.Parameter(torch.tensor([2.,-3.], dtype=torch.float64))
explicit = torch.nn.Parameter(coupled.detach().clone())
decoupled = torch.nn.Parameter(coupled.detach().clone())
opt_c = torch.optim.Adam([coupled], lr=.1, weight_decay=.05)
opt_e = torch.optim.Adam([explicit], lr=.1, weight_decay=0)
opt_d = torch.optim.AdamW([decoupled], lr=.1, weight_decay=.05)
for prescribed in ([.4,-.2], [-.3,.6], [.1,.2]):
    data_gradient = torch.tensor(prescribed, dtype=torch.float64)
    for opt in (opt_c,opt_e,opt_d):
        opt.zero_grad(set_to_none=True)
    (coupled*data_gradient).sum().backward()
    ((explicit*data_gradient).sum()+.025*explicit.square().sum()).backward()
    (decoupled*data_gradient).sum().backward()
    for opt in (opt_c,opt_e,opt_d):
        opt.step()
    torch.testing.assert_close(coupled, explicit, atol=1e-12, rtol=1e-10)
assert not torch.allclose(coupled, decoupled)

torch.manual_seed(402)
samples = torch.nn.functional.dropout(torch.full((100000,),2.), p=.5, training=True)
assert abs(samples.mean().item()-2) < .03
assert abs(samples.var(unbiased=False).item()-4) < .03
lam = .1
aug_X = np.vstack([A, np.sqrt(len(A)*lam)*np.eye(A.shape[1])])
aug_y = np.r_[a, np.zeros(A.shape[1])]
ridge, *_ = np.linalg.lstsq(aug_X, aug_y, rcond=None)
np.testing.assert_allclose(A.T @ (A @ ridge-a)/len(A)+lam*ridge, 0, atol=1e-10)
print("coupled/decoupled Adam, dropout variance, and ridge stationarity verified")
