"""Executable examples for lesson 03.1; run from any working directory."""

import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def mlp_forward(x, W1, b1, W2, b2):
    h = W1 @ x + b1
    a = relu(h)
    y = W2 @ a + b2
    return y, (x, h, a, W1, W2)

def mlp_backward(y, target, cache):
    x, h, a, W1, W2 = cache
    dy = y - target
    dW2 = np.outer(dy, a)
    db2 = dy.copy()
    da = W2.T @ dy
    dh = da * (h > 0)
    dW1 = np.outer(dh, x)
    db1 = dh.copy()
    return dW1, db1, dW2, db2

import torch
from torch import nn

class TinyMLP(nn.Module):
    def __init__(self, in_features=4, hidden_features=5, out_features=2):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.layer1(x)
        a = torch.relu(h)
        return self.layer2(a)

torch.manual_seed(301)
model = TinyMLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(8, 4)
target = torch.randn(8, 2)
optimizer.zero_grad()
prediction = model(x)
loss = 0.5 * (prediction - target).square().sum(dim=-1).mean()
loss.backward()
optimizer.step()

import numpy as np
import torch

rng = np.random.default_rng(3)
x_check = rng.normal(size=4)
W1_check = rng.normal(size=(5, 4))
b1_check = rng.normal(size=5)
W2_check = rng.normal(size=(2, 5))
b2_check = rng.normal(size=2)
target_check = rng.normal(size=2)
y_check, cache_check = mlp_forward(x_check, W1_check, b1_check, W2_check, b2_check)
manual_grads = mlp_backward(y_check, target_check, cache_check)

params = [torch.tensor(a, requires_grad=True)
          for a in [W1_check, b1_check, W2_check, b2_check]]
tw1, tb1, tw2, tb2 = params
tx = torch.tensor(x_check)
prediction_check = tw2 @ torch.relu(tw1 @ tx + tb1) + tb2
loss_check = 0.5 * (prediction_check - torch.tensor(target_check)).square().sum()
loss_check.backward()
for manual_grad, parameter in zip(manual_grads, params):
    np.testing.assert_allclose(manual_grad, parameter.grad.numpy(), atol=1e-12)
print("all four parameter gradients match")


# Input-gradient continuation and a finite-difference weight check.
tx_input = torch.tensor(x_check, requires_grad=True)
fixed_prediction = torch.tensor(W2_check) @ torch.relu(
    torch.tensor(W1_check) @ tx_input + torch.tensor(b1_check)) + torch.tensor(b2_check)
(0.5 * (fixed_prediction - torch.tensor(target_check)).square().sum()).backward()
h_check = W1_check @ x_check + b1_check
g_input = W1_check.T @ ((W2_check.T @ (y_check-target_check)) * (h_check > 0))
np.testing.assert_allclose(g_input, tx_input.grad.numpy(), atol=1e-12)
epsilon = 1e-5
assert np.min(np.abs(h_check)) > epsilon * abs(x_check[0]) * 2
plus, minus = W1_check.copy(), W1_check.copy()
plus[0, 0] += epsilon
minus[0, 0] -= epsilon
def objective(candidate):
    pred, _ = mlp_forward(x_check, candidate, b1_check, W2_check, b2_check)
    return 0.5 * np.sum((pred-target_check)**2)
numeric = (objective(plus)-objective(minus))/(2*epsilon)
np.testing.assert_allclose(numeric, manual_grads[0][0, 0], atol=1e-8, rtol=1e-6)
binary_inputs = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
xor_outputs = np.maximum(0, binary_inputs[:,0]-binary_inputs[:,1]) + np.maximum(
    0, binary_inputs[:,1]-binary_inputs[:,0])
np.testing.assert_array_equal(xor_outputs, [0,1,1,0])
print("input gradient, finite difference, and XOR representation verified")
