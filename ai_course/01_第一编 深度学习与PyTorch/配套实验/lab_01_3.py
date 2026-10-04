"""Executable examples for lesson 01.3; run from any working directory."""

import torch

previous_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)

x = torch.tensor(2.0, requires_grad=True)
u = x.square()
u.retain_grad()
loss = 3 * u + 2 * x + 1
loss.backward()
torch.testing.assert_close(x.grad, torch.tensor(14.0))
torch.testing.assert_close(u.grad, torch.tensor(3.0))

# A fresh graph, but the same leaf: gradients add.
loss_again = 3 * x.square() + 2 * x + 1
loss_again.backward()
torch.testing.assert_close(x.grad, torch.tensor(28.0))
x.grad = None

# Finite differences verify this scalar derivative independently.
def f(value):
    return 3 * value**2 + 2 * value + 1

eps = 1e-5
numeric = (f(2.0 + eps) - f(2.0 - eps)) / (2 * eps)
assert abs(numeric - 14.0) < 1e-8
print("finite difference:", numeric)

z = torch.tensor([1., 2.], requires_grad=True)
y = z.square()
y.backward(torch.tensor([1., 3.]))
torch.testing.assert_close(z.grad, torch.tensor([2., 12.]))

model = torch.nn.Linear(2, 1)
model.eval()
print("eval alone tracks gradients:", model(torch.ones(1, 2)).requires_grad)
with torch.no_grad():
    print("no_grad tracks gradients:", model(torch.ones(1, 2)).requires_grad)

torch.set_default_dtype(previous_dtype)


original = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
cloned = original.clone()
assert cloned.data_ptr() != original.data_ptr() and cloned.requires_grad
(cloned.square().sum()).backward()
torch.testing.assert_close(original.grad, 2 * original.detach())
detached_copy = original.detach().clone()
assert not detached_copy.requires_grad and detached_copy.data_ptr() != original.data_ptr()

# Mean-gradient accumulation over uneven batches: deterministic, no BatchNorm.
features = torch.tensor([[1., 0.], [0., 2.], [1., 3.]], dtype=torch.float64)
targets = torch.tensor([1., -1., 2.], dtype=torch.float64)
start = torch.tensor([0.2, -0.3], dtype=torch.float64)
full_w = start.clone().requires_grad_()
((features @ full_w - targets) ** 2).mean().backward()
micro_w = start.clone().requires_grad_()
for indices in [slice(0, 2), slice(2, 3)]:
    error = features[indices] @ micro_w - targets[indices]
    (error.square().sum() / len(targets)).backward()
torch.testing.assert_close(micro_w.grad, full_w.grad)
print("clone/detach and uneven micro-batch accumulation verified")
