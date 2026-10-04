"""Executable examples for lesson 02.2; run from any working directory."""

import numpy as np
import torch
import torch.nn.functional as F

logits = np.array([[2., 1., 0.], [1000., 999., 998.]], dtype=np.float64)
labels = np.array([0, 2])
shifted = logits - logits.max(axis=1, keepdims=True)
log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
probs = np.exp(log_probs)
manual_loss = -log_probs[np.arange(2), labels].mean()
one_hot = np.eye(3)[labels]
manual_grad = (probs - one_hot) / 2

z = torch.tensor(logits, requires_grad=True)
targets = torch.tensor(labels, dtype=torch.long)
loss = F.cross_entropy(z, targets)
loss.backward()
np.testing.assert_allclose(loss.item(), manual_loss)
np.testing.assert_allclose(z.grad.numpy(), manual_grad)

soft_targets = F.one_hot(targets, num_classes=3).to(z.dtype)
torch.testing.assert_close(F.cross_entropy(z.detach(), soft_targets), loss.detach())
torch.testing.assert_close(F.cross_entropy(z.detach() + 123.0, targets), loss.detach())
print("probabilities:\n", probs)
print("loss:", loss.item())
print("gradient:\n", z.grad)

torch.manual_seed(3)
X = torch.randn(90, 2, dtype=torch.float64)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).long()
model = torch.nn.Linear(2, 2).double()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
for _ in range(120):
    optimizer.zero_grad(set_to_none=True)
    train_loss = F.cross_entropy(model(X), y)
    train_loss.backward()
    optimizer.step()
print("training accuracy:", (model(X).argmax(-1) == y).double().mean().item())


soft_z = torch.tensor([[np.log(2), 0., 0.]], dtype=torch.float64, requires_grad=True)
soft_t = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float64)
F.cross_entropy(soft_z, soft_t).backward()
torch.testing.assert_close(soft_z.grad, soft_z.detach().softmax(-1) - soft_t)
torch.testing.assert_close(soft_z.grad.sum(-1), torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)

# Check CE all the way through an affine classifier, including its input.
torch.manual_seed(202)
features = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
weight = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
bias = torch.randn(2, dtype=torch.float64, requires_grad=True)
target_ids = torch.tensor([0, 1, 1, 0])
scores = features @ weight.T + bias
G = (scores.detach().softmax(-1) - F.one_hot(target_ids, 2)) / 4
F.cross_entropy(scores, target_ids).backward()
torch.testing.assert_close(weight.grad, G.T @ features.detach())
torch.testing.assert_close(bias.grad, G.sum(0))
torch.testing.assert_close(features.grad, G @ weight.detach())

# Extreme finite logits remain a well-defined CE input; softmax-then-log need not.
extreme = torch.tensor([[1000., -1000.]], dtype=torch.float64, requires_grad=True)
extreme_loss = F.cross_entropy(extreme, torch.tensor([1]))
extreme_loss.backward()
assert torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()
torch.testing.assert_close(extreme_loss, torch.tensor(2000., dtype=torch.float64))
print("genuinely soft targets, affine gradients, and extreme-logit stability verified")
