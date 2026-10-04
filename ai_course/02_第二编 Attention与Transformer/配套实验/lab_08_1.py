"""Runnable examples for lesson 08.1."""

import math
import torch
from torch import nn

torch.manual_seed(24)
B, T, D, dk, dv = 2, 4, 6, 3, 5
X = torch.randn(B, T, D, dtype=torch.float64)
q = nn.Linear(D, dk, bias=False).double()
k = nn.Linear(D, dk, bias=False).double()
v = nn.Linear(D, dv, bias=False).double()

def attend(inputs):
    Q, K, V = q(inputs), k(inputs), v(inputs)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
    weights = scores.softmax(-1)
    return weights @ V, weights

output, weights = attend(X)
loop_output = torch.empty_like(output)
for n in range(B):
    for i in range(T):
        query_scores = torch.stack([(q(X[n, i]) * k(X[n, j])).sum()
                                    for j in range(T)]) / math.sqrt(dk)
        loop_output[n, i] = (query_scores.softmax(0)[:, None] * v(X[n])).sum(0)
torch.testing.assert_close(output, loop_output)

permutation = torch.tensor([2, 0, 3, 1])
reordered, _ = attend(X[:, permutation])
torch.testing.assert_close(reordered, output[:, permutation])
torch.testing.assert_close(reordered.mean(1), output.mean(1))

changed = X.clone()
changed[1] += 100.0
unchanged_first_example, _ = attend(changed)
torch.testing.assert_close(unchanged_first_example[0], output[0])
small_Q = torch.tensor([[0.], [math.log(3)]], dtype=torch.float64)
small_K = torch.tensor([[1.], [0.]], dtype=torch.float64)
small_V = torch.tensor([[2.], [8.]], dtype=torch.float64)
small_A = (small_Q @ small_K.T).softmax(-1)
torch.testing.assert_close(small_A @ small_V, torch.tensor([[5.], [3.5]], dtype=torch.float64))
torch.testing.assert_close(small_A.sum(-1), torch.ones(2, dtype=torch.float64))
assert not torch.allclose(small_A.sum(0), torch.ones(2, dtype=torch.float64))
print("output / weights:", output.shape, weights.shape)
print("vectorization, equivariance, and batch isolation verified")
