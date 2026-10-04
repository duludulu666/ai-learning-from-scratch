"""Executable examples for lesson 06.3; run from any working directory."""

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(14)
V, D = 6, 3
embedding = nn.Embedding(V, D).double()
ids = torch.tensor([1, 1, 4])
lookup = embedding(ids)
one_hot = F.one_hot(ids, num_classes=V).double()
torch.testing.assert_close(lookup, one_hot @ embedding.weight)

lookup.sum().backward()
expected = torch.zeros(V, D, dtype=torch.float64)
expected[1] = 2.0
expected[4] = 1.0
torch.testing.assert_close(embedding.weight.grad, expected)

padded_embedding = nn.Embedding(V, D, padding_idx=0)
batch_ids = torch.tensor([[1, 2, 0], [4, 0, 0]])
mask = batch_ids.ne(0)
vectors = padded_embedding(batch_ids)
summary = (vectors * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
torch.testing.assert_close(summary[1], padded_embedding.weight[4])
summary.sum().backward()
torch.testing.assert_close(padded_embedding.weight.grad[0], torch.zeros(D))
print("embedding output:", vectors.shape)
print("repeated-token row gradient:", embedding.weight.grad[1])

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(15)
torch.set_num_threads(1)
ids = torch.tensor([
    [1, 2, 0], [2, 1, 0], [1, 1, 2], [2, 2, 0],
    [3, 4, 0], [4, 3, 0], [3, 3, 4], [4, 4, 0],
])
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

class TokenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 4, padding_idx=0)
        self.head = nn.Linear(4, 2)

    def forward(self, tokens):
        valid = tokens.ne(0)
        vectors = self.embedding(tokens)
        pooled = (vectors * valid[..., None]).sum(1)
        pooled = pooled / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

model = TokenClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
for _ in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(ids), labels)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    print("training accuracy:", model(ids).argmax(-1).eq(labels).float().mean().item())
    first = model(torch.tensor([[1, 2, 0]]))
    reordered = model(torch.tensor([[2, 1, 0]]))
    torch.testing.assert_close(first, reordered)


# Nonuniform output sensitivities accumulate only at selected table rows.
table = nn.Embedding(6,2).double()
tokens = torch.tensor([1,1,4])
upstream = torch.tensor([[1.,2.],[3.,-1.],[-2.,4.]],dtype=torch.float64)
(table(tokens)*upstream).sum().backward()
expected = torch.zeros_like(table.weight)
expected[1] = torch.tensor([4.,1.],dtype=torch.float64)
expected[4] = torch.tensor([-2.,4.],dtype=torch.float64)
torch.testing.assert_close(table.weight.grad,expected)
with torch.no_grad():
    before = model(torch.tensor([[1,2,0]]))
    torch.testing.assert_close(before,model(torch.tensor([[1,2,0,0,0]])))
    saved_pad = model.embedding.weight[0].clone()
    model.embedding.weight[0].fill_(37.)
    torch.testing.assert_close(before,model(torch.tensor([[1,2,0,0,0]])))
    empty_logits = model(torch.tensor([[0,0,0]]))
    torch.testing.assert_close(empty_logits[0],model.head.bias)
    model.embedding.weight[0].copy_(saved_pad)
print("weighted lookup gradients, masked padding invariance, and empty-input policy verified")
