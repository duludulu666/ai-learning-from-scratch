"""Runnable examples for lesson 10.1."""

import copy
import torch
from torch import nn

torch.manual_seed(32)
D = 8
norm = nn.LayerNorm(D, elementwise_affine=False).double()
x = torch.randn(D, dtype=torch.float64)

pre_jacobian = torch.autograd.functional.jacobian(lambda z: z + 0 * norm(z), x)
post_jacobian = torch.autograd.functional.jacobian(lambda z: norm(z + 0 * z), x)
torch.testing.assert_close(pre_jacobian, torch.eye(D, dtype=torch.float64))
torch.testing.assert_close(post_jacobian @ torch.ones(D, dtype=torch.float64),
                           torch.zeros(D, dtype=torch.float64), atol=1e-10, rtol=0)

class ResidualMLP(nn.Module):
    def __init__(self, width, pre):
        super().__init__()
        self.pre = pre
        self.norm = nn.LayerNorm(width)
        self.branch = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                     nn.Linear(2*width, width))

    def forward(self, z):
        if self.pre:
            return z + self.branch(self.norm(z))
        return self.norm(z + self.branch(z))

depth_results = []
for depth in [1, 4, 12, 24]:
    torch.manual_seed(33)
    pre = nn.Sequential(*[ResidualMLP(D, True) for _ in range(depth)]).double()
    post = copy.deepcopy(pre)
    for block in post:
        block.pre = False
    probe = torch.randn(2, 3, D, dtype=torch.float64)
    source = torch.randn_like(probe)
    norms = []
    for stack in [pre, post]:
        z = source.clone().requires_grad_()
        (stack(z) * probe).sum().backward()
        norms.append(z.grad.norm().item())
    depth_results.append((depth, *norms))

print("depth / pre-LN input grad norm / post-LN input grad norm:", depth_results)
print("zero-branch Jacobian identities verified")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r[0] for r in depth_results], [r[1] for r in depth_results], "o-", label="Pre-LN")
ax.plot([r[0] for r in depth_results], [r[2] for r in depth_results], "o-", label="Post-LN")
ax.set(xlabel="Residual FFN depth", ylabel="Input gradient norm",
       title="One initialization and random probe; not a training comparison")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "10_1_norm_paths.png", dpi=150)
plt.close(fig)
