"""Runnable examples for lesson 07.3."""

import numpy as np
import torch

rng = np.random.default_rng(23)
Q = rng.normal(size=(3, 4))
K = rng.normal(size=(5, 4))
V = rng.normal(size=(5, 2))
Gout = rng.normal(size=(3, 2))
scale = Q.shape[-1] ** -0.5

S = scale * Q @ K.T
S = S - S.max(axis=-1, keepdims=True)
A = np.exp(S)
A /= A.sum(axis=-1, keepdims=True)
O = A @ V
GA = Gout @ V.T
GV = A.T @ Gout
GS = A * (GA - (GA * A).sum(axis=-1, keepdims=True))
GQ, GK = scale * GS @ K, scale * GS.T @ Q

qt = torch.tensor(Q, requires_grad=True)
kt = torch.tensor(K, requires_grad=True)
vt = torch.tensor(V, requires_grad=True)
out = (scale * qt @ kt.T).softmax(-1) @ vt
(out * torch.tensor(Gout)).sum().backward()
np.testing.assert_allclose(O, out.detach().numpy(), atol=1e-12)
for manual, variable in [(GQ, qt), (GK, kt), (GV, vt)]:
    np.testing.assert_allclose(manual, variable.grad.numpy(), atol=1e-12)
np.testing.assert_allclose(GS.sum(-1), 0, atol=1e-12)
print("forward and Q/K/V gradients match")

variance_results = []
for width in [4, 16, 64, 256]:
    q = rng.normal(size=(4000, width))
    k = rng.normal(size=(4000, width))
    raw = (q * k).sum(-1)
    variance_results.append((width, raw.var(), (raw / np.sqrt(width)).var()))
print("width / raw variance / scaled variance:", variance_results)


# A second, independent graph includes all affine projections and their biases.
rng_projection = np.random.default_rng(230)
X = rng_projection.normal(size=(3, 4))
WQ, WK = [rng_projection.normal(size=(2, 4)) for _ in range(2)]
WV = rng_projection.normal(size=(3, 4))
bQ, bK, bV = [rng_projection.normal(size=n) for n in (2, 2, 3)]
upstream = rng_projection.normal(size=(3, 3))
q, k, v = X @ WQ.T + bQ, X @ WK.T + bK, X @ WV.T + bV
c = 2 ** -0.5
scores = c * q @ k.T
a = np.exp(scores - scores.max(-1, keepdims=True))
a /= a.sum(-1, keepdims=True)
ga = upstream @ v.T
gv = a.T @ upstream
gs = a * (ga - (a * ga).sum(-1, keepdims=True))
gq, gk = c * gs @ k, c * gs.T @ q
manual_gradients = [
    gq @ WQ + gk @ WK + gv @ WV,
    gq.T @ X, gk.T @ X, gv.T @ X,
    gq.sum(0), gk.sum(0), gv.sum(0),
]
variables = [torch.tensor(z, requires_grad=True)
             for z in (X, WQ, WK, WV, bQ, bK, bV)]
xt, wqt, wkt, wvt, bqt, bkt, bvt = variables
qt, kt, vt = xt @ wqt.T + bqt, xt @ wkt.T + bkt, xt @ wvt.T + bvt
read = (c * qt @ kt.T).softmax(-1) @ vt
(read * torch.tensor(upstream)).sum().backward()
for manual, variable in zip(manual_gradients, variables):
    np.testing.assert_allclose(manual, variable.grad.numpy(), atol=1e-11, rtol=1e-10)

# Central differences check a parameter without reusing the analytic backward.
def projection_loss(candidate_weight):
    projected_q = X @ candidate_weight.T + bQ
    s = c * projected_q @ k.T
    probabilities = np.exp(s - s.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    return ((probabilities @ v) * upstream).sum()

epsilon = 1e-5
plus, minus = WQ.copy(), WQ.copy()
plus[0, 1] += epsilon
minus[0, 1] -= epsilon
finite_difference = (projection_loss(plus) - projection_loss(minus)) / (2 * epsilon)
np.testing.assert_allclose(finite_difference, manual_gradients[1][0, 1],
                           atol=1e-8, rtol=1e-6)
print("shared input, all projection weights/biases, and finite difference verified")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([r[0] for r in variance_results], [r[1] for r in variance_results],
        "o-", label="Raw score variance")
ax.plot([r[0] for r in variance_results], [r[2] for r in variance_results],
        "o-", label="Scaled score variance")
ax.set(xlabel="Key/query width", ylabel="Measured variance",
       title="Dot-product variance under independent unit-variance inputs")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "07_3_variance.png", dpi=150)
plt.close(fig)
