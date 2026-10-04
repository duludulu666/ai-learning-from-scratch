"""Executable controlled optimizer comparison for lesson 04.1."""

import copy
import torch
from torch import nn

torch.manual_seed(17)
torch.set_num_threads(1)
X = torch.randn(240, 4)
y = (X[:, :1] * 2 - X[:, 1:2] + 0.5 * X[:, 2:3].square())
y += 0.1 * torch.randn_like(y)
initial_model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
initial = copy.deepcopy(initial_model.state_dict())
order_rng = torch.Generator().manual_seed(18)
orders = [torch.randperm(180, generator=order_rng) for _ in range(35)]
factories = {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.03),
    "Momentum": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "Adam": lambda p: torch.optim.Adam(p, lr=0.01),
    "AdamW(no decay)": lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0),
}
for name, factory in factories.items():
    model = copy.deepcopy(initial_model)
    model.load_state_dict(initial)
    optimizer = factory(model.parameters())
    norms = []
    for order in orders:
        model.train()
        for indices in order.split(30):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(X[indices]) - y[indices]).square().mean()
            loss.backward()
            norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
            norms.append(norm.item())
            optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = (model(X[180:]) - y[180:]).square().mean().item()
    print(name, "validation MSE:", val_loss, "mean gradient norm:", sum(norms) / len(norms))


# Prescribed gradients isolate update rules from model/data effects.
gradients = [torch.tensor(g, dtype=torch.float64)
             for g in ([2., -0.5], [-1., 0.25], [0., 3.], [0.5, -2.])]
for name in ("SGD", "Momentum", "Adam"):
    parameter = nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    manual = parameter.detach().clone()
    if name == "SGD":
        opt = torch.optim.SGD([parameter], lr=0.03)
    elif name == "Momentum":
        opt = torch.optim.SGD([parameter], lr=0.03, momentum=0.9)
    else:
        opt = torch.optim.Adam([parameter], lr=0.03, betas=(0.9,0.999), eps=1e-8)
    velocity, first, second = [torch.zeros_like(manual) for _ in range(3)]
    for t, gradient in enumerate(gradients, 1):
        if name == "SGD":
            direction = gradient
        elif name == "Momentum":
            velocity = 0.9 * velocity + gradient
            direction = velocity
        else:
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient.square()
            direction = (first/(1-0.9**t)) / ((second/(1-0.999**t)).sqrt()+1e-8)
        manual -= 0.03 * direction
        parameter.grad = gradient.clone()
        opt.step()
        torch.testing.assert_close(parameter, manual, atol=1e-12, rtol=1e-10)
        if name == "Adam":
            torch.testing.assert_close(opt.state[parameter]["exp_avg"], first)
            torch.testing.assert_close(opt.state[parameter]["exp_avg_sq"], second)
    print("prescribed-gradient update verified:", name)

clipped = nn.Parameter(torch.zeros(2, dtype=torch.float64))
clipped.grad = torch.tensor([3.,4.], dtype=torch.float64)
before_norm = torch.nn.utils.clip_grad_norm_([clipped], max_norm=2.0)
torch.testing.assert_close(before_norm, torch.tensor(5., dtype=torch.float64))
torch.testing.assert_close(clipped.grad, torch.tensor([1.2,1.6], dtype=torch.float64),
                           atol=1e-6, rtol=1e-6)
torch.manual_seed(401)
weights = torch.empty(512, 256, dtype=torch.float64)
nn.init.xavier_normal_(weights, gain=1)
assert abs(weights.var(unbiased=False).item() / (2/(512+256)) - 1) < 0.03
nn.init.kaiming_normal_(weights, mode="fan_in", nonlinearity="relu")
assert abs(weights.var(unbiased=False).item() / (2/256) - 1) < 0.03
print("global clipping and initialization sample variances verified")
