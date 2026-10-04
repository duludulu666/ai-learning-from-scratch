"""Executable examples for lesson 05.3; run from any working directory."""

import torch
from torch import nn

torch.manual_seed(12)
torch.set_num_threads(1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        z = self.activation(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        assert z.shape == identity.shape
        return self.activation(z + identity)

class TinyResNet(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            BasicBlock(8, 8),
            BasicBlock(8, 16, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")

    def forward(self, x):
        return self.head(self.backbone(x))

model = TinyResNet()
images = torch.randn(4, 1, 32, 32)
labels = torch.tensor([0, 1, 2, 0])
logits = model(images)
assert logits.shape == (4, 3)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
assert all(p.grad is not None for p in model.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.parameters())

# Freeze the backbone for a head-only training step.
for parameter in model.backbone.parameters():
    parameter.requires_grad_(False)
model.zero_grad(set_to_none=True)
model.train()
model.backbone.eval()  # Keep frozen BatchNorm running statistics fixed too.
optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
backbone_before = {k:v.clone() for k,v in model.backbone.state_dict().items()}
head_before = {k:v.clone() for k,v in model.head.state_dict().items()}
optimizer.zero_grad(set_to_none=True)
nn.CrossEntropyLoss()(model(images), labels).backward()
optimizer.step()
assert all(p.grad is None for p in model.backbone.parameters())
for key,value in model.backbone.state_dict().items():
    torch.testing.assert_close(value,backbone_before[key],atol=0,rtol=0)
assert any(not torch.equal(value,head_before[key]) for key,value in model.head.state_dict().items())
print("logits:", logits.shape, "parameters:", sum(p.numel() for p in model.parameters()))


# Freezing parameters alone does not freeze running statistics.
unfrozen_stats = nn.BatchNorm2d(3)
for parameter in unfrozen_stats.parameters():
    parameter.requires_grad_(False)
before = unfrozen_stats.running_mean.clone()
unfrozen_stats.train()
unfrozen_stats(torch.full((2,3,2,2),5.))
assert not torch.equal(unfrozen_stats.running_mean,before)
assert all(parameter.grad is None for parameter in unfrozen_stats.parameters())
print("backbone values/buffers frozen, head changed, BN negative control verified")
