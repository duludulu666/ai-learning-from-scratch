"""Tiny ResNet for lesson 05.3 (PyTorch).

BasicBlock: two 3x3 conv+BN, residual shortcut (1x1 conv when shape changes).
TinyResNet: stem -> two blocks -> adaptive pool -> linear head.
Also verifies freezing the backbone (requires_grad=False + eval) so only
the linear head updates.
"""

import torch
from torch import nn

torch.manual_seed(12)


# ---------- 1. BasicBlock: residual branch + identity shortcut ----------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        # shortcut: identity when shape matches, 1x1 conv when stride/channel changes
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
        return self.activation(z + identity)


# ---------- 2. TinyResNet: stem + two blocks + pool + linear head ----------
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------- 3. Forward/backward, then freeze backbone, train head only ----------
model = TinyResNet()
images = torch.randn(4, 1, 32, 32)
labels = torch.tensor([0, 1, 2, 0])
logits = model(images)
assert logits.shape == (4, 3)
nn.CrossEntropyLoss()(logits, labels).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters())

for p in model.backbone.parameters():
    p.requires_grad_(False)
model.zero_grad(set_to_none=True)   # clear grads left by the first backward
model.backbone.eval()               # freeze BN running buffers too
opt = torch.optim.SGD(model.head.parameters(), lr=0.1)
opt.zero_grad()
nn.CrossEntropyLoss()(model(images), labels).backward()
opt.step()
assert all(p.grad is None for p in model.backbone.parameters())

print("logits:", logits.shape, "parameters:", sum(p.numel() for p in model.parameters()))
print("ResNet forward/backward/head-only training verified")
