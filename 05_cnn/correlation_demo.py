"""Convolution (cross-correlation) from scratch for lesson 05.1 (NumPy + PyTorch).

Part 1: manual 2D correlation with padding/stride/dilation, verified vs F.conv2d.
Part 2: shared-weight kernel gradient: with L = sum(Y), the gradient is the
        sum of all input patches (the same kernel is used at every position).
Part 3: padding/stride/dilation combinations and Conv2d parameter count.
Part 4: input/kernel gradients under a nonuniform upstream, plus a
        finite-difference check of one kernel entry.
Part 5: multi-channel convolution sums over input channels (not averages).
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------- 1. Manual 2D correlation: slide kernel, dot each patch ----------
def correlate2d(image, kernel, padding=0, stride=1, dilation=1):
    padded = np.pad(image, ((padding, padding), (padding, padding)))   # zero pad all sides
    kh, kw = kernel.shape
    eh, ew = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1   # effective kernel size
    oh = (padded.shape[0] - eh) // stride + 1   # output height from valid slide starts
    ow = (padded.shape[1] - ew) // stride + 1
    output = np.empty((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = padded[i*stride:i*stride+eh:dilation,
                           j*stride:j*stride+ew:dilation]
            output[i, j] = (patch * kernel).sum()   # elementwise multiply then sum
    return output

image = np.arange(1., 10.).reshape(3, 3)
kernel = np.array([[1., 0.], [0., -1.]])
manual = correlate2d(image, kernel)   # 2x2 output, every entry -4
print("manual:\n", manual)

# ---------- 2. Match F.conv2d and verify the shared-weight gradient ----------
x = torch.tensor(image)[None, None]   # (1,1,3,3): add batch and channel dims
w = torch.tensor(kernel)[None, None].requires_grad_()
actual = F.conv2d(x, w)
np.testing.assert_allclose(actual.detach().numpy()[0, 0], manual)
np.testing.assert_allclose(manual, np.full((2, 2), -4.))

actual.sum().backward()   # upstream gradient all ones: L = sum(Y)
expected_dw = np.zeros_like(kernel)
for i in range(2):
    for j in range(2):
        expected_dw += image[i:i+2, j:j+2]   # kernel used at 4 positions: sum patches
np.testing.assert_allclose(w.grad.numpy()[0, 0], expected_dw)
print("w grad:\n", w.grad.numpy()[0, 0])

# ---------- 3. Padding/stride/dilation combinations and parameter count ----------
rng = np.random.default_rng(9)
image2 = rng.normal(size=(7, 7))
kernel2 = rng.normal(size=(3, 3))
for padding, stride, dilation in [(1, 2, 1), (2, 1, 2)]:
    manual2 = correlate2d(image2, kernel2, padding, stride, dilation)
    result = F.conv2d(torch.tensor(image2)[None, None],
                      torch.tensor(kernel2)[None, None],
                      padding=padding, stride=stride, dilation=dilation)
    np.testing.assert_allclose(result.numpy()[0, 0], manual2, atol=1e-12)
    print("output shape:", result.shape)

layer = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
print("weight:", layer.weight.shape)   # (C_out, C_in, kh, kw)
assert sum(p.numel() for p in layer.parameters()) == 8 * 3 * 3 * 3 + 8

# ---------- 4. Nonuniform upstream: input/kernel gradients + finite difference ----------
upstream = np.array([[1., -2.], [.5, 3.]])
gx = np.zeros_like(image)
gk = np.zeros_like(kernel)
for i in range(2):
    for j in range(2):
        gx[i:i+2, j:j+2] += upstream[i, j] * kernel   # scatter grad into input patch
        gk += upstream[i, j] * image[i:i+2, j:j+2]    # accumulate kernel gradient
xt = torch.tensor(image[None, None], requires_grad=True)
kt = torch.tensor(kernel[None, None], requires_grad=True)
(F.conv2d(xt, kt) * torch.tensor(upstream[None, None])).sum().backward()
np.testing.assert_allclose(xt.grad.numpy()[0, 0], gx)
np.testing.assert_allclose(kt.grad.numpy()[0, 0], gk)

epsilon = 1e-5   # central difference check of the kernel gradient
plus, minus = kernel.copy(), kernel.copy()
plus[0, 1] += epsilon
minus[0, 1] -= epsilon
numeric = ((correlate2d(image, plus) - correlate2d(image, minus)) * upstream).sum() / (2 * epsilon)
np.testing.assert_allclose(numeric, gk[0, 1], atol=1e-8)

# ---------- 5. Multi-channel: output sums over input channels, not averages ----------
random = np.random.default_rng(501)
batch = random.normal(size=(2, 2, 4, 4))
filters = random.normal(size=(3, 2, 2, 2))   # (C_out, C_in, kh, kw)
bias = random.normal(size=3)
manual = np.empty((2, 3, 3, 3))
for n in range(2):
    for o in range(3):
        manual[n, o] = sum(correlate2d(batch[n, c], filters[o, c]) for c in range(2)) + bias[o]
actual = F.conv2d(torch.tensor(batch), torch.tensor(filters), torch.tensor(bias))
np.testing.assert_allclose(manual, actual.numpy(), atol=1e-12)
print("convolution input/kernel gradients, finite difference, and channels verified")
