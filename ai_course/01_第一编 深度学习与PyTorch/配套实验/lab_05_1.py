"""Executable examples for lesson 05.1; run from any working directory."""

import numpy as np
import torch
import torch.nn.functional as F

def correlate2d(image, kernel, padding=0, stride=1, dilation=1):
    padded = np.pad(image, ((padding, padding), (padding, padding)))
    kh, kw = kernel.shape
    eh, ew = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1
    oh = (padded.shape[0] - eh) // stride + 1
    ow = (padded.shape[1] - ew) // stride + 1
    output = np.empty((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = padded[i*stride:i*stride+eh:dilation,
                           j*stride:j*stride+ew:dilation]
            output[i, j] = (patch * kernel).sum()
    return output

image = np.arange(1., 10.).reshape(3, 3)
kernel = np.array([[1., 0.], [0., -1.]])
manual = correlate2d(image, kernel)
x = torch.tensor(image)[None, None]
w = torch.tensor(kernel)[None, None].requires_grad_()
actual = F.conv2d(x, w)
np.testing.assert_allclose(actual.detach().numpy()[0, 0], manual)
np.testing.assert_allclose(manual, np.full((2, 2), -4.))

actual.sum().backward()  # All upstream entries are 1.
expected_dw = np.zeros_like(kernel)
for i in range(2):
    for j in range(2):
        expected_dw += image[i:i+2, j:j+2]
np.testing.assert_allclose(w.grad.numpy()[0, 0], expected_dw)

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
print("weight:", layer.weight.shape)
assert sum(p.numel() for p in layer.parameters()) == 8 * 3 * 3 * 3 + 8


# Scatter each weighted output gradient into its overlapping input patch.
upstream = np.array([[1.,-2.],[.5,3.]])
gx = np.zeros_like(image)
gk = np.zeros_like(kernel)
for i in range(2):
    for j in range(2):
        gx[i:i+2,j:j+2] += upstream[i,j]*kernel
        gk += upstream[i,j]*image[i:i+2,j:j+2]
xt = torch.tensor(image[None,None], requires_grad=True)
kt = torch.tensor(kernel[None,None], requires_grad=True)
(F.conv2d(xt,kt)*torch.tensor(upstream[None,None])).sum().backward()
np.testing.assert_allclose(xt.grad.numpy()[0,0],gx)
np.testing.assert_allclose(kt.grad.numpy()[0,0],gk)
epsilon = 1e-5
plus,minus = kernel.copy(),kernel.copy()
plus[0,1] += epsilon
minus[0,1] -= epsilon
numeric = ((correlate2d(image,plus)-correlate2d(image,minus))*upstream).sum()/(2*epsilon)
np.testing.assert_allclose(numeric,gk[0,1],atol=1e-8)

# Multi-channel outputs sum over input channels, not average them.
random = np.random.default_rng(501)
batch = random.normal(size=(2,2,4,4))
filters = random.normal(size=(3,2,2,2))
bias = random.normal(size=3)
manual = np.empty((2,3,3,3))
for n in range(2):
    for o in range(3):
        manual[n,o] = sum(correlate2d(batch[n,c],filters[o,c]) for c in range(2))+bias[o]
actual = F.conv2d(torch.tensor(batch),torch.tensor(filters),torch.tensor(bias))
np.testing.assert_allclose(manual,actual.numpy(),atol=1e-12)
print("convolution input/kernel gradients, finite difference, and channels verified")
