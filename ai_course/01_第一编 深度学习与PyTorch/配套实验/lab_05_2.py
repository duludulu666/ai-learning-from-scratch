"""Executable examples for lesson 05.2; run from any working directory."""

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(1)
x = torch.tensor([[[[1., 3.], [2., 0.]]]], requires_grad=True)
maximum = F.max_pool2d(x, 2)
maximum.sum().backward()
torch.testing.assert_close(x.grad, torch.tensor([[[[0., 1.], [0., 0.]]]]))
average = F.avg_pool2d(x.detach(), 2)
torch.testing.assert_close(average, torch.tensor([[[[1.5]]]]))

features = torch.randn(4, 3, 5, 5)
bn = nn.BatchNorm2d(3, affine=False, track_running_stats=False)
mean = features.mean(dim=(0, 2, 3), keepdim=True)
var = features.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
manual_bn = (features - mean) / torch.sqrt(var + bn.eps)
torch.testing.assert_close(bn(features), manual_bn, atol=1e-6, rtol=1e-5)

sequence = torch.randn(2, 4, 6)
ln = nn.LayerNorm(6, elementwise_affine=False)
mu = sequence.mean(-1, keepdim=True)
variance = sequence.var(-1, unbiased=False, keepdim=True)
torch.testing.assert_close(ln(sequence), (sequence - mu) / (variance + ln.eps).sqrt())

branch = nn.Linear(3, 3, bias=False)
with torch.no_grad():
    branch.weight.zero_()
inputs = torch.randn(2, 3, requires_grad=True)
outputs = inputs + branch(inputs)
outputs.sum().backward()
torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
torch.testing.assert_close(outputs, inputs)
print("pooling, normalization, and residual checks passed")


# Pooling's backward rule for a non-unit upstream scalar.
avg_input = torch.tensor([[[[1.,3.],[2.,0.]]]], dtype=torch.float64, requires_grad=True)
(2*F.avg_pool2d(avg_input,2)).sum().backward()
torch.testing.assert_close(avg_input.grad,torch.full_like(avg_input,.5))

# Default BN: biased train variance, unbiased running variance, then frozen eval.
torch.manual_seed(502)
data = torch.randn(4,3,2,2,dtype=torch.float64)
tracked = nn.BatchNorm2d(3,affine=False,momentum=.1).double()
batch_mean = data.mean((0,2,3))
batch_var = data.var((0,2,3),unbiased=False)
batch_unbiased = data.var((0,2,3),unbiased=True)
result = tracked(data)
torch.testing.assert_close(result,(data-batch_mean[None,:,None,None])/
                           (batch_var[None,:,None,None]+tracked.eps).sqrt())
torch.testing.assert_close(tracked.running_mean,.1*batch_mean)
torch.testing.assert_close(tracked.running_var,.9*torch.ones_like(batch_var)+.1*batch_unbiased)
tracked.eval()
saved_buffers = {k:v.clone() for k,v in tracked.named_buffers()}
expected = (data-tracked.running_mean[None,:,None,None]) / (
    tracked.running_var[None,:,None,None]+tracked.eps).sqrt()
torch.testing.assert_close(tracked(data),expected)
changed = data.clone()
changed[1:] += 100
torch.testing.assert_close(tracked(data)[0],tracked(changed)[0])
for k,v in tracked.named_buffers():
    torch.testing.assert_close(v,saved_buffers[k])

# Batch-dependent normalization is observable even in eval if tracking is off.
untracked = nn.BatchNorm2d(3,affine=False,track_running_stats=False).double()
assert not torch.allclose(untracked(data)[0],untracked(changed)[0])
untracked.eval()
assert not torch.allclose(untracked(data)[0],untracked(changed)[0])
per_example = nn.LayerNorm([3,2,2],elementwise_affine=False).double()
torch.testing.assert_close(per_example(data)[0],per_example(changed)[0])
wrong_axis = nn.LayerNorm(2,elementwise_affine=False).double()
# It runs because W=2, but normalizes width, not the 3 channels.
torch.testing.assert_close(wrong_axis(data).mean(-1),torch.zeros(4,3,2,dtype=torch.float64),
                           atol=1e-12,rtol=0)
cancel = torch.randn(2,3,requires_grad=True)
(cancel-cancel).sum().backward()
torch.testing.assert_close(cancel.grad,torch.zeros_like(cancel))
print("BN train/eval buffers, batch dependence, LN axes, and residual cancellation verified")


# Derive LayerNorm's backward through mean and variance, not just its forward.
torch.manual_seed(503)
ln_input = torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
affine_ln = nn.LayerNorm(5).double()
with torch.no_grad():
    affine_ln.weight.copy_(torch.randn(5,dtype=torch.float64))
    affine_ln.bias.copy_(torch.randn(5,dtype=torch.float64))
upstream = torch.randn_like(ln_input)
(affine_ln(ln_input)*upstream).sum().backward()
with torch.no_grad():
    centered = ln_input-ln_input.mean(-1,keepdim=True)
    inverse_scale = (centered.square().mean(-1,keepdim=True)+affine_ln.eps).rsqrt()
    normalized = centered*inverse_scale
    u = upstream*affine_ln.weight
    manual_dx = inverse_scale*(u-u.mean(-1,keepdim=True)-
                               normalized*(u*normalized).mean(-1,keepdim=True))
    torch.testing.assert_close(manual_dx,ln_input.grad,atol=1e-11,rtol=1e-9)
    torch.testing.assert_close(affine_ln.weight.grad,(upstream*normalized).sum((0,1)))
    torch.testing.assert_close(affine_ln.bias.grad,upstream.sum((0,1)))
    torch.testing.assert_close(manual_dx.sum(-1),torch.zeros(2,3,dtype=torch.float64),
                               atol=1e-11,rtol=0)
print("LayerNorm input/affine backward through statistics verified")
