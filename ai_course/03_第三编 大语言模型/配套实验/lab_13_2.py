"""LoRA layer: dimensions, manual gradients, zero-B initialization, merge, training."""
import torch
from torch import nn
import torch.nn.functional as F
from llm_core import seed_all

seed_all()
class LoRALinear(nn.Module):
    def __init__(self,base,rank=2,alpha=2.):
        super().__init__()
        self.base=base
        for parameter in base.parameters(): parameter.requires_grad_(False)
        self.A=nn.Parameter(torch.randn(rank,base.in_features,dtype=base.weight.dtype)*.1)
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=base.weight.dtype))
        self.scale=alpha/rank
    def forward(self,x):
        return self.base(x)+self.scale*((x@self.A.T)@self.B.T)

# Hand-worked rank-1 example from the lesson.
a=torch.tensor([[1.,2.]],dtype=torch.float64)
b=torch.tensor([[3.],[4.]],dtype=torch.float64)
example_x=torch.tensor([1.,-1.],dtype=torch.float64)
torch.testing.assert_close((torch.eye(2,dtype=torch.float64)+b@a)@example_x,
                           torch.tensor([-2.,-5.],dtype=torch.float64))
layer=LoRALinear(nn.Linear(6,8,bias=False).double())
x=torch.randn(20,6,dtype=torch.float64)
upstream=torch.randn(20,8,dtype=torch.float64)
torch.testing.assert_close(layer(x),layer.base(x))
(layer(x)*upstream).sum().backward()
torch.testing.assert_close(layer.A.grad,torch.zeros_like(layer.A))
assert layer.B.grad.norm()>0 and layer.base.weight.grad is None
saved_rng=torch.get_rng_state()
dead=LoRALinear(nn.Linear(6,8,bias=False).double())
with torch.no_grad(): dead.A.zero_()
(dead(x)*upstream).sum().backward()
torch.testing.assert_close(dead.A.grad,torch.zeros_like(dead.A))
torch.testing.assert_close(dead.B.grad,torch.zeros_like(dead.B))
torch.set_rng_state(saved_rng)
with torch.no_grad(): layer.B.normal_(0,.1)
layer.zero_grad(set_to_none=True)
probe=x.clone().requires_grad_()
(layer(probe)*upstream).sum().backward()
manual_B=layer.scale*upstream.T@(x@layer.A.T)
manual_A=layer.scale*(upstream@layer.B).T@x
manual_x=upstream@(layer.base.weight+layer.scale*layer.B@layer.A)
torch.testing.assert_close(layer.B.grad,manual_B)
torch.testing.assert_close(layer.A.grad,manual_A)
torch.testing.assert_close(probe.grad,manual_x)
merged=layer.base.weight+layer.scale*layer.B@layer.A
torch.testing.assert_close(layer(x),F.linear(x,merged))
frozen=layer.base.weight.detach().clone()
target_delta=torch.randn(8,2,dtype=torch.float64)@torch.randn(2,6,dtype=torch.float64)*.05
target=F.linear(x,frozen+target_delta).detach()
optimizer=torch.optim.Adam([layer.A,layer.B],lr=.03)
first=F.mse_loss(layer(x),target).item()
for _ in range(350):
    optimizer.zero_grad(set_to_none=True)
    loss=F.mse_loss(layer(x),target)
    loss.backward();optimizer.step()
last=F.mse_loss(layer(x),target).item()
assert last<first/100
torch.testing.assert_close(layer.base.weight,frozen,atol=0,rtol=0)
trainable=sum(p.numel() for p in layer.parameters() if p.requires_grad)
assert trainable==2*(6+8)
print("trainable/base weights:",trainable,frozen.numel())
print("low-rank fixture MSE before/after:",first,last)
print("zero-init paths, full manual gradients, merge equivalence and frozen base passed")
# Quantization arithmetic only, not a quantized CUDA training implementation.
weights=torch.tensor([-.8,-.3,.2,.7],dtype=torch.float64)
scale=weights.abs().max()/7
quantized=(weights/scale).round().clamp(-7,7)
restored=quantized*scale
assert (restored-weights).abs().max()<=scale/2+1e-12
print("symmetric toy 4-bit max reconstruction error:",float((restored-weights).abs().max()))


# Central finite difference after training checks an adapter coordinate independently.
probe_index=(0,0);step=1e-6
layer.zero_grad(set_to_none=True)
objective=(layer(x)*upstream).sum()
objective.backward()
analytic=layer.A.grad[probe_index].item()
with torch.no_grad():
    original=layer.A[probe_index].item()
    layer.A[probe_index]=original+step
    plus=(layer(x)*upstream).sum().item()
    layer.A[probe_index]=original-step
    minus=(layer(x)*upstream).sum().item()
    layer.A[probe_index]=original
assert abs(analytic-(plus-minus)/(2*step))<1e-6
with torch.no_grad():
    torch.testing.assert_close(layer(x),F.linear(x,layer.base.weight+layer.scale*layer.B@layer.A))
fixed_scale=.1
saturated=round(2./fixed_scale)
saturated=max(-7,min(7,saturated))*fixed_scale
assert abs(2.-saturated)>fixed_scale/2
zeros=torch.zeros(4)
zero_scale=1. if zeros.abs().max()==0 else float(zeros.abs().max()/7)
assert torch.equal((zeros/zero_scale).round()*zero_scale,zeros)
assert 4*1_000_000+16*10_000==4_160_000
print("adapter finite difference, post-training merge, saturation/zero-scale and memory ledger passed")
