import torch
from multimodal_core import setup

def quantize(w):
    if w.ndim!=2 or w.numel()==0 or not w.is_floating_point() or not torch.isfinite(w).all():
        raise ValueError("Need a finite nonempty floating-point weight matrix")
    maximum = w.abs().amax(dim=1,keepdim=True)
    scale = torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    q = torch.round(w/scale).clamp(-127,127).to(torch.int8)
    return q,scale

def main():
    setup(182)
    big = torch.tensor([1e5],dtype=torch.float32)
    assert torch.isinf(big.to(torch.float16)).all()
    assert torch.isfinite(big.to(torch.bfloat16)).all()
    w = torch.randn(8,16)
    w[0] = 0
    q, scale = quantize(w)
    restored = q.float()*scale
    assert (restored-w).abs().max() <= scale.max()/2+1e-6
    x = torch.randn(4,16)
    err = (x@restored.T-x@w.T).abs()
    bound = (x.abs() @ scale.expand_as(w).T)/2
    assert (err<=bound+1e-5).all()
    print(f"per-row int8 max weight error={(restored-w).abs().max():.6f}; max output error={err.max():.6f}")
    print("stored weight bytes:",w.numel()*4,"->",q.numel()+scale.numel()*4)
    layers,batch,length,kv_heads,head_dim,bytes_per = 12,2,128,4,16,2
    kv_bytes = 2*layers*batch*length*kv_heads*head_dim*bytes_per
    assert kv_bytes == 786432
    # Static loss scaling algebra, without fp16/autocast or dynamic overflow handling.
    p = torch.tensor([.1,.2],requires_grad=True)
    loss = p.square().sum()
    direct = torch.autograd.grad(loss,p,retain_graph=True)[0]
    scaled = torch.autograd.grad(1024*loss,p)[0]/1024
    torch.testing.assert_close(direct,scaled)

    # Unscale THEN clip: reversing the order changes the intended update.
    gradient=torch.tensor([3.,4.])
    correct_clip=gradient*min(1.,2./gradient.norm().item())
    scaled_gradient=1024*gradient
    wrong_clip=scaled_gradient*min(1.,2./scaled_gradient.norm().item())/1024
    torch.testing.assert_close(correct_clip,torch.tensor([1.2,1.6]))
    assert wrong_clip.norm() < correct_clip.norm()/100
    for malformed in [torch.tensor([[float("nan")]]),torch.empty(0,2),torch.ones(2,2,dtype=torch.long)]:
        try:
            quantize(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid quantization input accepted")
    # One-token speculative sampling identity: accepted mass + corrected residual.
    target=torch.tensor([.6,.3,.1],dtype=torch.float64)
    draft=torch.tensor([.2,.3,.5],dtype=torch.float64)
    accept=torch.minimum(torch.ones_like(target),target/draft)
    accepted_mass=draft*accept
    residual=(target-draft).clamp_min(0)
    rejection=1-accepted_mass.sum()
    corrected=accepted_mass+rejection*residual/residual.sum()
    torch.testing.assert_close(corrected,target)
    assert abs(rejection.item()-.4)<1e-12
    print("unscale-before-clip and one-token speculative distribution identity passed")
    print("KV cache analytic bytes:",kv_bytes,"; static scaling gradient equivalence passed")
    print("Quantize/dequantize reference uses FP32 matmul; no int8 speedup or GPU AMP claimed.")

if __name__ == "__main__":
    main()
