import statistics
import time
import torch
from multimodal_core import setup

def main():
    setup(181)
    batch, inner, out = 8, 128, 64
    x, w = torch.randn(batch,inner), torch.randn(inner,out)
    flops = 2*batch*inner*out
    minimum_bytes = (batch*inner + inner*out + batch*out)*4
    intensity = flops/minimum_bytes
    assert flops == 131072 and minimum_bytes == 38912
    for _ in range(10):
        x @ w
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(500):
            x @ w
        samples.append((time.perf_counter()-start)/500)
    param = torch.randn(1000)
    assert param.numel()*param.element_size() == 4000
    # Hypothetical roofline ceilings, not discovered hardware properties.
    peak_flops, bandwidth = 10e12, 500e9
    lower_bound = max(flops/peak_flops, minimum_bytes/bandwidth)

    # Checkpointing trades retained activations for recomputation, not changed gradients.
    import copy
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    ordinary=nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,2)).double()
    recomputed=copy.deepcopy(ordinary)
    a=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    c=a.detach().clone().requires_grad_()
    ordinary(a).square().mean().backward()
    checkpoint(recomputed,c,use_reentrant=False).square().mean().backward()
    torch.testing.assert_close(a.grad,c.grad)
    for p,q in zip(ordinary.parameters(),recomputed.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    print("activation-checkpoint gradients matched on CPU; memory saving not benchmarked")
    # FLOPs for backward through a dense trainable matrix, excluding bias/activation.
    assert 3*flops == 393216  # forward + dX + dW
    print(f"FLOPs={flops}, ideal bytes={minimum_bytes}, intensity={intensity:.3f} FLOP/byte")
    print(f"CPU matmul median={statistics.median(samples)*1e6:.3f} us; threads=1")
    print(f"hypothetical roofline lower bound={lower_bound*1e6:.3f} us (not a GPU measurement)")
    print("FP32 parameter+gradient+two Adam moments for 1000 parameters:",1000*16,"bytes")

if __name__ == "__main__":
    main()
