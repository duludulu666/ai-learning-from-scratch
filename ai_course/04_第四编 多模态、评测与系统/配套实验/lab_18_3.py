import copy
import torch
from torch import nn
import torch.nn.functional as F
from multimodal_core import setup

def main():
    setup(183)
    base = nn.Linear(3,2).double()
    x = torch.randn(5,3,dtype=torch.float64)
    y = torch.tensor([0,1,0,1,1])
    full = copy.deepcopy(base)
    F.cross_entropy(full(x),y).backward()
    counts = [2,3]
    correct, wrong = [],[]
    start=0
    for count in counts:
        local = copy.deepcopy(base)
        local_loss = F.cross_entropy(local(x[start:start+count]),y[start:start+count])
        start+=count
        # DDP averages across R ranks. Multiply local mean by R*n_r/N.
        scaled_grads = torch.autograd.grad(local_loss*(2*count/5),tuple(local.parameters()),retain_graph=True)
        naive_grads = torch.autograd.grad(local_loss,tuple(local.parameters()))
        correct.append(scaled_grads)
        wrong.append(naive_grads)
    naive_error=0.
    for j,p in enumerate(full.parameters()):
        combined = (correct[0][j]+correct[1][j])/2
        naive = (wrong[0][j]+wrong[1][j])/2
        torch.testing.assert_close(combined,p.grad,atol=1e-12,rtol=1e-12)
        naive_error += (naive-p.grad).abs().sum().item()
    assert naive_error>1e-4
    # Linear tensor parallel algebra: output concatenation versus input partial sums.
    w,b = base.weight.detach(),base.bias.detach()
    col_parts = [F.linear(x,part) for part in w.chunk(2,dim=0)]
    torch.testing.assert_close(torch.cat(col_parts,dim=-1)+b,F.linear(x,w,b))
    row_parts = [xx@ww.T for xx,ww in zip(x.split([1,2],dim=1),w.split([1,2],dim=1))]
    torch.testing.assert_close(sum(row_parts)+b,F.linear(x,w,b))
    # Checkpoint restores actual Adam state on an independent copy; CPU only.
    model = copy.deepcopy(base)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    def step(m,o):
        o.zero_grad()
        F.cross_entropy(m(x),y).backward()
        o.step()
    step(model,optimizer)
    model_state, opt_state = copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())
    step(model,optimizer)
    restored = copy.deepcopy(base)
    restored.load_state_dict(model_state)
    restored_opt = torch.optim.AdamW(restored.parameters(),lr=.01)
    restored_opt.load_state_dict(opt_state)
    step(restored,restored_opt)
    for p,q in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-12)

    # Add a zero-token rank: graph-connected empty selection, three-rank average.
    batches=[(x[:2],y[:2]),(x[2:],y[2:]),(x[:0],y[:0])]
    rank_grads=[]
    for xx,yy in batches:
        local=copy.deepcopy(base)
        logits=local(xx)
        summed=F.cross_entropy(logits,yy,reduction="sum") if len(yy) else logits.reshape(-1)[:0].sum()
        rank_grads.append(torch.autograd.grad(3*summed/len(y),tuple(local.parameters())))
    for j,p in enumerate(full.parameters()):
        torch.testing.assert_close(sum(g[j] for g in rank_grads)/3,p.grad)
        assert rank_grads[-1][j].abs().sum()==0
    # Empty global batch: optimizer.step must be skipped, even for zero gradients.
    no_data=copy.deepcopy(base)
    no_data_opt=torch.optim.AdamW(no_data.parameters(),lr=.1,weight_decay=.1)
    before=copy.deepcopy(no_data.state_dict())
    for p in no_data.parameters():
        p.grad=torch.zeros_like(p)
    no_data_opt.step()
    assert any(not torch.equal(before[k],v) for k,v in no_data.state_dict().items())
    print("zero-token rank normalization passed; demonstrated zero-gradient AdamW can still change weights")
    print(f"unequal-rank weighting matched; naive gradient L1 error={naive_error:.6f}")
    print("tensor partition identities and AdamW restoration passed")
    print("Single-process CPU simulation; no process group, network collective, or multi-GPU execution.")

if __name__ == "__main__":
    main()
