import torch
from eval_core import paired_bootstrap
from multimodal_core import setup
from torch import nn
import torch.nn.functional as F

def run_ablation():
    """Actual fixed-budget regression comparison; predeclared seeds, no selection."""
    def data(n,seed):
        g=torch.Generator().manual_seed(seed)
        x=torch.randn(n,2,generator=g)
        y=(x[:,0]*x[:,1])[:,None]+.05*torch.randn(n,1,generator=g)
        return x,y
    x,y=data(160,1721)
    tx,ty=data(120,1722)
    rows=[]
    for seed in [7,17,27]:
        features = [x,torch.cat([x,(x[:,0]*x[:,1])[:,None]],1)]
        tests = [tx,torch.cat([tx,(tx[:,0]*tx[:,1])[:,None]],1)]
        errors=[]
        for train_features,test_features in zip(features,tests):
            torch.manual_seed(seed)
            model=nn.Linear(train_features.shape[1],1)
            opt=torch.optim.SGD(model.parameters(),lr=.08)
            for _ in range(80):
                opt.zero_grad()
                F.mse_loss(model(train_features),y).backward()
                opt.step()
            with torch.no_grad():
                errors.append((model(test_features)-ty).square().squeeze())
        delta,ci=paired_bootstrap(errors[0],errors[1],seed=seed)
        rows.append((errors[0].mean().item(),errors[1].mean().item()))
        print(f"ablation seed={seed}: base MSE={rows[-1][0]:.5f}, interaction MSE={rows[-1][1]:.5f}, paired delta={delta:.5f}, CI={ci}")
    assert all(new<old for old,new in rows)
    print("Three predeclared seeds; 3 vs 4 parameters; same 80 updates, NOT exact equal FLOPs.")
    print("CI is per checkpoint over shared test examples; seeds are not independent test datasets.")


def main():
    setup(172)
    a = torch.tensor([1,1,1,1,1,1,0,0,0,0])
    b = torch.tensor([1,1,1,1,1,0,1,1,0,0])
    mean, ci = paired_bootstrap(a,b)
    assert abs(mean-.1) < 1e-12
    assert ci[0] <= 0 <= ci[1]
    same, interval = paired_bootstrap(a,a)
    assert same == 0 and interval == [0,0]
    gains, regressions = ((a==0)&(b==1)).sum(), ((a==1)&(b==0)).sum()
    # Two sources, 20 highly correlated examples each; resample sources, not frames.
    source_delta = torch.tensor([.5,-.4])
    group_mean, group_ci = paired_bootstrap(torch.zeros(2),source_delta)
    print(f"paired delta={mean:.3f}; percentile 95% interval={ci}; gains={gains}; regressions={regressions}")
    print(f"two-source illustrative delta={group_mean:.3f}; interval={group_ci}; too few groups for confidence")

    for aa,bb,rr in [([0],[float("nan")],10),([0],[1],0),([0],[1],True)]:
        try:
            paired_bootstrap(aa,bb,repeats=rr)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid bootstrap input accepted")
    # Unequal group sizes: equal-group and equal-example estimands differ.
    sizes=torch.tensor([1.,3.],dtype=torch.float64)
    means=torch.tensor([1.,-1.],dtype=torch.float64)
    assert means.mean()==0 and (means*sizes).sum()/sizes.sum()==-.5
    run_ablation()


if __name__ == "__main__":
    main()
