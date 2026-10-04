import torch
from torch import nn
import torch.nn.functional as F
from multimodal_core import setup,save_figure
from generative_core import mixture,VAE,vae_terms

def main():
    setup(191)
    x=mixture(512,1)
    test=mixture(512,2)
    model=VAE()
    opt=torch.optim.Adam(model.parameters(),lr=.004)
    losses=[]
    for _ in range(650):
        reconstruction,mean,logvar=model(x)
        recon,kl=vae_terms(reconstruction,x,mean,logvar)
        loss=(recon+kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        reconstruction,mean,logvar=model(test)
        recon,kl=vae_terms(reconstruction,test,mean,logvar)
        generated=model.decoder(torch.randn(512,2))
    # Check analytic KL and the reparameterization gradient with common fixed noise.
    mu=torch.tensor([[.4,-.2]],dtype=torch.float64,requires_grad=True)
    lv=torch.tensor([[.1,-.3]],dtype=torch.float64,requires_grad=True)
    q=torch.distributions.Normal(mu,torch.exp(.5*lv))
    p=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(mu))
    expected=torch.distributions.kl_divergence(q,p).sum(-1)
    manual=.5*(mu.square()+lv.exp()-1-lv).sum(-1)
    torch.testing.assert_close(expected,manual)
    eps=torch.tensor([[.7,-.4]],dtype=torch.float64)
    z=mu+torch.exp(.5*lv)*eps
    gmu,glv=torch.autograd.grad(z.sum(),(mu,lv))
    torch.testing.assert_close(gmu,torch.ones_like(mu))
    torch.testing.assert_close(glv,.5*torch.exp(.5*lv.detach())*eps)
    # Actual alternating GAN updates, with explicit graph ownership.
    gen=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,2))
    disc=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,1))
    go=torch.optim.Adam(gen.parameters(),lr=.002)
    do=torch.optim.Adam(disc.parameters(),lr=.002)
    for _ in range(300):
        real=x[torch.randint(len(x),(64,))]
        go.zero_grad(set_to_none=True)
        fake=gen(torch.randn(64,2))
        dloss=F.softplus(-disc(real)).mean()+F.softplus(disc(fake.detach())).mean()
        do.zero_grad(set_to_none=True)
        dloss.backward()
        assert all(p.grad is None for p in gen.parameters())
        do.step()
        do.zero_grad(set_to_none=True)
        for parameter in disc.parameters():
            parameter.requires_grad_(False)
        gloss=F.softplus(-disc(gen(torch.randn(64,2)))).mean()
        go.zero_grad(set_to_none=True)
        gloss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in gen.parameters())
        assert all(p.grad is None for p in disc.parameters())
        go.step()
        for parameter in disc.parameters():
            parameter.requires_grad_(True)
    with torch.no_grad():
        gan=gen(torch.randn(512,2))
    assert torch.isfinite(generated).all() and torch.isfinite(gan).all()
    assert sum(losses[-50:])/50 < sum(losses[:50])/50
    print(f"VAE held-out recon={recon.mean():.4f}, KL={kl.mean():.4f}; KL/reparameterization checks passed")
    print(f"GAN final D loss={dloss.item():.4f}, G loss={gloss.item():.4f}; graph ownership passed")
    print("sample std: data",test.std(0).tolist(),"VAE means",generated.std(0).tolist(),"GAN",gan.std(0).tolist())
    print(f"negative-x fractions data={float((test[:,0]<0).float().mean()):.3f}, VAE={float((generated[:,0]<0).float().mean()):.3f}, GAN={float((gan[:,0]<0).float().mean()):.3f}")

    # Exact small latent model: verify log evidence = ELBO + posterior KL gap.
    likelihood=torch.tensor([.2,.8],dtype=torch.float64)
    prior=torch.tensor([.5,.5],dtype=torch.float64)
    approximate=torch.tensor([.4,.6],dtype=torch.float64)
    joint=likelihood*prior
    evidence=joint.sum()
    posterior=joint/evidence
    elbo=(approximate*(joint.log()-approximate.log())).sum()
    gap=(approximate*(approximate.log()-posterior.log())).sum()
    torch.testing.assert_close(elbo+gap,evidence.log())
    # For unit observation variance, full VAE samples add independent unit noise.
    obs_rng=torch.Generator().manual_seed(1919)
    observation=generated+torch.randn(generated.shape,generator=obs_rng)
    print(f"discrete ELBO={elbo.item():.6f}; log evidence={evidence.log().item():.6f}; KL gap={gap.item():.6f}")
    print("VAE full-observation sample std:",observation.std(0).tolist(),
          "; true population y variance=.1225 < model observation variance floor=1")
    # Non-saturating versus minimax generator-logit gradients at a badly rejected fake.
    a=torch.tensor(-5.,requires_grad=True)
    saturating=torch.log1p(-a.sigmoid())
    non_saturating=F.softplus(-a)
    small=torch.autograd.grad(saturating,a,retain_graph=True)[0]
    large=torch.autograd.grad(non_saturating,a)[0]
    torch.testing.assert_close(small,-a.detach().sigmoid())
    torch.testing.assert_close(large,a.detach().sigmoid()-1)
    assert large.abs()>100*small.abs()
    print("generator logit gradients at -5:",small.item(),large.item())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,3.4))
    plot_limit=max(4.,float(torch.cat([test,generated,observation,gan])[:,0].abs().max())+.3)
    y_limit=max(3.,float(torch.cat([test,generated,observation,gan])[:,1].abs().max())+.2)
    for ax,points,title in zip(axes,[test,generated,observation,gan],
                               ["Held-out data","VAE decoder means","VAE observation samples","GAN: 300-update failure"]):
        ax.scatter(points[:,0],points[:,1],s=5,alpha=.4)
        ax.set(xlim=(-plot_limit,plot_limit),ylim=(-y_limit,y_limit),title=title,xlabel="x1",ylabel="x2")
    fig.tight_layout()
    save_figure(fig,"19_vae_gan.png")
    plt.close(fig)

if __name__=="__main__":
    main()
