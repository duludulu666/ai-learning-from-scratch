import torch
import torch.nn.functional as F
from multimodal_core import setup,save_figure
from generative_core import mixture,Denoiser,mixture_noise_mean

def main():
    setup(192)
    steps=80
    beta=torch.linspace(.001,.12,steps)
    alpha=1-beta
    abar=alpha.cumprod(0)
    previous=torch.cat([torch.ones(1),abar[:-1]])
    posterior_var=beta*(1-previous)/(1-abar)
    assert posterior_var[0] == 0
    # Conditional marginal check at fixed x0, using independent Monte Carlo samples.
    fixed=torch.tensor([[2.,-1.]])
    mc=12000
    recursive=fixed.expand(mc,-1).clone()
    for t in range(30):
        recursive=alpha[t].sqrt()*recursive+beta[t].sqrt()*torch.randn_like(recursive)
    expected_mean=abar[29].sqrt()*fixed[0]
    expected_var=1-abar[29]
    torch.testing.assert_close(recursive.mean(0),expected_mean,atol=.025,rtol=0)
    torch.testing.assert_close(recursive.var(0,unbiased=False),expected_var.expand(2),atol=.025,rtol=0)
    x=mixture(1024,3)
    test=mixture(512,4)
    model=Denoiser()
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    for _ in range(1500):
        clean=x[torch.randint(len(x),(128,))]
        t=torch.randint(steps,(128,))
        noise=torch.randn_like(clean)
        noisy=abar[t,None].sqrt()*clean+(1-abar[t,None]).sqrt()*noise
        prediction=model(noisy,(t+1).float()/steps)
        loss=F.mse_loss(prediction,noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        g=torch.Generator().manual_seed(193)
        t=torch.randint(steps,(len(test),),generator=g)
        noise=torch.randn(test.shape,generator=g)
        noisy=abar[t,None].sqrt()*test+(1-abar[t,None]).sqrt()*noise
        held=F.mse_loss(model(noisy,(t+1).float()/steps),noise)
        baseline=noise.square().mean()
        samples=torch.randn(512,2)
        for t_index in reversed(range(steps)):
            time=torch.full((len(samples),),(t_index+1)/steps)
            eps_hat=model(samples,time)
            mean=(samples-beta[t_index]/(1-abar[t_index]).sqrt()*eps_hat)/alpha[t_index].sqrt()
            if t_index>0:
                samples=mean+posterior_var[t_index].sqrt()*torch.randn_like(samples)
            else:
                samples=mean
        oracle_noise=torch.randn_like(test)
        time_index=37
        corrupted=abar[time_index].sqrt()*test+(1-abar[time_index]).sqrt()*oracle_noise
        recovered=(corrupted-(1-abar[time_index]).sqrt()*oracle_noise)/abar[time_index].sqrt()
        torch.testing.assert_close(recovered,test,atol=1e-5,rtol=1e-5)
        clean_coefficient = previous[time_index].sqrt()*beta[time_index]/(1-abar[time_index])
        noisy_coefficient = alpha[time_index].sqrt()*(1-previous[time_index])/(1-abar[time_index])
        posterior_mean = clean_coefficient*test + noisy_coefficient*corrupted
        noise_form = (corrupted-beta[time_index]/(1-abar[time_index]).sqrt()*oracle_noise)/alpha[time_index].sqrt()
        torch.testing.assert_close(posterior_mean,noise_form,atol=1e-5,rtol=1e-5)

    # Known-generator oracle is not a learned baseline; compare it on the SAME noise.
    with torch.no_grad():
        oracle_prediction=mixture_noise_mean(noisy,abar[t])
        oracle_mse=F.mse_loss(oracle_prediction,noise)
        learned_prediction=model(noisy,(t+1).float()/steps)
        for left,right in [(0,20),(20,60),(60,80)]:
            use=(t>=left)&(t<right)
            print(f"times {left+1}..{right}: count={use.sum().item()}, learned MSE={F.mse_loss(learned_prediction[use],noise[use]).item():.4f}, oracle MSE={F.mse_loss(oracle_prediction[use],noise[use]).item():.4f}")
    probe=torch.tensor([[.3,-.2],[2.,1.]],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([.4,.8],dtype=torch.float64)
    centers=probe.new_tensor([[-2.,0.],[2.,0.]])
    variance=a*.35**2+(1-a)
    residual=probe[:,None,:]-a[:,None,None].sqrt()*centers
    log_density=torch.logsumexp(-.5*residual.square().sum(-1)/variance[:,None],dim=1)-torch.log(variance)
    score=torch.autograd.grad(log_density.sum(),probe)[0]
    torch.testing.assert_close(-mixture_noise_mean(probe,a)/(1-a[:,None]).sqrt(),score)
    # Posterior-mean parameterizations at every schedule index, including t=1.
    clean=torch.tensor([2.,-1.],dtype=torch.float64)
    eps=torch.tensor([.3,.7],dtype=torch.float64)
    bb=beta.double()
    aa=1-bb
    ab=aa.cumprod(0)
    prev=torch.cat([torch.ones(1,dtype=torch.float64),ab[:-1]])
    noisy_all=ab[:,None].sqrt()*clean+(1-ab[:,None]).sqrt()*eps
    original=(prev.sqrt()*bb/(1-ab))[:,None]*clean+(aa.sqrt()*(1-prev)/(1-ab))[:,None]*noisy_all
    rewritten=(noisy_all-(bb/(1-ab).sqrt())[:,None]*eps)/aa[:,None].sqrt()
    torch.testing.assert_close(original,rewritten,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(original[0],clean)
    print(f"known-generator oracle noise MSE={oracle_mse.item():.4f}; mixture score identity and all-time posterior means passed")
    assert torch.isfinite(samples).all() and held<baseline

    print(f"alpha_bar_T={abar[-1]:.6f}; held-out noise MSE={held:.4f}; zero predictor={baseline:.4f}")
    print(f"generated mean={samples.mean(0).tolist()}, std={samples.std(0).tolist()}, negative-x fraction={float((samples[:,0]<0).float().mean()):.4f}")
    print("Forward moments, oracle inversion, posterior-mean equivalence, t=1 zero noise, training/sampling passed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,points,title in zip(axes,[test,noisy,samples],["Clean held-out data","Mixed-time noisy inputs","Learned DDPM samples"]):
        ax.scatter(points[:,0],points[:,1],s=5,alpha=.4)
        ax.set(xlim=(-4,4),ylim=(-4,4),title=title,xlabel="x1",ylabel="x2")
    fig.tight_layout()
    save_figure(fig,"19_diffusion.png")
    plt.close(fig)

if __name__=="__main__":
    main()
