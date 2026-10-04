"""Two-dimensional synthetic generation utilities."""
import torch
from torch import nn

def mixture(n,seed):
    g=torch.Generator().manual_seed(seed)
    component=torch.randint(0,2,(n,),generator=g)
    x=.35*torch.randn(n,2,generator=g)
    x[:,0]+=torch.where(component==0,-2.,2.)
    return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2,32),nn.Tanh())
        self.mean=nn.Linear(32,2)
        self.logvar=nn.Linear(32,2)
        self.decoder=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,2))
    def encode(self,x):
        h=self.encoder(x)
        return self.mean(h),self.logvar(h)
    def forward(self,x):
        mean,logvar=self.encode(x)
        z=mean+torch.exp(.5*logvar)*torch.randn_like(mean)
        return self.decoder(z),mean,logvar

def vae_terms(reconstruction,x,mean,logvar):
    # Unit-variance Gaussian likelihood, constants omitted; per-example sum.
    recon=.5*(reconstruction-x).square().sum(-1)
    kl=.5*(mean.square()+logvar.exp()-1-logvar).sum(-1)
    return recon,kl


def mixture_noise_mean(noisy, alpha_bar):
    """Analytic Bayes predictor for this known equal-weight Gaussian mixture.
    Diagnostic only: it uses the known data generator, not learned parameters.
    alpha_bar: (B,), noisy: (B,2).
    """
    centers=noisy.new_tensor([[-2.,0.],[2.,0.]])
    a=alpha_bar[:,None,None]
    means=a.sqrt()*centers[None,:,:]
    variance=a*.35**2+(1-a)
    residual=noisy[:,None,:]-means
    log_weights=-.5*(residual.square()/variance).sum(-1)
    responsibilities=log_weights.softmax(-1)
    score=(-residual/variance*responsibilities[:,:,None]).sum(1)
    return -(1-alpha_bar[:,None]).sqrt()*score

class Denoiser(nn.Module):

    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x,t_fraction):
        return self.net(torch.cat([x,t_fraction[:,None]],dim=-1))
