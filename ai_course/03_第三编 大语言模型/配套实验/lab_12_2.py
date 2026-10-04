"""Illustrative scaling law with explicitly synthetic constants, not fitted frontier data."""
import math
import numpy as np
from llm_core import figure_path

irreducible,A,B,alpha,beta=1.,2.,3.,.5,.5
def loss(n,d):
    return irreducible+A*n**(-alpha)+B*d**(-beta)
budget=100.
# Units: n,d are normalized toy units; budget is C/kappa, so n*d=budget.
optimal_n=((alpha*A)/(beta*B)*budget**beta)**(1/(alpha+beta))
optimal_d=budget/optimal_n
grid=np.logspace(-1,3,1200)
values=loss(grid,budget/grid)
grid_best=float(grid[values.argmin()])
assert abs(math.log(grid_best/optimal_n))<.01
eps=1e-5
derivative=(loss(optimal_n+eps,budget/(optimal_n+eps))-
            loss(optimal_n-eps,budget/(optimal_n-eps)))/(2*eps)
assert abs(derivative)<1e-7
print("toy compute-optimal N,D,loss:",optimal_n,optimal_d,loss(optimal_n,optimal_d))
print("too small/balanced/too large:",[loss(n,budget/n) for n in [1.,optimal_n,100.]])
for multiplier in [1,4,16]:
    n=((alpha*A)/(beta*B)*(budget*multiplier)**beta)**(1/(alpha+beta))
    d=budget*multiplier/n
    print("compute multiplier / N / D:",multiplier,n,d)
path=figure_path("12_2_budget.png")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4))
ax.semilogx(grid,values,label="Fixed compute: D = 100 / N")
ax.axvline(optimal_n,ls="--",color="black",label="Analytic toy optimum")
ax.set(xlabel="Model size N (normalized toy units)",ylabel="Illustrative loss",
       title="A fixed budget creates a model/data tradeoff",ylim=(1,15))
ax.legend()
fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
print("closed-form optimum and finite-difference derivative verified")


# Unequal exponents and constrained optima: the symmetric example is not universal.
aa,bb,al,be,kk=2.,3.,.4,.2,100.
unconstrained=((al*aa)/(be*bb)*kk**be)**(1/(al+be))
n8=((al*aa)/(be*bb)*(8*kk)**be)**(1/(al+be))
assert math.isclose(n8/unconstrained,2.)
assert math.isclose((8*kk/n8)/(kk/unconstrained),4.)
def unequal(n):
    return 1+aa*n**(-al)+bb*(kk/n)**(-be)
step=unconstrained*1e-5
assert abs((unequal(unconstrained+step)-unequal(unconstrained-step))/(2*step))<1e-8
feasible=np.linspace(.5,2.,1000)
clipped=min(max(unconstrained,.5),2.)
assert math.isclose(float(feasible[np.argmin(unequal(feasible))]),clipped)
assert math.isclose(6*1e8*1e9/1e13/3600,16+2/3)
print("unequal exponent scaling, feasible-bound optimum and compute-unit conversion passed")
