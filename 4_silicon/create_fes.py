import numpy as np
import matplotlib.pyplot as plt

colvar="../COLVAR"
data=np.loadtxt(colvar)

time=data[:,0]
cv_m=data[:,1]
cv_mt=data[:,2]
ene=data[:,3]
bias=data[:,5]
kl=data[:,7]
#coft=data[:,8]
lr=data[:,8]

t=30000
beta=1./13.3

cv_min=0.
cv_max=216.
nbins=532

cv_m=cv_m[time>t]
cv_mt=cv_mt[time>t]
bias=bias[time>t]
bias=bias + np.max(bias)

def create_fes(cv,Min,Max,bins,beta,bias):
    x=np.linspace(Min,Max,bins)
    dx=(Max-Min)/(bins-1)
    h=np.zeros(bins)
    weights=np.exp(beta*bias)
        
    for c,b in np.c_[cv,weights]:
        index=int(c/dx)
        h[index]+=b

    sum_w=np.sum(weights)
    h=h/sum_w
    
    f=-1./beta *np.log(h)
    
    return x,h,f


x,h,f=create_fes(cv_mt,cv_min,cv_max,nbins,beta,bias)
#plt.plot(x,f)

np.savetxt('reweight-fes.dat', np.c_[x,f,h], fmt='%.18e', header="cv - fes - histogram")
