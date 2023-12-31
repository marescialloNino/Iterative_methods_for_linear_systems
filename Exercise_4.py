import numpy as np
import scipy as sp
from professor import numgrid
from professor import delsq
import ilupp
import numba
from mypcg import my_pcg
import matplotlib.pyplot as plt


tol=1e-8
maxit=5000

A = sp.io.loadmat('A')['A'].tocsr()
n=A.shape[0]

np.random.seed(500)
c=np.random.rand(n)
b=A@c

I=sp.sparse.eye(n)
F=sp.sparse.diags(1/A.diagonal())
L=ilupp.IChol0Preconditioner(A)

non_cond=my_pcg(A,b,tol,maxit,L=I)
Jacobi_cond=my_pcg(A,b,tol,maxit,L=F)
Cholesky_cond=my_pcg(A,b,tol,maxit,L=L)

fig, sol=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
sol[0].plot(non_cond[0]-c)
sol[1].plot(Jacobi_cond[0]-c)
sol[2].plot(Cholesky_cond[0]-c)
for a in sol:
    a.grid()
    a.set_ylabel('Solution')
sol[0].set_title('No conditioning')
sol[1].set_title('Jacobi')
sol[2].set_title('IC(0)')
plt.show()

plt.plot(non_cond[1],'r+-')
plt.plot(Jacobi_cond[1],'bo-',mfc='none')
plt.plot(Cholesky_cond[1],'g*-')
plt.yscale('log')
plt.xscale('log')
plt.legend(['No conditioning','Jacobi','IC(0)'])
plt.grid()
plt.ylabel('Residual norm')
plt.xlabel('Iteration number')
plt.title('Residual comparison')
plt.show()
