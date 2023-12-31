import numpy as np
import scipy as sp
from professor import numgrid
from professor import delsq
from mypcg import my_pcg
import ilupp

tol=1e-8
maxit=5000

for i in range(0,4,1):
    iteration=np.zeros(4)
    
    nx=2+100*(2**i)
    A=delsq(numgrid('S',nx))
    n=A.shape[0]
    b=1/np.sqrt(np.arange(1,n+1,1))
    
    I=sp.sparse.eye(n)
    L0=ilupp.IChol0Preconditioner(A)
    L2=ilupp.ICholTPreconditioner(A, threshold=1e-2)
    L3=ilupp.ICholTPreconditioner(A, threshold=1e-3)
    
    iteration[0]=my_pcg(A,b,tol,maxit,L=I)[2]
    iteration[1]=my_pcg(A,b,tol,maxit,L=L0)[2]
    iteration[2]=my_pcg(A,b,tol,maxit,L=L2)[2]
    iteration[3]=my_pcg(A,b,tol,maxit,L=L3)[2]
    print(nx-2,'\t',iteration,'\n')
