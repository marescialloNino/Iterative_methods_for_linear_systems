import numpy as np
import scipy as sp
from professor import numgrid
from professor import delsq
import ilupp
import matplotlib.pyplot as plt
import time

def my_pcg(A, b, tol, maxit, L):
    x=L@b
    r=b-(A@x)
    p=L@r
    
    resvec=[]
    
    for i in range(0,maxit,1): 
        c=p@A@p
        
        a=(r@p)/c
        x=x+(a*p)
        r=r-a*(A@p)

        v=L@r
        
        b=-(v@A@p)/c
        p=v+b*p
        
        resvec.append(sp.linalg.norm(r))
        if sp.linalg.norm(r)/sp.linalg.norm(b)<tol:
            break
    iter=i+1
    return [x,np.array(resvec),iter]

A=delsq(numgrid('S',102))
n=A.shape[0]
b=A@np.ones(n)
tol=1e-08
maxit=200

I=sp.sparse.eye(n)
L=ilupp.IChol0Preconditioner(A)

print('No conditioning:')
start = time.time()
trial_1=my_pcg(A,b,tol,maxit,L=I)
print('execution time: ', time.time() - start)

print('\nIC(0):')
start = time.time()
trial_2=my_pcg(A,b,tol,maxit,L=L)
print('execution time: ', time.time() - start)


L=ilupp.ichol0(A)
print('\n\nPreconditioner calculation:')
start = time.time()
K=sp.sparse.linalg.inv(L@L.T)
print('execution time: ', time.time() - start)

print('\nScipy default:')
start = time.time()
default=sp.sparse.linalg.cg(A,b,x0=np.zeros(A.shape[0]),tol=tol,maxiter=maxit,M=K)
print('execution time: ', time.time() - start)


plt.plot(trial_1[1],'r+-')
plt.plot(trial_2[1],'go-',mfc='none')
plt.ylabel('Residual norm')
plt.xlabel('Iteration number')
plt.legend(['No preconditioner','IC(0)'],loc=0)
plt.yscale('log')
plt.show()

fig, sol=plt.subplots(nrows=1,ncols=3,figsize=(20,5))
sol[0].plot(default[0]-1)
sol[0].set_title('Scipy default solution')
sol[1].plot(trial_1[0]-1)
sol[1].set_title('No preconditioner solution')
sol[2].plot(trial_2[0]-1)
sol[2].set_title('IC(0) solution')
plt.show()
