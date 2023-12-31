import numpy as np
import scipy as sp
from professor import numgrid
from professor import delsq
import ilupp
import numba
from mypcg import my_pcg
import matplotlib.pyplot as plt


n=int(1e4)
diff=5
D_init=200*np.arange(1,diff+1,1)
D_final=np.repeat(1,n-diff)
D=np.concatenate((D_init,D_final))
A=sp.sparse.diags(D)

tol=1e-8
maxit=5000

np.random.seed(42)
b=np.random.rand(n)

I=sp.sparse.eye(n)

res=my_pcg(A,b,tol,maxit,L=I)

plt.plot(res[1])
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel('Residual norm')
plt.title('Semilog-y plot')
plt.legend(['Residual norm'],loc=0)
plt.show()
