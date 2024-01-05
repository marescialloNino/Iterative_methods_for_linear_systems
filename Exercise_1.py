import numpy as np
import scipy as sp
from numgrid_delsq import numgrid
from numgrid_delsq import delsq

from mypcg import my_pcg
import ilupp
import matplotlib.pyplot as plt
import time

# SPD matrix arising from the finite difference 2D discretization of the Laplacian
A=delsq(numgrid('S',102))
n=A.shape[0]
b=A@np.ones(n)
tol=1e-08
maxit=200


def create_callback(A, b, residuals):
    """ Factory function to create a callback function for storing residuals at 
     each iteration for scipy's cg methods which does not return the residuals vector. """
    def callback(xk):
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))
    return callback

#identity matrix
I=sp.sparse.eye(n)

start = time.time()
trial_1=my_pcg(A,b,tol,maxit,L=I)
print('No conditioning my_pcg execution time: ', time.time() - start)

# Incomplete Cholesky factor (lower triangular matrix such that A = L*L') 
L=ilupp.IChol0Preconditioner(A)

start = time.time()
trial_2=my_pcg(A,b,tol,maxit,L=L)
print('my_pcg with incomplete cholesky preconditioning execution time: ', time.time() - start)

residuals_1 = []
callback_1 = create_callback(A, b, residuals_1)

start = time.time()
default=sp.sparse.linalg.cg(A,b,x0=np.zeros(A.shape[0]),tol=tol,maxiter=maxit, callback=callback_1)
print('Scipy cg method without preconditioning execution time: ', time.time() - start)

L=ilupp.ichol0(A)
start = time.time()
M=sp.sparse.linalg.inv(L@L.T)
print('Preconditioner M=L*L\' calculation execution time: ', time.time() - start)

residuals_2 = []
callback_2 = create_callback(A, b, residuals_2)

start = time.time()
default_cholesky=sp.sparse.linalg.cg(A,b,x0=np.zeros(A.shape[0]),tol=tol,maxiter=maxit,M=M,callback=callback_2)
print('Scipy cg method with cholesky IC(0) execution time: ', time.time() - start)


plt.plot(trial_1[1],'r*-')
plt.plot(residuals_1,'yo-',mfc='none')
plt.plot(trial_2[1],'b*-')
plt.plot(residuals_2,'go-',mfc='none')
plt.ylabel('Residual norm')
plt.xlabel('Iteration number')
plt.legend(['mypcg no preconditioner','scipy no preconditioner','mypcg IC(0)','scipy IC(0)'],loc=0)
plt.yscale('log')
plt.show()

fig, sol=plt.subplots(nrows=1,ncols=3,figsize=(20,5))
sol[0].plot(default[0]-1,'g')
sol[0].set_title('Scipy default solution')
sol[1].plot(trial_1[0]-1,'g')
sol[1].set_title('No preconditioner solution')
sol[2].plot(trial_2[0]-1,'g')
sol[2].set_title('IC(0) solution')
plt.show()