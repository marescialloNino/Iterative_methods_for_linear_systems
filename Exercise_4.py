import numpy as np
import scipy as sp
import ilupp
from mypcg import my_pcg
import matplotlib.pyplot as plt

tol = 1e-9
maxit = 5000

# load SPD matrix created with matlab gallery('whaten',100,100) command
A = sp.io.loadmat('A_exercise4.mat')['A'].tocsr()
n = A.shape[0]

# random solution
x0 = np.random.rand(n)
# rhs relative to the random solution
b = A@x0

# define preconditioners
I=sp.sparse.eye(n)
Jac=sp.sparse.diags(1/A.diagonal())
Cho=ilupp.IChol0Preconditioner(A)

# solve linear system
x_cg, res_cg, iter_cg = my_pcg(A, b, tol, maxit, L=I)
x_jac, res_jac, iter_jac = my_pcg(A, b, tol, maxit, L=Jac)
x_cho, res_cho, iter_cho = my_pcg(A, b, tol, maxit, L=Cho)


plt.figure(figsize=(10, 6))

# Plot the residuals for each method
plt.semilogy(range(iter_cg + 1), res_cg, label='CG with no preconditioning',marker='*')
plt.semilogy(range(iter_jac + 1), res_jac, label='CG with Jacobi preconditioner',marker='*')
plt.semilogy(range(iter_cho + 1), res_cho, label='CG with IC(0) preconditioner',marker='*')

# Add labels and legend
plt.xlabel('Iteration Number')
plt.ylabel('Residual Norm')
plt.title('Convergence of PCG with Different Preconditioners')
plt.legend()

plt.show()