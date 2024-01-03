import numpy as np
import scipy as sp
from mypcg import my_pcg
import matplotlib.pyplot as plt

tol=1e-8
maxit=5000

# Initialize the matrix size
n = int(1e4)

# Define the diagonal elements
diagonal = [200 * (i + 1) if i <= 5 else 1 for i in range(n)]

# Create the sparse diagonal matrix
A1 = sp.sparse.diags(diagonal, 0)

# random rhs 
b = np.random.rand(n)

# identity matrix (no preconditioning)
I = sp.sparse.eye(n)

# solve liner system
x, resvec, iter = my_pcg(A1,b,tol,maxit,L=I)

plt.semilogy(range(iter+1), resvec, 'o-')
plt.xlabel('Iteration number')
plt.ylabel('Residual norm')
plt.show()

