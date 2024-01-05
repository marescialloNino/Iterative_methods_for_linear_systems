import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from gmresnew import my_gmres
import time

# Read the file and extract rows, columns, and data
# Assuming the file has three columns: row, column, value
rows, cols, data = np.loadtxt('./matrixes/mat13041.rig', unpack=True)

# Convert to zero-based indexing if necessary
rows = rows.astype(int) - 1  # Subtract 1 if the file uses 1-based indexing
cols = cols.astype(int) - 1

# Create the COO sparse matrix
A = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()

def my_gmres(A, b, tol, maxit, x0):
    """
    Implement GMRES method.

    Parameters
    ----------
    A : ndarray or csr_matrix
        Coefficient matrix.
    b : ndarray
        Right-hand side vector.
    tol : float
        Tolerance for convergence.
    maxit : int
        Maximum number of iterations.
    x0 : ndarray
        Initial guess vector.

    Returns
    -------
    x : ndarray
        Solution vector.
    iter : int
        Number of iterations employed.
    resvec : ndarray
        Vector with the norm of the residuals.
    flag : int
        Flag indicating termination status (0 for canonical termination, -1 for breakdown).
    """
    
    r0 = b - A @ x0
    beta = sp.linalg.norm(r0)
    resvec = [beta]
    q = np.zeros((A.shape[0], maxit + 1), dtype=A.dtype)
    q[:, 0] = r0 / beta
    h = np.zeros((maxit + 1, maxit), dtype=A.dtype)
    flag = 0

    for k in range(maxit):
        # Arnoldi process
        v = A @ q[:, k]
        for j in range(k + 1):
            h[j, k] = np.dot(q[:, j].conj().T, v)
            v -= h[j, k] * q[:, j]
        hnorm = sp.linalg.norm(v)
        if hnorm <= tol:
            flag = -1
            break
        h[k + 1, k] = hnorm
        q[:, k + 1] = v / hnorm

        # Solve least squares problem and update residual
        y = np.linalg.lstsq(h[:k + 2, :k + 1], beta * np.eye(k + 2, 1)[:, 0], rcond=None)[0]
        r = sp.linalg.norm(b - A @ (x0 + q[:, :k + 1] @ y))
        resvec.append(r)

        if r < tol:
            break

    x = x0 + q[:, :k + 1] @ y
    iter = k + 1
    return x, iter, np.array(resvec), flag


n=A.shape[0]
c=1/np.sqrt(np.arange(1,n+1))
b=A@c

I=sp.sparse.eye(n)
tol=1e-10
itmax=550

start = time.time()
sol = my_gmres(A,b,tol,itmax,np.ones(n))
end = time.time()
print(end - start)

err=sol[2]
print(sol[1],sol[3],err[-1])

plt.plot(sol[2],'go-',mfc='none')
plt.ylabel('Absolute residual')
plt.xlabel('Iteration number')
plt.legend(['ILU(0)'],loc=0)
plt.yscale('log')
plt.grid()
plt.show()

