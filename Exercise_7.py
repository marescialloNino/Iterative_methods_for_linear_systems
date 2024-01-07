import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import ilupp
import pyamg

# Read the file and extract rows, columns, and data
# Assuming the file has three columns: row, column, value
rows, cols, data = np.loadtxt('./matrixes/mat13041.rig', unpack=True)

# Convert to zero-based indexing if necessary
rows = rows.astype(int) - 1  # Subtract 1 if the file uses 1-based indexing
cols = cols.astype(int) - 1

# Create the COO sparse matrix
A = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()

n=A.shape[0]
c=1/np.sqrt(np.arange(1,n+1))
b=A@c

tol=1e-10
itmax=550


def create_gmres_callback(A,b,resvec):
    #Create a callback function for GMRES that stores residual norms.
    beta = np.linalg.norm(b)
    def store_relative_residuals(xk):
        r = (b - A @ xk)/beta
        resvec.append(np.linalg.norm(r))
    return store_relative_residuals

x0 = np.zeros(n)

restrt_values = [10, 20, 30, 50]

# Dictionary to store solutions and residual vectors
solutions = {}
residual_vectors = {}

for i in range(len(restrt_values)):
    # Initialize residual vector list
    resvec = []
    callback = create_gmres_callback(A, b, resvec)

    # Run GMRES
    start = time.time()
    sol = pyamg.krylov.gmres(A, b, x0, callback=callback, M=ilupp.ILUTPreconditioner(A, threshold=0.01), restrt=restrt_values[i], maxiter=itmax, tol=0.000000001)
    end = time.time()

    # Store the solution and residual vector
    solutions[restrt_values[i]] = sol
    residual_vectors[restrt_values[i]] = resvec

    print(f"Time taken for gmres with restart={restrt_values[i]}: {end - start} seconds, {len(resvec)} iterations, final relative residual={resvec[-1]}")


plt.plot(residual_vectors[restrt_values[0]],'go-',mfc='none')
plt.plot(residual_vectors[restrt_values[1]],'ro-',mfc='none')
plt.plot(residual_vectors[restrt_values[2]],'bo-',mfc='none')
plt.plot(residual_vectors[restrt_values[3]],'y*-',mfc='none')
plt.ylabel('Relative residual norm')
plt.xlabel('Iteration number')
plt.legend(['restart = 10','restart = 20','restart = 30','restart = 50'],loc=0)
plt.yscale('log')
plt.show()
