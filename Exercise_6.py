import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import ilupp
import pyamg

def my_prec_gmres(A, b, tol, maxit, x0, L, U):
    # Initial residual computation
    initial_residual = b - (A @ x0)
    # Solve lower triangular system
    lower_step = sp.sparse.linalg.spsolve_triangular(L, initial_residual, lower=True)
    
    # Solve for the tolerance vector
    upper_step = sp.sparse.linalg.spsolve_triangular(L, lower_step, lower=True)
    tol_vector = sp.sparse.linalg.spsolve_triangular(U, upper_step, lower=False)
    
    # Compute tolerance check value
    tol_limit = tol * sp.linalg.norm(tol_vector)
    
    # Initialize beta and basis vectors
    beta = sp.linalg.norm(lower_step)
    basis_vectors = []
    basis_vectors.append(lower_step / beta)
    
    # Initialize H matrix and residual vector
    H_matrix = np.zeros((maxit + 1, maxit))
    residual_vector = []
    for k in range(maxit):
        # Perform iteration steps
        upper_solve = sp.sparse.linalg.spsolve_triangular(U, basis_vectors[k], lower=False)
        basis_vectors.append(sp.sparse.linalg.spsolve_triangular(L, A @ upper_solve, lower=True))
        
        # Orthogonalize the basis vectors
        for j in range(k + 1):
            H_matrix[j, k] = np.dot(basis_vectors[k + 1], basis_vectors[j])
            basis_vectors[k + 1] -= H_matrix[j, k] * basis_vectors[j]
        
        # Normalize the next basis vector
        H_matrix[k + 1, k] = np.linalg.norm(basis_vectors[k + 1])
        basis_vectors[k + 1] /= H_matrix[k + 1, k]

        # Perform QR decomposition
        Q, R = np.linalg.qr(H_matrix[:k + 2, :k + 1], mode='complete')
        g = beta * Q[0, :]

        # Update residual vector
        residual_vector.append(np.abs(g[-1]))
        iteration_flag = 0

        # Solve for y and update x
        y = sp.linalg.solve_triangular(R[:-1, :], g[:-1], lower=False, check_finite=False)
        z = np.array(basis_vectors[:-1]).T @ y
        x = x0 + sp.sparse.linalg.spsolve_triangular(U, z, lower=False)
        
        # Check for convergence
        new_residual = sp.sparse.linalg.spsolve_triangular(L, b - (A @ x), lower=True)
        new_tol_vector = sp.sparse.linalg.spsolve_triangular(U, new_residual, lower=False)
        if sp.linalg.norm(new_tol_vector) < tol_limit:
            iteration_flag = -1
            break

    return [x, k + 1, residual_vector, iteration_flag]


# Read the file and extract rows, columns, and data
# Assuming the file has three columns: row, column, value
rows, cols, data = np.loadtxt('./matrixes/mat13041.rig', unpack=True)

# Convert to zero-based indexing if necessary
rows = rows.astype(int) - 1  # Subtract 1 if the file uses 1-based indexing
cols = cols.astype(int) - 1

# Create the COO sparse matrix
A = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()

# Compute the ILU decomposition
L, U = ilupp.ilut(A,threshold=0.1 )

n=A.shape[0]
c=1/np.sqrt(np.arange(1,n+1))
b=A@c
beta = np.linalg.norm(b)

tol=1e-10
itmax=550

start = time.time()
sol=my_prec_gmres(A,b,tol,itmax,np.zeros(n),L,U)
end = time.time()
print("my_prec_gmres cpu time: ",end - start)
print("my_prec_gmres iterations: ",sol[1])
print("my_prec_gmres final absolute residual: ",sol[2][-1])
print("my_prec_gmres final relative residual: ",sol[2][-1]/beta)
print("my_prec_gmres 'true' final residual: ",sp.linalg.norm(b-A@sol[0]))


def create_gmres_callback(A,b,resvec):
    #Create a callback function for GMRES that stores residual norms.
    beta = np.linalg.norm(b)
    def store_relative_residuals(xk):
        r = (b - A @ xk)/beta
        resvec.append(np.linalg.norm(r))
    return store_relative_residuals

resvec = []  # List to store residual norms
callback = create_gmres_callback(A,b,resvec)

# initial guess
x0 = np.zeros(n)

start = time.time()
solamg = pyamg.krylov.gmres(A,b,x0,callback=callback,M=ilupp.ILUTPreconditioner(A, threshold=0.1),maxiter=itmax,tol=0.000000001)
end = time.time()
print("pyAMG gmres cpu time: ",end - start)
print("pyAMG gmres iterations: ",len(resvec))
print("pyAMG gmres final relative residual: ",resvec[-1])


plt.plot(sol[2]/beta,'*-',mfc='none')
plt.plot(resvec,'*-',mfc='none')
plt.ylabel('Relative residual')
plt.xlabel('Iteration number')
plt.legend(['my_prec_gmres ILU(0.1)','pyamg ILU(0.1)'],loc=0)
plt.yscale('log')
plt.show()
