import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import ilupp
import pyamg



A = sp.io.loadmat('./matrixes/ML.mat')['ML'].tocsr()

n = A.shape[0]

def create_gmres_callback(A,b,resvec):
    #Create a callback function for GMRES that stores residual norms.
    def callback(xk):
        r = b - A @ xk
        resvec.append(np.linalg.norm(r))
    return callback


exact_sol = np.ones(n)

b = A@exact_sol

# Initial guess
x0 = np.zeros(A.shape[0])

# Define droptol values
droptol_values = [2e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5]

# Dictionary to store solutions and residual vectors
solutions = {}
residual_vectors = {}

# Iterate over droptol values
for droptol in droptol_values:

    
    # Initialize residual vector list
    resvec = []
    callback = create_gmres_callback(A, b, resvec)

    # Start measuring time for preconditioner
    start_prec = time.time()
    M = ilupp.ILUTPreconditioner(A, threshold=droptol)
    end_prec = time.time()

    L, U=ilupp.ilut(A, threshold=droptol)
    density = (L.nnz + U.nnz - n)/A.nnz

    # Run GMRES
    start_sol = time.time()
    sol = pyamg.krylov.gmres(A, b, x0, callback=callback, M=M, restrt=50, maxiter=550, tol=0.000000000001)
    end_sol = time.time()

    # Store the solution and residual vector
    solutions[droptol] = sol
    residual_vectors[droptol] = resvec

    # Calculate times
    tprec = end_prec - start_prec
    tsol = end_sol - start_sol
    
    # Print or log the time taken and iterations
    print(f"Droptol={droptol:.1e}, Time taken: Preconditioner={tprec:.2f} s, GMRES={tsol:.2f} s, Total={tprec+tsol:.2f} s, Iterations={len(resvec)}, Density={density}")

plt.plot(residual_vectors[droptol_values[0]],'g*-')
plt.plot(residual_vectors[droptol_values[1]],'r*-')
plt.plot(residual_vectors[droptol_values[2]],'b*-')
plt.plot(residual_vectors[droptol_values[3]],'y*-')
plt.plot(residual_vectors[droptol_values[4]],'pink*-')
plt.plot(residual_vectors[droptol_values[5]],'orange*-')
plt.ylabel('Absolute residual')
plt.xlabel('Iteration number')
plt.legend(['droptol = 2e-2','droptol = 1e-2','droptol = 3e-3','droptol = 1e-3','droptol = 1e-4','droptol = 1e-5'],loc=0)
plt.yscale('log')
plt.grid()
plt.show()