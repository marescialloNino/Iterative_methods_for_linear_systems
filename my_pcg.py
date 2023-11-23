import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse import diags, kron
from scipy.sparse.linalg import LinearOperator, cg, splu

def discrete_laplacian(n):
    """
    Create a discrete Laplacian Matrix for a square grid of size n x n
    utilizing kroenecker product.
    Returns a 2D laplacian in sparse matrix form.
    """
    # Create a 1D discrete Laplacian
    laplacian_1d = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    # Create a 2D discrete Laplacian using the Kronecker product
    I_n = diags([1], [0], shape=(n, n))
    laplacian_2d = kron(laplacian_1d, I_n) + kron(I_n, laplacian_1d)

    return laplacian_2d


def mypcg(A, b, tol, maxit, L):
    x = np.zeros_like(b)  # initial guess
    r = b - A.dot(x)
    resvec = [np.linalg.norm(r)]

    # Cholesky solve for the preconditioning step
    y = cho_solve((L, True), r)
    p = z = y

    for iter in range(maxit):
        Ap = A.dot(p)
        r_dot_z = np.dot(r, z)
        alpha = r_dot_z / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        # Check for convergence
        res_norm = np.linalg.norm(r)
        resvec.append(res_norm)
        if res_norm < tol:
            break

        # Preconditioning step
        y = cho_solve((L, True), r)
        beta = np.dot(r, y) / r_dot_z
        p = y + beta * p
        z = y

    return x, resvec, iter


# Create the 2D Laplacian matrix
A = discrete_laplacian(100)

# Define the right-hand side b
b = A.dot(np.ones(A.shape[0]))

# Tolerance and maximum iterations
tol = 1e-8
maxit = 50

# Solve without preconditioning
x_no_precond, info_no_precond = cg(A, b, tol=tol, maxiter=maxit)

# Incomplete Cholesky Preconditioner
M2 = splu(A).L
M_x = lambda x: splu(A).solve(x)
M = LinearOperator((A.shape[0], A.shape[0]), M_x)

# Solve with IC(0) preconditioning
x_precond, info_precond = cg(A, b, M=M, tol=tol, maxiter=maxit)

print(f"Solution without preconditioner: {x_no_precond[:10]}")
print(f"Solution with IC(0) preconditioner: {x_precond[:10]}")
