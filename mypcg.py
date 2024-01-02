import numpy as np
import scipy as sp

def my_pcg(A, b, tol, maxit, L):
    """ Conjugate gradient method with Cholesky preconditioning implementation.
     INPUTS: 
        A = SPD matrix
        b = right hand side of linear system
        tol = tolerance (exit test performed on the relative residual)
        maxit = max number of iterations
        L = Cholesky preconditioner 
     OUTPUTS: 
        x = solutions vector
        resvec =  vector with the norm of the absolute residual at each iteration
        iter = number of iterations employed """

    x = L@b
    # initial residual
    r = b - A@x 
    #apply preconditioner
    z = L@r
    p = z.copy()
    
    resvec=[np.linalg.norm(r)]
    
    for i in range(maxit):
        
        Ap = A@p
        
        alpha = (r@z)/(p@Ap)
        x += (alpha * p)
        # residual update 
        r -= alpha * Ap
        resnorm = np.linalg.norm(r)
        resvec.append(resnorm)

        if resnorm / np.linalg.norm(b) < tol:
            break
        
        z = L@r
        
        b = -(z@Ap)/(p@Ap)
        p = z + b * p
        
    return x, np.array(resvec), i + 1