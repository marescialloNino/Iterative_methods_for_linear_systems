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

    x=L@b
    # initial residual
    r=b-(A@x)
    p=L@r
    
    resvec=[]
    
    for i in range(0,maxit,1):
        
        c=p@A@p
        
        a=(r@p)/c
        x=x+(a*p)
        # residual update 
        r=r-a*(A@p)

        v=L@r
        
        b=-(v@A@p)/c
        p=v+b*p
        
        resvec.append(sp.linalg.norm(r))
        if sp.linalg.norm(r)/sp.linalg.norm(b)<tol:
            break
    iter=i+1
    return [x,np.array(resvec),iter]