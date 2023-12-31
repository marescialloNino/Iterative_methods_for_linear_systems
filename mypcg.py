import numpy as np
import scipy as sp

def my_pcg(A, b, tol, maxit, L):
    x=L@b
    r=b-(A@x)
    p=L@r
    
    resvec=[]
    
    for i in range(0,maxit,1):
        c=p@A@p
        
        a=(r@p)/c
        x=x+(a*p)
        r=r-a*(A@p)

        v=L@r
        
        b=-(v@A@p)/c
        p=v+b*p
        
        resvec.append(sp.linalg.norm(r))
        if sp.linalg.norm(r)/sp.linalg.norm(b)<tol:
            break
    iter=i+1
    return [x,np.array(resvec),iter]