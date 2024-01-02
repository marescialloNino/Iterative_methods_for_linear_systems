import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Read the file and extract rows, columns, and data
# Assuming the file has three columns: row, column, value
rows, cols, data = np.loadtxt('./mat13041.rig', unpack=True)

# Convert to zero-based indexing if necessary
rows = rows.astype(int) - 1  # Subtract 1 if the file uses 1-based indexing
cols = cols.astype(int) - 1

# Create the COO sparse matrix
A = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()

def my_GMRES(A,b,tol,maxit,x0,L,U):
    n=A.shape[0]
    w=b-(A@x0)
    r0=sp.sparse.linalg.spsolve_triangular(L,w,lower=True)
    
    w=sp.sparse.linalg.spsolve_triangular(L,r0,lower=True)
    tol_vec=sp.sparse.linalg.spsolve_triangular(U,w,lower=False)
    
    tol_check=tol*sp.linalg.norm(tol_vec)
    
    beta=sp.linalg.norm(r0)
    v=[]
    
    v.append(r0/beta)
    
    h=np.zeros((maxit+1,maxit))
    resvec=[]
    for k in range(0,maxit,1):
        w=sp.sparse.linalg.spsolve_triangular(U,v[k],lower=False)
        v.append(sp.sparse.linalg.spsolve_triangular(L,A@w,lower=True))
        
        for j in range(0,k+1,1):
            h[j,k]=v[k+1]@v[j]
            v[k+1]=v[k+1]-(h[j,k]*v[j])
        
        h[k+1,k]=np.linalg.norm(v[k+1])
        v[k+1]=v[k+1]/np.linalg.norm(v[k+1])

        H=h[:k+2,:k+1] #H k+1,k
        
        Q=np.linalg.qr(H,mode='complete')[0]
        
        g=beta*(Q[0,:])
        
        resvec.append(np.abs(g[-1]))
        flag=0
        R=np.linalg.qr(H,mode='complete')[1]
        V=np.array(v[:-1]).T
        #print(R[:-1,:].shape)
        y=sp.linalg.solve_triangular(R[:-1,:], g[:-1], lower=False, check_finite=False)
        z=V@y
        x=x0+sp.sparse.linalg.spsolve_triangular(U,z,lower=False)
        
        w=sp.sparse.linalg.spsolve_triangular(L,b-(A@x),lower=True)
        temp_tol=sp.sparse.linalg.spsolve_triangular(U,w,lower=False)
        
        if sp.linalg.norm(temp_tol)<tol_check:
            flag=-1
            break
    return [x,k+1,resvec,flag]

np.random.seed(52)

n=A.shape[0]
c=1/np.sqrt(np.arange(1,n+1))
b=A@c

I=sp.sparse.eye(n)
tol=1e-10
itmax=550

sol=my_GMRES(A,b,tol,itmax,np.ones(n),I,I)
err=sol[2]
print(sol[1],sol[3],err[-1])

plt.plot(sol[2],'go-',mfc='none')
plt.ylabel('Absolute residual')
plt.xlabel('Iteration number')
plt.legend(['ILU(0)'],loc=0)
plt.yscale('log')
plt.grid()
plt.show()

plt.plot(sol[0]-c)
plt.show()
