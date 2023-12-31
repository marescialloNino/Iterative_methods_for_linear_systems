import numpy as np
from scipy import sparse

def numgrid(R,n):
    """
    NUMGRID Number the grid points in a two dimensional region.
    """
    x = np.ones((n,1))*np.linspace(-1,1,n)
    y = np.flipud(x.T)
    if R == 'S':
        G = (x > -1) & (x < 1) & (y > -1) & (y < 1)
    elif R == 'L':
        G = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ( (x > 0) | (y > 0))
    elif R == 'C':
        G = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ((x+1)**2+(y+1)**2 > 1)
    elif R == 'D':
        G = x**2 + y**2 < 1
    elif R == 'A':
        G = ( x**2 + y**2 < 1) & ( x**2 + y**2 > 1/3)
    elif R == 'H':
        RHO = .75
        SIGMA = .75
        G = (x**2+y**2)*(x**2+y**2-SIGMA*y) < RHO*x**2
    elif R == 'B':
        t = np.arctan2(y,x)
        r = np.sqrt(x**2 + y**2)
        G = (r >= np.sin(2*t) + .2*np.sin(8*t)) & (x > -1) & (x < 1) & (y > -1) & (y < 1)
    else:
        print('numgrid:InvalidRegionType')
    G = np.where(G,1,0) # boolean to integer
    k = np.where(G.T)
    kT = (k[1],k[0])
    G[kT] = 1 + np.arange( k[0].shape[0] )
    return G

def delsq(G):
    """
    DELSQ  Construct five-point finite difference Laplacian.
    delsq(G) is the sparse form of the two-dimensional,
    5-point discrete negative Laplacian on the grid G.
    """
    [m,n] = G.shape
    # Indices of interior points
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = p.shape[0]
    # Connect interior points to themselves with 4's.
    i = G1[p]-1
    j = G1[p]-1
    s = 4*np.ones(p.shape)

    # for k = north, east, south, west
    for k in [-1, m, 1, -m]:
       # Possible neighbors in k-th direction
       Q = G1[p+k]
       # Index of points with interior neighbors
       q = np.where(Q)[0]
       # Connect interior points to neighbors with -1's.
       i = np.concatenate([i, G1[p[q]]-1])
       j = np.concatenate([j,Q[q]-1])
       s = np.concatenate([s,-np.ones(q.shape)])
    # sparse matrix with 5 diagonals
    return sparse.csr_matrix((s,(i,j)),(N,N))

def test_numgrid(n):
    print("numgrid('S',n)")
    print(numgrid('S',n))
    print("numgrid('L',n)")
    print(numgrid('L',n))
    print("numgrid('C',n)")
    print(numgrid('C',n))
    print("numgrid('D',n)")
    print(numgrid('D',n))
    print("numgrid('A',n)")
    print(numgrid('A',n))
    print("numgrid('H',n)")
    print(numgrid('H',n))
    print("numgrid('B',n)")
    print(numgrid('B',n))

def test_delsq(n):
    lapl = delsq(numgrid('S',n))
    print(lapl.todense())

def application(n):
    import matplotlib.pyplot as plt
    from scipy.sparse import linalg
    G = numgrid('B',n)
    D = delsq(G)
    N = D.shape[0]
    rhs = np.ones((N,1))
    u = linalg.spsolve(D, rhs) # vector solution
    U = np.zeros(G.shape) # map u back onto G space
    U[G>0] = u[G[G>0]-1]
    plt.contour(U)
    plt.show()

def main():
    n = 6
    test_numgrid(n)
    test_delsq(n)
    application(320)

if __name__ == '__main__':
    main()
