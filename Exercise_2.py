import numpy as np
import scipy as sp
from numgrid_delsq import numgrid
from numgrid_delsq import delsq
from mypcg import my_pcg
import ilupp

tol=1e-8
maxit=5000

# Define nx values
nx_values = [102, 202, 402, 802]

# Dictionary to store iterations for each nx and preconditioner
iteration_results = {}

# Iterate over nx_values to solve each system
for nx in nx_values:
    # Create the Poisson matrix A
    A_nx=delsq(numgrid('S',nx))

    # Create the RHS vector b
    n = A_nx.shape[0]
    b = np.array([np.sqrt(1/i) for i in range(1, n+1)])

    # define different preconditioner for Anx
    I=sp.sparse.eye(n)
    L0=ilupp.IChol0Preconditioner(A_nx)
    L2=ilupp.ICholTPreconditioner(A_nx, threshold=1e-2)
    L3=ilupp.ICholTPreconditioner(A_nx, threshold=1e-3)

    preconditioners = {"I": I, "L0": L0, "L2": L2, "L3": L3}

    for pre_name, pre_matrix in preconditioners.items():
        x, resvec, iter = my_pcg(A_nx, b, tol, maxit, L=pre_matrix)
        # Store iterations in the results dictionary
        iteration_results[(nx, pre_name)] = iter


# Print the results in a table format
for nx in nx_values:
    for pre_name in preconditioners.keys():
        print(f"""  
                n={nx-2},
                h={1/(nx-2)},
                Preconditioner={pre_name},
                Iterations={iteration_results[(nx, pre_name)]}""")
    
