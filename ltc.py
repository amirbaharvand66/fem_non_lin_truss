from math import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

from inp_files.e02 import * # select input file
from functions.ltc_funcs import *
from functions.plots import *
from functions.out_opr import *
from functions.num_sol import *

# import truss data
X = np.matrix(X)
IX = np.matrix(IX)
mprop = np.matrix(mprop)
loads = np.matrix(loads)
bound =np.matrix(bound)

# necessary data
neq = X.shape[0] * X.shape[1] # number of equations
ne = IX.shape[0] # number of elements
nb = bound.shape[0] # number of boundary conditions
nl = loads.shape[0] # number of point loads

# initials
K = np.zeros((neq, neq)) # global stiffness matrix
p = np.zeros((neq, 1)) # point load vector
D = np.zeros((neq, 1)) # global displacement vector
le = np.zeros((ne, 1)) # element strain
ls = np.zeros((ne, 1)) # element stress
N = np.zeros((ne, 1)) # element force
R = np.zeros((neq, 1)) # internal force (residual vector)
RF = np.zeros((neq, 1)) # reaction forces at supports

# computing K
for n in range(ne):
    [L0, edof, B0] = strn_displ_vec(IX, X, n)
    c = IX[n, 2]
    E = mprop[c - 1, 0]
    A = mprop[c - 1, 1]
    ke = A * E * L0 * np.outer(B0, np.transpose(B0)) # local stiffness matrix
    # K assembly
    for ii in range(edof.shape[1]):
        for jj in range(edof.shape[1]):
            K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]] 


# applying the boundary conditions on K
K = bnd_cnd_K(bound, nb, K)


# build-up point load vector
p = build_load_vec(loads, nl, p)

# computing D
# for K matrices close to singular or ill-conditioned, 
# pseudo-inverse of K based on singular value decomposition (SVD)
D = np.linalg.pinv(K).dot(p)

# or you can choose myh explicit Gaussian elimination function
# D = gauss_elm(K, p, D)

# computing R
for n in range(ne):
    [L0, edof, B0] = strn_displ_vec(IX, X, n)
    c = IX[n, 2]
    E = mprop[c - 1, 0]
    A = mprop[c - 1, 1]
    d = D[edof] #local displacement vector
    le[n, 0] = np.matmul(np.transpose(B0), d)
    ls[n, 0] = E * le[n, 0]
    N[n, 0] = A * ls[n, 0]
    # computing residual vector 
    re = A * E * L0 * np.matmul(np.matmul(B0, np.transpose(B0)), d) # local residual
    R[edof, 0] = re + R[edof, 0]

# computing reaction forces at supports
RF = R - p


plot_def_udef(IX, X, D, ls, ne)
plt.axis('square')
plt.show()


# write_output(ne, neq, D, RF, le, ls, N)
# show_output(ne, neq, D, RF, le, ls, N)








    

