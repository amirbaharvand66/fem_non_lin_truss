from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from inp_files.gnl_slender_truss_beam_buckl import *
from functions.ltc_funcs import *
from functions.nltc_funcs import *
from functions.gnltc_sol import *
from functions.plots import *
from functions.gnltc_funcs import * # material nonlinearity incremental methods


# import truss data
X = np.matrix(X)
IX = np.matrix(IX)
mprop = np.matrix(mprop)
loads = np.matrix(loads)
bound = np.matrix(bound)
st = np.matrix(st) # spring stiffness

# required data
neq = X.shape[0] * X.shape[1] # number of equations
ne = IX.shape[0] # number of elements
nb = bound.shape[0] # number of boundary conditions
nl = loads.shape[0] # number of point loads
nst = st.shape[0] # number of spring stiffness

# initials
p = np.zeros((neq, 1)) # point load vector
D = np.zeros((neq, 1)) # global displacement vector
dD = np.zeros((neq, 1), dtype = 'd') # infinitesimal displacement vector
R = np.zeros((neq, 1)) # internal force (residual vector)
dpR = np.zeros((neq, 1)) # (dP - R)
inc = 3 # number of increments
d_ = np.zeros((neq, inc)) # output displacement
p_ = np.zeros((neq, inc)) # output force

# Newton-Raphson
max_itr = 100 # maximum iteration in NR method
epsilon = 1e-12 # for accepting difference between the external force and residuals
d_, p_, D = gnltc_nr(X, IX, mprop, loads, bound, st, neq, ne, nl, nb, nst, p, D, dD, R, inc, d_, p_, \
             max_itr, epsilon, spring = 'off')

# adding point (0, 0)
d_ = np.insert(d_, 0 , np.zeros((1, inc)), axis = 0)
p_ = np.insert(p_, 0 , np.zeros((1, inc)), axis = 0)

# what to plot at node 81 and 83
d_p = -d_[81, :] # appending displacements for the required node
p_p = -(p_[81, :] + p_[83, :]) # appending forces for the required node
    
plt.plot(d_p, p_p, '-o', label = 'Newton-Raphson Method')
plt.legend()
plt.xlabel("Displacement [m]")
plt.ylabel("Force [N]")
plt.show()

plot_def_udef_not_fancy(IX, X, D, ne)
plt.show()