from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from inp_files.e03 import * # select input file
from functions.funcs import *
from functions.plots import *
from functions.output_operate import *
from functions.signorini_stress_strain import *
from functions.mnltc import *

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
p = np.zeros((neq, 1)) # point load vector
D = np.zeros((neq, 1)) # global displacement vector
dD = np.zeros((neq, 1), dtype = 'd') # infinitesimal displacement vector
R = np.zeros((neq, 1)) # internal force (residual vector)
dpR = np.zeros((neq, 1)) # (dP - R)
inc = 20 # number of increments
D_n = np.zeros(inc) # final displacement vector for plotting
p_n = np.zeros(inc) # final force vector for plotting


# calculating constants
c1 = 1 # signorini constant
c2 = 50 # signorini constant
c3 = 0.1 # signorini constant
c4 = 100 # signorini constant

# signorini stress strain relation
signorini_stress_strain(X, IX, mprop, loads, c1, c2, c3, c4)

# pure euler method (explicit)
mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, 'PE')

# euler method with ones step equilibrium correction (explicit)
mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, 'E1SC')


# newton-raphson method (implicit)
max_itr = 100 # maximum iteration in NR method
epsilon = 1e-12 # for accepting difference between the external force and residuals
mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, 'NR', max_itr = max_itr, epsilon = epsilon)

# modified newton-raphson method (implicit)
mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, 'MNR', max_itr = max_itr, epsilon = epsilon)


# legends and labels for plots
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [N]')
plt.legend(loc = 'lower right')
plt.show











    

