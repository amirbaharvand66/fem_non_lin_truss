from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from inp_files.e03 import * # select input file
from functions.ltc_funcs import *
from functions.nltc_funcs import *
from functions.signorini_stress_strain import *
from functions.mnltc_inc_mtd import * # material nonlinearity incremental methods

# import truss data
X = np.matrix(X)
IX = np.matrix(IX)
mprop = np.matrix(mprop)
loads = np.matrix(loads)
bound =np.matrix(bound)

# required data
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
d_ = np.zeros((neq, inc)) # final displacement vector for plotting
p_ = np.zeros((neq, inc)) # final force vector for plotting


# calculating constants
c1 = 1 # signorini constant
c2 = 50 # signorini constant
c3 = 0.1 # signorini constant
c4 = 100 # signorini constant

# signorini stress strain relation
signorini_stress_strain(X, IX, mprop, loads, c1, c2, c3, c4)

# pure euler method (explicit)
d_, p_, marker, label = mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, \
                        dpR, inc, d_, p_, c1, c2, c3, c4, 'PE')

# euler method with ones step equilibrium correction (explicit)
# d_, p_, marker, label = mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD,\
#     R, dpR, inc, d_, p_, c1, c2, c3, c4, 'E1SC')


# newton-raphson method (implicit)
max_itr = 100 # maximum iteration in NR method
epsilon = 1e-12 # for accepting difference between the external force and residuals
# d_, p_, marker, label = mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, \
#                               R, dpR, inc, d_, p_, c1, c2, c3, c4, 'NR', max_itr = max_itr, epsilon = epsilon)

# modified newton-raphson method (implicit)
# d_, p_, marker, label = mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, \
#                               R, dpR, inc, d_, p_, c1, c2, c3, c4, 'MNR', max_itr = max_itr, epsilon = epsilon)


# plot displacement and force for the desire node
plt.plot(d_[4, :], p_[4, :], marker, label = label)
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [N]')
plt.legend(loc = 'lower right')
plt.show()