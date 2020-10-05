from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from inp_files.gnl_e04 import *
from functions.ltc_funcs import *
from functions.nltc_funcs import *
from functions.gnltc_sol import *
from functions.plots import *
from functions.gnltc_funcs import * # material nonlinearity incremental methods

"""
In this script, due to geometry nonlinearity there are some notes required to be read in advance.

1. For geometry nonlinearity, strain-displacements depend on the displacement vector.

2. To achieve this, Green-Lagrange strain is utilized which removes the peculiarity due to the 
existence of rotation in Cauchy’s strain. Cauchy’s strain is suitable for small displacements 
and cannot handle large rotations as its accuracy decreases as the rotation get larger. 
e = 1 / 2 * (F + transpose(F)) – I (includes both rotation and transition)
where, 
e : Cauchy’s strain
F : deformation gradient (I + dU / dx) where
	I : identity matrix
	U : displacemt vector
	x : spatial coordinate system (deformed state)

Green-Lagrange strain removes the effect of rotation where 
E = 1 / 2 * (transpose(F) * F - I) where E is Green-Lagrange strain

3. Here, I use a total Lagrangian approach coupled with a Newton-Raphson method. Since the d
eformed state is not predictable, the total Lagrangian is quite useful.

4. Due to the total Lagrangian approach, the stress state is determined using the force 
equilibrium of the reference and deformed state (2nd Piola-Kirchhoff stress). The idea 
of 1st and 2nd  Piola-Kirchhoff stresses are to compute the stress in the deformed state 
with regard to the deformed state. The 1st Piola-Kirchhoff stress is NOT symmetric while 
the 2nd Piola-Kirchhoff stress is symmetric and computationally less expensive and easier 
to manipulate.

5. Assumptions:
    5.1. Large displacements but small displacement gradients, i.e, small strains
    5.2. Constant material properties
"""

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
inc = 20 # number of increments
d_ = np.zeros((neq, inc)) # output displacement
p_ = np.zeros((neq, inc)) # output force

# compare Newton-Raphson with analytical solution (Krenk 1993) 
spring = krenk_1993(mprop, inc) #non-modified solution
# spring = krenk_1993(mprop, inc, spring = 'on') # modified solution with spring stiffness

# Newton-Raphson
max_itr = 100 # maximum iteration in NR method
epsilon = 1e-12 # for accepting difference between the external force and residuals
d_, p_, D = gnltc_nr(X, IX, mprop, loads, bound, st, neq, ne, nl, nb, nst, p, D, dD, R, inc, d_, p_, \
             max_itr, epsilon, spring = spring)

# adding point (0, 0)
d_ = np.insert(d_, 0 , np.zeros((1, neq)), axis = 1)
p_ = np.insert(p_, 0 , np.zeros((1, neq)), axis = 1)

# what to plot at node 81 and 83
d_p = d_[3, :] # appending displacements for the required node
p_p = p_[3, :] # appending forces for the required node
    
plt.plot(d_p, p_p, '-o', label = 'Newton-Raphson Method')
plt.legend()
plt.xlabel("Displacement [m]")
plt.ylabel("Force [N]")
plt.show()

plot_def_udef_not_fancy(IX, X, D, ne)
plt.show()