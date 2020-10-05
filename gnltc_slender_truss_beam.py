from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from inp_files.gnl_slender_truss_beam import *
from functions.ltc_funcs import *
from functions.nltc_funcs import *
from functions.gnltc_sol import *
from functions.plots import *
from functions.gnltc_funcs import * # material nonlinearity incremental methods

"""
1. First, use the 'ltc.py' to plot the truss and what appears does not show 
buckling and no critical loading by plotting force-displacement. That is due 
to the equal applied load at the right end of the truss.

2. To see the effect of critical loading available in buckling, P_cr, do the 
following: (use 'gnl_slender_truss_beam_p_cr.py' as input file)
    
    2.1. Find the effective flexural rigidity, EI, of the truss where E and I 
    are Young's modulus and second moment of area, respectively.
    
    2.2. Apply a slight vertical load (y-direction), e.g. P = 1e-4, at the end
    point of the truss. Again use 'ltc.py' to calculate the displacement, d.
    
    2.3. Use d = P * L^3 / (3 * EI) to find the flexural rigidity of the truss.
    
    2.4. Calculate P_cr = pi^2 / 4 * ( EI / L^2 )
    
3. Now to see the buckling effect of the truss we need P_cr applied at the 
right end of the truss being slightly different, e.g., P_cr + P_cr * 0.02 
and P_cr - P_cr * 0.02

"""
from inp_files.gnl_slender_truss_beam import *
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
D_n = np.zeros(inc) # final displacement vector for plotting
p_n = np.zeros(inc) # final force vector for plotting

# 1. use 'gnl_slender_truss_beam.py'
# 2. use 'gnl_slender_truss_beamp_cr.py'
from ltc import *
# 2.1 extract the maximum displacement at the right end of the truss
delta = D[81]
# 2.1 extract the corresponding resultant force
p = p[81]
L = 20
# 2.3 calculate EI
EI = p * L**3 / (3 * delta)
# 2.4 calculate P_cr
p_cr = (pi**2 * EI) / (4 * L**2)
# modify the load
loads = [[41, 1, -p_cr - p_cr  * 0.02] ,
         [42, 1, -p_cr + p_cr  * 0.02]]
# 3. run 'gnltc_slender_truss_beam_buckl.py'