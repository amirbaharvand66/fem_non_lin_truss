from math import *
import numpy as np


def strn_displ_vec(IX, X, n:int):
    """
    calculating strain-displacement vector (B0)
    
    input(s):
    IX : nodal coordinates
    X : topology matrix (element connection matrix)
    n : element number
    
    output(s):
    L0 : initial elemnt length
    edof : element defree og freedom vector
    B0 : strain-displacement vector

    originally coded by Amir Baharvand (AB) (09-20)
    """
    
    a = IX[n, 0]
    b = IX[n, 1]
    dx = X[IX[n, 1] - 1, 0] - X[IX[n, 0] - 1, 0] # dx = xj - xi
    dy = X[IX[n, 1] - 1, 1] - X[IX[n, 0] - 1, 1] # dy = yj - yi
    L0 = sqrt(dx**2 + dy**2)
    # element degree of freedom (dof)
    # Originally edof =  np.matrix([ [2 * a - 1, 2 * a , 2 * b - 1, 2 * b] ]) 
    # but Python starts from 0
    edof =  np.matrix([ [2 * a - 2, 2 * a - 1, 2 * b - 2, 2 * b - 1] ]) 
    B0 = (1 / L0 **2) * np.matrix([[-dx], [-dy], [dx], [dy]])

    return L0, edof, B0


def bnd_cnd_K(bound, nb, K):
    """
    apply the boundary condition on K
    global stiffness matrix (linear)
    or 
    tangential stiffness matrix (nonlinear)

    inputs(s):
    bound : boundary condition(s) from input file
    nb : number of boundary conditions
    K : global / tangential stiffness matrix 

    originally coded by Amir Baharvand (AB) (09-20)
    """
    for ii in range(nb):
        a = int(bound[ii, 0])
        b = int(bound[ii, 1])
        if (b % 2) != 0:
            K[2 * a - 2, :] = 0
            K[:, 2 * a - 2] = 0
            K[2 * a - 2, 2 * a - 2] = 1
        else: # if (b % 2) == 0
            K[2 * a - 1, :] = 0
            K[:, 2 * a - 1] = 0
            K[2 * a - 1, 2 * a - 1] = 1

    return K


def bnd_cnd_rsdl(bound, nb, Residual):
    """
    apply boundary condition on
    internal force (residual vector) (R)
    or
    dpR = dP_n - R_(n-1) in Euler method with ones step equilibrium correction
    (difference between external load and residual)

    input(s):
    bound : boundary condition(s) from input file
    nb : number of boundary conditions
    Residual : type (R / dpR)

    originally coded by Amir Baharvand (AB) (09-20)
    """
    for ii in range(nb):
        a = int(bound[ii, 0])
        b = int(bound[ii, 1])
        if (b % 2) != 0:
            Residual[2 * a - 2, :] = 0
        else: # if (b % 2) == 0
            Residual[2 * a - 1, :] = 0

    return Residual


def build_load_vec(loads, nl, p):
    """
    build load vector

    input(s):
    loads : load(s) from input file
    nl : number of point loads
    p : point load vector

    originally coded by Amir Baharvand (AB) (09-20)
    """
    for ii in range(nl):
        a = int(loads[ii, 0])
        b = int(loads[ii, 1])
        c = loads[ii, 2]
        if (b % 2) != 0:
            p[2 * a - 2] = c
        else:
            p[2 * a - 1] = c

    return p