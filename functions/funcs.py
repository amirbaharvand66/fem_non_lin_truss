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
