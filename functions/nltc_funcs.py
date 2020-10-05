from math import *
import numpy as np
from functions.ltc_funcs import *

def mnl_K(IX, X, mprop, ne, D, K, c1, c2, c3, c4):
    """
    material nonlinearity tangential stiffness matrix (K)

    input(s):
    X : topology matrix (element connection matrix)
    IX : nodal coordinates
    mprop : material properties
    ne = number of elements
    D : global displacement vector
    K : initial tangential stiffness matrix
    c1..c4 : signorini constant

    originally coded by Amir Baharvand (AB) (09-20)
    """
    for n in range(ne):
        [L0, edof, B0] = strn_displ_vec(IX, X, n)
        c = IX[n, 2]
        d = D[edof] #local displacement vector
        le = np.matmul(np.transpose(B0), d)
        lambDa = 1 + c4 * le
        Et = c4 * ( c1 * (1 + 2 * lambDa**(-3)) + 3 * c2 * lambDa**(-4) + \
                    3 * c3 * (-1 + lambDa**(2) - 2 * lambDa**(-3) + 2 * lambDa**(-4)) )    
        A = mprop[c - 1, 1]
        ke = A * Et[0, 0] * L0 * np.outer(B0, np.transpose(B0)) # local stiffness matrix
        # K assembly
        for ii in range(edof.shape[1]):
            for jj in range(edof.shape[1]):
                K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]] 
        
    return K


def mnl_int_force(IX, X, mprop, ne, D, R_int, c1, c2, c3, c4):
    """
    material nonlinearity internal force (R_int) for
    computing residual vector (R)
    R = R_int - R_ext

    input(s):
    X : topology matrix (element connection matrix)
    IX : nodal coordinates
    mprop : material properties
    ne = number of elements
    D : global displacement vector
    R_int : internal force vector
    c1..c4 : signorini constant

    originally coded by Amir Baharvand (AB) (09-20)
    """
    for n in range(ne):
        [L0, edof, B0] = strn_displ_vec(IX, X, n)
        c = IX[n, 2]
        d = D[edof] #local displacement vector
        le = np.matmul(np.transpose(B0), d)
        lambDa = 1 + c4 * le
        Et = c4 * ( c1 * (1 + 2 * lambDa**(-3)) + 3 * c2 * lambDa**(-4) + \
                    3 * c3 * (-1 + lambDa**(2) - 2 * lambDa**(-3) + 2 * lambDa**(-4)) )
        A = mprop[c - 1, 1]
        ls = c1 * (lambDa - lambDa**(-2)) + c2 * (1 - lambDa**(-3)) + \
                    c3 * (1 - 3 * lambDa + lambDa**(3) - 2 * lambDa**(-3) + 3 * lambDa**(-2))
        N = A * ls
        # computing residual vector 
        re = B0 * N * L0  # local residual
        re = np.transpose(re)
        R_int[edof, 0] = re + R_int[edof, 0]
    
    return R_int