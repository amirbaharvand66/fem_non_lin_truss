from math import *
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from functions.ltc_funcs import *


def gnltc_nr(X, IX, mprop, loads, bound, st, neq, ne, nl, nb, nst, p, D, dD, R, inc, d_, p_, \
             max_itr, epsilon, spring = 'off'):
    """
    FEM nonlinear
    Geometry Non-Linearity Truss Code (gnltc) + Newton_Raphson (nr) method
    
    input(s):
    X : topology matrix (element connection matrix)
    IX : nodal coordinates
    mprop : material properties
    loads : applied external load
    bound : boundary conditions
    neq : number of equations (= dof)
    ne : number of elements
    nl : number of point loads
    nb : number of boundary conditions
    p : point load vector
    D : global displacement vector
    dD : infinitesimal displacement vector
    R : internal force (residual vector)
    dpR : (dP - R), dP : infinitesimal load
    inc : number of increments
    D_n : final displacement vector for plotting
    p_n : final force vector for plotting
    max_itr : maximum iteration for each load increment in NR and MNR methods
    epsilon : required for acceptance criterion of residuals in NR and MNR methods
    spring : spring stiffness(on/off)

    originally coded by Amir Baharvand (AB) (10-20)
    """
    # build-up point load vector
    p = build_load_vec(loads, nl, p)
    dp = p / inc # creating load increment
    for l_inc in range(inc): # loop over l_inc (load increment)
        
        for eq_itr in range (max_itr): # loop over iteration in each load increment
            
            # initializing
            K = np.zeros((neq, neq)) # tangential stiffness matrix (K) resets for each iteration in an increment
            R_int = np.zeros((neq, 1)) # internal force
            k_spr = np.zeros((neq, neq)) # spring stiffness (k_spr) 
            
            # computing R_int
            for n in range(ne):
                [L0, edof, B0] = strn_displ_vec(IX, X, n)
                c = IX[n, 2]
                E = mprop[0, 0]
                A = mprop[0, 1]
                d = D[edof]
                # strain-displacement matrix for displacement dependent vector
                Bd = (1 / L0**2) * np.matmul( np.matrix([[1, 0, -1, 0], 
                                                         [0, 1, 0, -1], 
                                                         [-1, 0, 1, 0], 
                                                         [0, -1, 0, 1]]), d )
                Bd = np.transpose(Bd)
                
                e0 = np.matmul(np.transpose(B0), d) # linear strain
                ed = np.matmul(np.transpose(Bd), d) # displacement dependent strain
                ge = e0 + 0.5 * ed # Green-Lagrange strain
                # ge = float(ge)
                ls = E * ge # local stress
                N = A * ls # element nodal force
                # computing residual vector 
                re = N * (np.transpose(B0) + np.transpose(Bd)) * L0  # local residual
                R_int[edof, 0] = re + R_int[edof, 0]
                
                
            
            # decide whether to enter spring stiffness into computation
            if spring == 'on':
                # extract spring stiffness (k_spr) from input file
                for ii in range(nst):
                    a = int(st[ii, 0])
                    b = int(st[ii, 1])
                    c = st[ii, 2]
                    if (b % 2) != 0:
                        k_spr[2 * a - 2, 2 * a - 2] = c
                    else:
                        k_spr[2 * a - 1, 2 * a - 1] = c
                # computing R = R_int - ( p - k_spr * D)
                R = R_int - ((l_inc + 1) * dp - np.matmul(k_spr, D))
            else:
                # computing R = R_int - p
                R = R_int - (l_inc + 1) * dp
            
            # applying boundary conditions on R
            for ii in range(nb):
                a = int(bound[ii, 0])
                b = int(bound[ii, 1])
                if (b % 2) != 0:
                    R[2 * a - 2, :] = 0
                else: # if (b % 2) == 0
                    R[2 * a - 1, :] = 0
        
                # strop iteration criterion
                if np.linalg.norm(R) <= epsilon * np.linalg.norm(p):
                    break
            
            # computing tangential stiffness matrix (K)
            for n in range(ne):
                [L0, edof, B0] = strn_displ_vec(IX, X, n)
                c = IX[n, 2]
                E = mprop[0, 0]
                A = mprop[0, 1]
                d = D[edof] #local displacement vector
                # strain-displacement matrix for displacement dependent vector
                Bd = (1 / L0**2) * np.matmul( np.matrix([[1, 0, -1, 0], 
                                                         [0, 1, 0, -1], 
                                                         [-1, 0, 1, 0], 
                                                         [0, -1, 0, 1]]), d )
                Bd = np.transpose(Bd)
                
                e0 = np.matmul(np.transpose(B0), d) # linear strain
                ed = np.matmul(np.transpose(Bd), d) # displacement dependent strain
                ge = e0 + 0.5 * ed # Green-Lagrange strain
                
                if np.shape(ge) == (1, 1):
                    ge = float(ge)
                
                ls = E * ge # local stress
                N = A * ls # element nodal force
                
                # computing element tangential stiffness matrices
                # initial stress stiffness matrix
                k_sigma = (1 / L0 **2) * np.matrix([[1, 0, -1, 0], 
                                                    [0, 1, 0, -1], 
                                                    [-1, 0, 1, 0], 
                                                    [0, -1, 0, 1]]) * N * L0 
                # linear stiffness matrix
                k_0 = A *E *L0 * np.matmul(B0, np.transpose(B0))
                # initial displacement stiffness matrix
                k_d = A * E * L0 * ( np.matmul(B0, np.transpose(Bd)) +
                                     np.matmul(Bd, np.transpose(B0)) + 
                                     np.matmul(Bd, np.transpose(Bd)))
                ke = k_sigma + k_0 + k_d # local stiffness matrix
                
                # K assembly
                for ii in range(edof.shape[1]):
                    for jj in range(edof.shape[1]):
                        K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]]
                        
            if spring == 'on':
                # K + k_spr
                K = K + k_spr
    
            # applying the boundary conditions on K
            K = bnd_cnd_K(bound, nb, K)
            
                    
            #### LU factorization
            (LUM, PM) = sp.linalg.lu_factor(-K) # Lower Upper Matrix, Permutation Matrix
            
            # computing dD
            dD = sp.linalg.lu_solve((LUM, PM), R) # -inv(UM) * (inv(LM) * R);
            D = D + dD
            
        d_[:, l_inc] = D[:, 0]
        p_[:, l_inc] = (l_inc + 1) * dp[:, 0]
        
        # create loading because of slow lu_factor and lu_solve
        # will be enhanced later by bandwidth method for K matrix
        print("Load increment number {0}\n".format(l_inc + 1), end = "\r")
        
    return d_, p_, D