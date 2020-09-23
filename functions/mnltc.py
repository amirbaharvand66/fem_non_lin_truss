import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

from functions.funcs import *


def mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, method, max_itr = 0, epsilon = 0):
    """
    FEM nonlinear
    Material NonLinearity  Truss Codes (mnltc)
    including:
    
    1. Pure Euler method (PE)
    2. Euler method with ones step equilibrium correction (E1SC)
    3. Newton-Raphson method (NR)
    4. Modified Newton-Raphson method (MNR)
    
    input(s):
    X : topology matrix (element connection matrix)
    IX : nodal coordinates
    mprop : material properties
    loads : applied external load
    bound : boundary conditions
    neq : number of equations (= dof)
    ne = number of elements
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
    c1..c4 : signorini constant
    method : method for solving problem
    
    """
    # 1. Pure Euler method (PE) method
    if method == 'PE':
    
        # build-up point load vector
        for ii in range(nl):
            a = int(loads[ii, 0])
            b = int(loads[ii, 1])
            c = loads[ii, 2]
            if (b % 2) != 0:
                p[2 * a - 2] = c
            else:
                p[2 * a - 1] = c
        
        dp = p / inc
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            # computing tangential stiffness matrix (K)
            K = np.zeros((neq, neq)) # K resets after each increment
            
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
        
            # applying the boundary conditions on K
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
                    
            # computing dD
            # for K matrices close to singular or ill-conditioned, 
            # pseudo-inverse of K based on singular value decomposition (SVD)
            dD = np.linalg.pinv(K).dot(dp)
            D = D + dD
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
        
        
        plt.plot(D_n, p_n, '-^', label = 'Pure Euler Method')
    
    
    # 2. Euler method with ones step equilibrium correction (E1SC) method
    if method == 'E1SC':
        
        # build-up point load vector
        for ii in range(nl):
            a = int(loads[ii, 0])
            b = int(loads[ii, 1])
            c = loads[ii, 2]
            if (b % 2) != 0:
                p[2 * a - 2] = c
            else:
                p[2 * a - 1] = c
        
        dp = p / inc
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            # computing tangential stiffness matrix (K)
            K = np.zeros((neq, neq)) # K resets after each increment
            R_int = np.zeros((neq, 1)) # internal force
            
            for n in range(ne):
                [L0, edof, B0] = strn_displ_vec(IX, X, n)
                c = IX[n, 2]
                d = D[edof]
                e = np.matmul(np.transpose(B0), d)
                lambDa = 1 + c4 * e
                E = c4 * ( c1 * (1 + 2 * lambDa**(-3)) + 3 * c2 * lambDa**(-4) + \
                          3 * c3 * (-1 + lambDa**(2) - 2 * lambDa**(-3) + 2 * lambDa**(-4)) )
                A = mprop[c - 1, 1]
                ke = A * E[0, 0] * L0 * np.outer(B0, np.transpose(B0)) # local stiffness matrix
                # K assembly
                for ii in range(edof.shape[1]):
                    for jj in range(edof.shape[1]):
                        K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]] 
        
            # applying the boundary conditions on K
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
            
            dpR = dp - R # dpR = dP_n - R_(n-1)
            
            # applying boundary conditions on dpR
            for ii in range(nb):
                a = int(bound[ii, 0])
                b = int(bound[ii, 1])
                if (b % 2) != 0:
                    dpR[2 * a - 2, :] = 0
                else: # if (b % 2) == 0
                    dpR[2 * a - 1, :] = 0
        
            
            # computing dD
            # for K matrices close to singular or ill-conditioned, 
            # pseudo-inverse of K based on singular value decomposition (SVD)
            dD = np.linalg.pinv(K).dot(dpR)
            D = D + dD
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
        
            
            # computing R
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
                for ii in range(edof.shape[1]):
                    R_int[edof[0, ii], 0] = re[ii, 0] + R_int[edof[0, ii], 0]
            
            # updating R = R_int - p
            R = R_int - (l_inc + 1) * dp
    
        plt.plot(D_n, p_n, '-s', label = 'Euler method with ones step equilibrium correction')


    # 3. Newton-Raphson (NR) method
    if method == 'NR':
        
        # build-up point load vector
        for ii in range(nl):
            a = int(loads[ii, 0])
            b = int(loads[ii, 1])
            c = loads[ii, 2]
            if (b % 2) != 0:
                p[2 * a - 2] = c
            else:
                p[2 * a - 1] = c
        
        dp = p / inc
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            for eq_itr in range (max_itr):
                
                K = np.zeros((neq, neq)) # tangential stiffness matrix (K) resets for each iteration in an increment
                R_int = np.zeros((neq, 1)) # internal force
                
                # computing R
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
                    for ii in range(edof.shape[1]):
                        R_int[edof[0, ii], 0] = re[ii, 0] + R_int[edof[0, ii], 0]
                
                # updating R = R_int - p
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
                    d = D[edof]
                    e = np.matmul(np.transpose(B0), d)
                    lambDa = 1 + c4 * e
                    E = c4 * ( c1 * (1 + 2 * lambDa**(-3)) + 3 * c2 * lambDa**(-4) + \
                              3 * c3 * (-1 + lambDa**(2) - 2 * lambDa**(-3) + 2 * lambDa**(-4)) )
                    A = mprop[c - 1, 1]
                    ke = A * E[0, 0] * L0 * np.outer(B0, np.transpose(B0)) # local stiffness matrix
                    # K assembly
                    for ii in range(edof.shape[1]):
                        for jj in range(edof.shape[1]):
                            K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]] 
                
                # applying the boundary conditions on K
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
                        
                # LU factorization
                (LUM, PM) = sp.linalg.lu_factor(K) # Lower Upper Matrix, Permutation Matrix
                
                # computing dD
                dD = sp.linalg.lu_solve((-LUM, PM), R) # -inv(UM) * (inv(LM) * R);
                D = D + dD
                
            
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
            
        plt.plot(D_n, p_n, '-o', label = 'Newton-Raphson Method')
        
        
    # 3. Modified Newton-Raphson (MNR) method
    if method == 'MNR':
        
        K = np.zeros((neq, neq)) # tangential stiffness matrix (K)
        
        # build-up point load vector
        for ii in range(nl):
            a = int(loads[ii, 0])
            b = int(loads[ii, 1])
            c = loads[ii, 2]
            if (b % 2) != 0:
                p[2 * a - 2] = c
            else:
                p[2 * a - 1] = c
        
        dp = p / inc
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            
            # computing tangential stiffness matrix (K)
            for n in range(ne):
                [L0, edof, B0] = strn_displ_vec(IX, X, n)
                c = IX[n, 2]
                d = D[edof]
                e = np.matmul(np.transpose(B0), d)
                lambDa = 1 + c4 * e
                E = c4 * ( c1 * (1 + 2 * lambDa**(-3)) + 3 * c2 * lambDa**(-4) + \
                          3 * c3 * (-1 + lambDa**(2) - 2 * lambDa**(-3) + 2 * lambDa**(-4)) )
                A = mprop[c - 1, 1]
                ke = A * E[0, 0] * L0 * np.outer(B0, np.transpose(B0)) # local stiffness matrix
                # K assembly
                for ii in range(edof.shape[1]):
                    for jj in range(edof.shape[1]):
                        K[edof[0, ii], edof[0, jj]] = ke[ii, jj] + K[edof[0, ii], edof[0, jj]] 
            
            # applying the boundary conditions on K
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
                    
            
            # LU factorization
                (LUM, PM) = sp.linalg.lu_factor(K) # Lower Upper Matrix, Permutation Matrix
            
            for eq_itr in range (max_itr):
                
                K = np.zeros((neq, neq)) # tangential stiffness matrix (K) resets for each iteration in an increment
                R_int = np.zeros((neq, 1)) # internal force
                
                # computing R
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
                    for ii in range(edof.shape[1]):
                        R_int[edof[0, ii], 0] = re[ii, 0] + R_int[edof[0, ii], 0]
                
                # updating R = R_int - p
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
                    
                # computing dD
                dD = sp.linalg.lu_solve((-LUM, PM), R) # -inv(UM) * (inv(LM) * R);
                D = D + dD
                
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0   
                
                
        plt.plot(D_n, p_n, '-x', label = 'Modified Newton-Raphson Method')
