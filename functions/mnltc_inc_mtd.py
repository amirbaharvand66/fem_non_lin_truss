import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import matplotlib
from matplotlib import pyplot as plt

# material nonlinearity incremental methods
from functions.ltc_funcs import *
from functions.nltc_funcs import *
from functions.mnltc_inc_mtd import *


def mnltc(X, IX, mprop, loads, bound, neq, ne, nl, nb, p, D, dD, R, dpR, inc, D_n, p_n, c1, c2, c3, c4, method, max_itr = 0, epsilon = 0):
    """
    FEM nonlinear
    Material NonLinearity Truss Codes (mnltc) incremental methods
    including:
    1. Pure Euler method (PE)
    2. Euler method with one step equilibrium correction (E1SC)
    3. Newton-Raphson method (NR)
    4. Modified Newton-Raphson method (MNR)
    
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
    c1..c4 : signorini constant
    method : method for solving problem

    originally coded by Amir Baharvand (AB) (09-20)
    """
    ##################################
    # 1. Pure Euler method (PE) method
    ##################################
    if method == 'PE':
    
        # build-up point load vector
        p = build_load_vec(loads, nl, p)
        dp = p / inc # creating load increment
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            # initializing
            K = np.zeros((neq, neq)) # K resets after each increment

            # computing tangential stiffness matrix (K)
            K = mnl_K(IX, X, mprop, ne, D, K, c1, c2, c3, c4)
            
            # applying the boundary conditions on K
            K = bnd_cnd_K(bound, nb, K)
                    
            # computing dD
            # for K matrices close to singular or ill-conditioned, 
            # pseudo-inverse of K based on singular value decomposition (SVD)
            dD = np.linalg.pinv(K).dot(dp)
            D = D + dD
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
        
        
        plt.plot(D_n, p_n, '-^', label = 'Pure Euler Method')
    

    #####################################################################
    # 2. Euler method with ones step equilibrium correction (E1SC) method
    #####################################################################
    if method == 'E1SC':
        
        # build-up point load vector
        p = build_load_vec(loads, nl, p)
        dp = p / inc # creating load increment
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            # initializing
            K = np.zeros((neq, neq)) # K resets after each increment
            R_int = np.zeros((neq, 1)) # internal force

            # computing tangential stiffness matrix (K)
            K = mnl_K(IX, X, mprop, ne, D, K, c1, c2, c3, c4)
            
            # applying the boundary conditions on K
            K = bnd_cnd_K(bound, nb, K)
            
            dpR = dp - R # dpR = dP_n - R_(n-1)
            
            # applying boundary conditions on dpR
            dpR = bnd_cnd_rsdl(bound, nb, dpR)
        
            # computing dD
            # for K matrices close to singular or ill-conditioned, 
            # pseudo-inverse of K based on singular value decomposition (SVD)
            dD = np.linalg.pinv(K).dot(dpR)
            D = D + dD
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
        
            
            # computing R_int
            R_int = mnl_int_force(IX, X, mprop, ne, D, R_int, c1, c2, c3, c4)
            
            # computing R = R_int - p
            R = R_int - (l_inc + 1) * dp
    
        plt.plot(D_n, p_n, '-s', label = 'Euler method with ones step equilibrium correction')


    ###############################
    # 3. Newton-Raphson (NR) method
    ###############################
    if method == 'NR':
        
        # build-up point load vector
        p = build_load_vec(loads, nl, p)
        dp = p / inc # creating load increment
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            for eq_itr in range (max_itr):
                
                # initializing
                K = np.zeros((neq, neq)) # tangential stiffness matrix (K) resets for each iteration in an increment
                R_int = np.zeros((neq, 1)) # internal force
                
                # computing R_int
                R_int = mnl_int_force(IX, X, mprop, ne, D, R_int, c1, c2, c3, c4)
            
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
                K = mnl_K(IX, X, mprop, ne, D, K, c1, c2, c3, c4)

                # applying the boundary conditions on K
                K = bnd_cnd_K(bound, nb, K)
                        
                # LU factorization
                (LUM, PM) = sp.linalg.lu_factor(K) # Lower Upper Matrix, Permutation Matrix
                
                # computing dD
                dD = sp.linalg.lu_solve((-LUM, PM), R) # -inv(UM) * (inv(LM) * R);
                D = D + dD
                
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0
            
        plt.plot(D_n, p_n, '-o', label = 'Newton-Raphson Method')
    

    #########################################
    # 4. Modified Newton-Raphson (MNR) method
    #########################################
    if method == 'MNR':
        
        # build-up point load vector
        p = build_load_vec(loads, nl, p)
        dp = p / inc # creating load increment

        # initializing
        K = np.zeros((neq, neq)) # tangential stiffness matrix (K)
        
        for l_inc in range(inc): # loop over l_inc (load increment)
            p = l_inc * p + dp
            
            # computing tangential stiffness matrix (K)
            K = mnl_K(IX, X, mprop, ne, D, K, c1, c2, c3, c4)

            # applying the boundary conditions on K
            K = bnd_cnd_K(bound, nb, K)
                    
            # LU factorization
            (LUM, PM) = sp.linalg.lu_factor(K) # Lower Upper Matrix, Permutation Matrix
            
            for eq_itr in range (max_itr):

                # reset tangential stiffness matrix (K) for each iteration in an increment
                K = np.zeros((neq, neq)) 
                R_int = np.zeros((neq, 1)) # internal force
                
                # computing R_int
                R_int = mnl_int_force(IX, X, mprop, ne, D, R_int, c1, c2, c3, c4)
            
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
                    
                # computing dD
                dD = sp.linalg.lu_solve((-LUM, PM), R) # -inv(UM) * (inv(LM) * R);
                D = D + dD
                
            idx = np.where(dp != 0)[0]
            D_n[l_inc] = D[idx] # appending displacements for the node where p != 0
            p_n[l_inc] = (l_inc + 1) * dp[dp != 0] # appending forces for the node where p != 0   
                       
        plt.plot(D_n, p_n, '-x', label = 'Modified Newton-Raphson Method')