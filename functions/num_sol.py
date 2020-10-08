import numpy as np


def gauss_elm(k, p, d):
    """Gaussian elimination for solving linear system of equation
    i.e., [k]{d} = {p} where k is a square matrix and both d and p are
    column vectors. (1) The first loop k matrix is transformed to an
    upper triangle matrix and (2) in the next phase the system of
    equations is solved using backward substitution approach.
    
    inpout(s)
    k : square matrix of unknown coefficients
    p : column vector of linear system of equation solutions
    d : empty column vector of solutions
    
    output(s)
    d : column vector of solutions
    
    originally coded by Amir Baharvand(AB) (09-20)
    """
    
    # combining k and p
    k = np.concatenate((k, p), axis = 1)
    
    # number of rows
    n = k.shape[0]
    
    # transorming k to an upper triangle
    for ii in range(n):
        for jj in range(n - (ii + 1)): # iterate over rows
            piv = -k[jj + (ii + 1), ii ] / k[ii, ii] # pivot value
            k[jj + (ii + 1), :] = piv * k[ii, :] + k[jj + (ii + 1), :]
             
    # the last coefficient solution (d[n] = p[n] / k[n, n])
    d[n - 1, 0] = k[n - 1, n] / k[n - 1, n - 1]
    
    # backward substition
    for ii in range(n - 2, -1, -1): # moving substition in rows
        a = k[ii, n] # creating an intermediate variable instead of p
        for jj in range(ii, n - 1): # p assembly
            a = a - k[ii, jj + 1] * d[jj + 1, 0]
        d[ii, 0] = a / k[ii, ii] # solving for d[n - 1]
    
    return d
