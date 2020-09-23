import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from inp_files.e03 import * # select input file

def signorini_stress_strain(X, IX, mprop, loads, c1, c2, c3, c4):
    """
    Signorini stress-strain relation that mimics rubber response in one-dimension
    """

    ne = IX.shape[0] # number of elements
    d = 0 # displacement
    p = 0 # force
    p_max = 200 # maximum force
    A = mprop[0, 1]
    L = 0
    for n in range(ne):
        a = IX[n, 0]
        b = IX[n, 1]
        dx = X[IX[n, 1] - 1, 0] - X[IX[n, 0] - 1, 0] # dx = xj - xi
        L = L + dx
    
    p_n = np.zeros(200)
    d_n = np.zeros(200)
    n = 0 # increment
    while (p < p_max):
        d = d + 1e-03 # displacement increment
        e = d / L # strain
        x = 1 + c4 * e # lambda
        p = A * ( c1 * (x - x**(-2)) + c2 * (1 - x**(-3)) + \
                 c3 * (1 - 3 * x + x**(3) - 2 * x**(-3) + 3 * x**(-2)) ) #force increment
        p_n[n] = p
        d_n[n] = d
        n += 1
    
    plt.plot(d_n, p_n, 'k', label = 'Signorini stress-strain')
    
    
