from math import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def krenk_1993(mprop, inc, spring = 'off'):
    """
    Krenk analytical solution 
    
    input(s):
    E : Young's modulus
    A : cross-sectional area
    inc : number of increments
    spring = spring stiffness (on/off)

    """
    E = mprop[0, 0]
    A = mprop[0, 1]
    a = 0.4
    L = sqrt(a**2 + 1.5**2)
    D = np.linspace(-1.5, 0, inc)
    
    if spring == 'off':
        P = 2 * E * A * (a / L)**3 * (D / a + 3 / 2 *(D / a)**2 + 1 / 2 * (D / a)**3)
        plt.plot(D, P, label = "Krenk non-modified solution")
        spring = 'off'
    else:
        k = 0.05
        P = 2 * E * A * (a / L)**3 * (D / a + 3 / 2 *(D / a)**2 + 1 / 2 * (D / a)**3) + k * D
        plt.plot(D, P, label = "Krenk modified solution")
        spring = 'on'
    
    return spring