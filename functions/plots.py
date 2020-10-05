from math import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_def_udef(IX, X, D:list, ls:list, ne:int):
    """
    plot truss deformed and undeformed states
    
    input(s):
    IX : nodal coordinates
    X : topology matrix (element connection matrix)
    D : global displacement vector
    ls : stress vector
    ne : number of elements

    originally coded by Amir Baharvand (AB) (09-20)
    """
    # plot lines for legend
    legend_lines = [Line2D([0], [0], color = 'b'),
    Line2D([0], [0], color = 'r'),
    Line2D([0], [0], color = 'g')]
    fig, ax = plt.subplots()
    
    for n in range(ne):
        
        # plot the undeformed state
        a = IX[n, 0]
        b = IX[n, 1]
        # initialize x, y
        x = np.zeros((1, X.shape[1]))
        y = np.zeros((1, X.shape[1]))
        x = np.array([[ X[a - 1, 0], X[b - 1, 0] ]])
        y = np.array([[X[a - 1, 1], X[b - 1, 1] ]])
        plt.plot(x.T, y.T, 'k--')
        # plot the deformed state
        edof =  np.matrix([ [2 * a - 2, 2 * a - 1, 2 * b - 2, 2 * b - 1] ]) 
        x = x + D[edof[0, 0::2], 0]
        y = y + D[edof[0, 1::2], 0]
        # plot the deformed state based on stress state
        if ls[n, 0] < 0: # compressive 
            plt.plot(x.T, y.T, 'r')
        elif ls[n, 0] > 0: # tension 
            plt.plot(x.T, y.T, 'b')
        else:
            plt.plot(x.T, y.T, 'g')
    
    ax.legend(legend_lines, ['Tension', 'Compression', 'Not-loaded'], ncol=3, \
              bbox_to_anchor=(0.925,-0.08))
        
        
def plot_def_udef_not_fancy(IX, X, D:list, ne:int):
    """
    plot truss deformed and undeformed states wothout stress state
    (no compressive/tensile stress indicator)
    
    input(s):
    IX : nodal coordinates
    X : topology matrix (element connection matrix)
    D : global displacement vector
    ne : number of elements
    
    originally coded by Amir Baharvand (AB) (10-20)
    """
    fig, ax = plt.subplots()
    label_list = []
    for n in range(ne):
        # plot the undeformed state
        a = IX[n, 0]
        b = IX[n, 1]
        # initialize x, y
        x = np.zeros((1, X.shape[1]))
        y = np.zeros((1, X.shape[1]))
        x = np.array([[ X[a - 1, 0], X[b - 1, 0] ]])
        y = np.array([[X[a - 1, 1], X[b - 1, 1] ]])
        plt.plot(x.T, y.T, 'k--', label = 'Undeformed')
        # plot the deformed state
        edof =  np.matrix([ [2 * a - 2, 2 * a - 1, 2 * b - 2, 2 * b - 1] ]) 
        x = x + D[edof[0, 0::2], 0]
        y = y + D[edof[0, 1::2], 0]
        plt.plot(x.T, y.T, 'k', label = 'Deformed state')
    
        handles, labels = ax.get_legend_handles_labels()
        for a_label in labels:
            if a_label not in label_list:
                label_list.append(a_label)
    plt.legend(label_list)
    plt.axis('square')