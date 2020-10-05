# Coordinates of nodes,
X = [ [0, 0],
      [1.5, 0.4],
      [3, 0] ]

# Topology matrix IX(node1,node2,propno),
IX = [[1, 2, 1],
      [2, 3, 1]]

# Element property matrix mprop = [ E A ],
mprop = [[1, 1]]

# Prescribed loads mat(node,ldof,force)
loads = [[ 2, 2, -0.3 ]]

# Boundary conditions mat(node,ldof,disp)
bound = [[1, 1, 0],
         [1, 2, 0],
         [3, 1, 0],
         [3, 2, 0]]

# spring matrix
st = [[2, 2, 0.05]]

# Control Parameters
plotdof = 3