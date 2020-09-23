# Coordinates of 3 nodes,
X = [  [0.00,  0.00], 
          [2.00,  0.00], 
          [4.00,  0.00],
          [4.00,  2.00],
          [2.00,  2.00],
          [0.00,  2.00],]

# Topology matrix IX(node1,node2,propno),
IX = [  [1,  2,  1],
          [2,  3,  1],
          [3,  4,  1],
          [4,  5,  1],
          [5,  6,  1],
          [6,  1,  1],
          [2,  6,  1],
          [2,  5,  1],
          [2,  4,  1], ]
      
# Element property matrix mprop = [ E A ],
mprop = [ [1.0, 1.0] ]

# Prescribed loads mat(node,ldof,force)
loads = [ [3,   2, -0.01]]

# Boundary conditions mat(node,ldof,disp)   
bound = [ [1,  1,  0.0],
                  [1,  2,  0.0],
                  [6,  1,  0.0] ]

# Control Parameters
plotdof = 12
