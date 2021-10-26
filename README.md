# fem_non_lin_truss
Finite Element Method (FEM) for non-linear trusses

## Features
- Material non-linearity (explicit-implicit)
- Geometry non-lineary (using Newton-Raphson method and for a linear material)
- Supports 2d trusses

## Modulus
*ltc*, linear truss code

*mnltc*, material non-linearity truss code 

*gnltc*, geometry non-linearity truss code (for a linear material)

*gnltc_slender_truss_beam_buckl*, geometry non-linearity truss code (for a linear material) under buckling

## Dependencies
Numpy, Matplotlib and SciPy

## Note
For geometry non-linearity the code still is slow because of large size of the tangential stiffness matrix. In next version, a bandwidth approach will be implemented to reduce the computational cost.

Gaussian elimination is available in *functions/num_sol*.