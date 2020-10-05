# fem_non_lin_truss
Finite Element Method (FEM) for non-linear trusses

**Version 0.0.3**

## Features
- Material non-linearity (explicit-implicit)
- Geometry non-lineary (using Newton-Raphson method and for a linear material)
- Supports 2d trusses

## Modulus
*ltc*, linear truss code

*mnltc*, material non-linearity truss code 

*gnltc*, geometry non-linearity truss code (for a linear material)

*gnltc_slender_truss_beam_buckl*, geometry non-linearity truss code (for a linear material) under buckling

## Installation
Coded with Python. Required Numpy, Matplotlib and SciPy packages to run.

## Note
For geometry non-linearity the code still is slow because of large size of the tangential stiffness matrix. In later version, a bandwidth approach will be implemented to reduce the computational cost.

## Contributor(s)
[Amir Baharvand](ambahar@outlook.com)
