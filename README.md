# PINNs-Keras-PDE-Solver

This repository provides simple and educational Keras-based Physics-Informed Neural Network implementations for solving different partial differential equations.

The codes are written in a clear and readable style so that students, researchers, and beginners in scientific machine learning can understand, modify, and extend them for academic and research purposes.

## Highlights

Key highlights include:

- Keras-based implementation of Physics-Informed Neural Networks
- Solves different partial differential equations using PINNs
- Simple and readable code structure
- Uses automatic differentiation for computing PDE residuals
- Includes boundary and initial condition handling
- Uses the L-BFGS-B optimizer from SciPy for training
- Demonstrates a practical Keras–SciPy optimization workflow by flattening trainable neural network weights into a single vector, optimizing them using SciPy's L-BFGS-B optimizer, and restoring the optimized vector back into the Keras model
- External training, validation, or reference data can be imported from files using Pandas or xlrd
- Suitable for academic learning, classroom teaching, and research purposes
- Easy to modify for new PDEs, domains, and boundary conditions
- Useful starting point for scientific machine learning and computational physics applications

## Implemented Examples

- Poisson equation
- Burgers' equation

More PDE examples will be added gradually.

## Repository Structure

```text
PINNs-Keras-PDE-Solver/
│
├── burgers_equation/
├── poisson_equation/
└── README.md
