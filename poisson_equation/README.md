# Poisson Equation using PINNs

This folder contains a simple Keras-based Physics-Informed Neural Network implementation for solving the two-dimensional Poisson equation on a square domain.

## Governing Equation

The governing equation is:

$$
\frac{\partial^2 u}{\partial x^2}
+
\frac{\partial^2 u}{\partial y^2}
=
-\sin(\pi x)\sin(\pi y)
$$

or equivalently,

$$
\frac{\partial^2 u}{\partial x^2}
+
\frac{\partial^2 u}{\partial y^2}
+
\sin(\pi x)\sin(\pi y)
=
0
$$

where:

- $u(x,y)$ is the unknown solution
- $x$ and $y$ are spatial coordinates
- $\sin(\pi x)\sin(\pi y)$ is the source term

## Computational Domain

The computational domain is a square domain:

$$
\Omega = [0,1] \times [0,1]
$$

That is:

$$
0 \leq x \leq 1, \qquad 0 \leq y \leq 1
$$

## Boundary Conditions

Zero Dirichlet boundary conditions are applied on all boundaries of the square domain:

$$
u(x,y) = 0, \qquad (x,y) \in \partial \Omega
$$

Equivalently:

$$
u(0,y) = 0, \qquad 0 \leq y \leq 1
$$

$$
u(1,y) = 0, \qquad 0 \leq y \leq 1
$$

$$
u(x,0) = 0, \qquad 0 \leq x \leq 1
$$

$$
u(x,1) = 0, \qquad 0 \leq x \leq 1
$$

