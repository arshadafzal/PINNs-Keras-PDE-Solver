## Burgers' Equation

In this example, the one-dimensional viscous Burgers' equation is solved using a Physics-Informed Neural Network.

The governing equation is:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}
= \nu \frac{\partial^2 u}{\partial x^2}
$$

where:

- $u(x,t)$ is the velocity field
- $x$ is the spatial coordinate
- $t$ is time
- $\nu$ is the kinematic viscosity

The computational domain is defined as:

$$
x \in [-1,1], \qquad t \in [0,1]
$$

The initial condition is:

$$
u(x,0) = -\sin(\pi x)
$$

The boundary conditions are:

$$
u(-1,t) = 0
$$

$$
u(1,t) = 0
$$

Therefore, the PINN is trained to satisfy the governing equation, the initial condition, and the boundary conditions simultaneously.
