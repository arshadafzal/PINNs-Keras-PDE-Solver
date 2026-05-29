# PINNs-Keras-PDE-Solver
Simple Keras-based Physics-Informed Neural Network codes for solving different PDEs for academic and research purposes.

## Optimizer

This code uses the **L-BFGS-B optimizer from SciPy** for training the Physics-Informed Neural Network.

The optimizer is implemented using:

```python
scipy.optimize.minimize(method="L-BFGS-B")
