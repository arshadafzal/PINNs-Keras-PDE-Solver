# Simulation of the Poisson Equation Using Physics-Informed Neural Networks

import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import scipy as sc
from matplotlib import pyplot as plt
print('Keras version: ', keras.__version__)
print('Numpy version: ', np.__version__)
print('Tensorflow version: ', tf.__version__)

tf.keras.backend.set_floatx('float64')
tf.config.experimental.enable_tensor_float_32_execution(False)
# Reading grid data from Excel file
with pd.ExcelFile("grid_poisson.xlsx.") as xls:
    df1 = pd.read_excel(xls, 'bdry_bottom')
    df2 = pd.read_excel(xls, 'bdry_top')
    df3 = pd.read_excel(xls, 'bdry_left')
    df4 = pd.read_excel(xls, 'bdry_right')
    df5 = pd.read_excel(xls, 'domain_data')
bdry_b = df1.to_numpy(); bdry_t = df2.to_numpy(); bdry_l = df3.to_numpy(); bdry_r = df4.to_numpy()
domain = df5.to_numpy()
# Data Preparation - From Numpy arrays to tensors

# Boundary data
bdry_b_data = tf.convert_to_tensor(bdry_b[:, 0:2], dtype=tf.float64)
utrain_b = tf.convert_to_tensor(bdry_b[:, 2:3], dtype=tf.float64)
bdry_t_data = tf.convert_to_tensor(bdry_t[:, 0:2], dtype=tf.float64)
utrain_t = tf.convert_to_tensor(bdry_t[:, 2:3], dtype=tf.float64)
bdry_l_data = tf.convert_to_tensor(bdry_l[:, 0:2], dtype=tf.float64)
utrain_l = tf.convert_to_tensor(bdry_l[:, 2:3], dtype=tf.float64)
bdry_r_data = tf.convert_to_tensor(bdry_r[:, 0:2], dtype=tf.float64)
utrain_r = tf.convert_to_tensor(bdry_r[:, 2:3], dtype=tf.float64)

# Domain points
x_d = tf.convert_to_tensor(domain[:, 0:1], dtype=tf.float64)
y_d = tf.convert_to_tensor(domain[:, 1:2], dtype=tf.float64)
bdry_dom_data = tf.convert_to_tensor(domain, dtype=tf.float64)

# Build Network Architecture
model = keras.Sequential()
model.add(keras.Input(shape=(2, )))
for i in range(4):
    model.add(tf.keras.layers.Dense(32, activation='tanh', kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, activity_regularizer=None))
    # tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
    #                                    gamma_initializer="ones", beta_regularizer=None, gamma_regularizer=None,
    #                                    beta_constraint=None, gamma_constraint=None)
model.add(tf.keras.layers.Dense(1))
model.summary()

# Calculate global loss


def global_loss(b_bdry, t_bdry, l_bdry, r_bdry, d_x, d_y):
    u_b = model(b_bdry)
    u_t = model(t_bdry)
    u_l = model(l_bdry)
    u_r = model(r_bdry)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(d_x); tape2.watch(d_y)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(d_x); tape1.watch(d_y)

            u_d = model(tf.concat([d_x, d_y], 1))
        u_x = tape1.gradient(u_d, d_x)
        u_y = tape1.gradient(u_d, d_y)

    u_xx = tape2.gradient(u_x, d_x)
    u_yy = tape2.gradient(u_y, d_y)

    del tape1
    del tape2

    loss_b = (
        tf.reduce_mean((u_b - utrain_b)**2) +
        tf.reduce_mean((u_t - utrain_t)**2) +
        tf.reduce_mean((u_l - utrain_l)**2) +
        tf.reduce_mean((u_r - utrain_r)**2)
    )
    pi = tf.constant(np.pi, dtype=tf.float64)
    loss_d = tf.reduce_mean((u_xx + u_yy + tf.sin(pi * d_x)*tf.sin(pi * d_y))**2)

    loss = 100.0 * loss_b + loss_d
    return loss


# Convert model weights from tensors to a vector and back
weights = model.get_weights()
shapes = [w.shape for w in weights]
weights_vector = np.concatenate([w.flatten() for w in weights])


def vector_to_weights(vector, weight_shapes):
    wts = []
    idx = 0

    for shape in weight_shapes:
        size = np.prod(shape)
        wts.append(vector[idx:idx+size].reshape(shape))
        idx += size
    return wts


# Network Training
it = 0


def val_grad(parameters):
    par = vector_to_weights(parameters, shapes)
    model.set_weights(par)
    with tf.GradientTape() as tp:
        loss_fun = global_loss(bdry_b_data, bdry_t_data, bdry_l_data, bdry_r_data, x_d, y_d)
        model_grads = tp.gradient(loss_fun, model.trainable_variables)
        # Flatten and concatenate into a single vector
        grad_vector = np.array([])
        for lst in model_grads:
            lst_vector = lst.numpy().flatten()
            grad_vector = np.append(grad_vector, lst_vector)
    return tf.reshape(loss_fun, ()).numpy(), grad_vector


def det_print(_):
    global it
    loss_fun = global_loss(bdry_b_data, bdry_t_data, bdry_l_data, bdry_r_data, x_d, y_d)
    print(f"Epochs {it:d} ---- Loss: {loss_fun: 3.6f}")
    it += 1


res = sc.optimize.minimize(fun=val_grad, x0=weights_vector,
                           method='L-BFGS-B',
                           jac=True,
                           callback=det_print,
                           options={
                               'maxiter': 10000,
                               'ftol': 1e-11,
                               'gtol': 1e-8
                           })

print(res)

model.save('poisson_pinn.keras')
# Contour Plot
x = tf.linspace(0, 1, 40)
y = tf.linspace(0, 1, 40)
X, Y = tf.meshgrid(x, y)
z = model(tf.concat([tf.reshape(X, (1600, 1)), tf.reshape(Y, (1600, 1))], 1))
Z = tf.reshape(z, (40, 40))
Z_analytic = (np.sin(np.pi*X) * np.sin(np.pi*Y)) / (2 * np.pi**2)
Error = tf.abs(Z - Z_analytic)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
C1 = ax1.contourf(X, Y, Z, 10, cmap='RdGy')
fig.colorbar(C1, shrink=1.0, orientation='horizontal', pad=0.08)
ax1.minorticks_on()
ax1.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
ax1.set_xlabel('x', fontsize=14, fontstyle='normal', color='black')
ax1.set_ylabel('y', fontsize=14, fontstyle='normal')
ax1.set_title('Predicted', fontsize=14)
ax1.set_box_aspect(1)
C2 = ax2.contourf(X, Y, Z_analytic, 10, cmap='RdGy')
fig.colorbar(C2, shrink=1.0, orientation='horizontal', pad=0.08)
ax2.minorticks_on()
ax2.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
ax2.set_xlabel('x', fontsize=14, fontstyle='normal', color='black')
ax2.set_ylabel('y', fontsize=14, fontstyle='normal')
ax2.set_title('Analytic', fontsize=14)
ax2.set_box_aspect(1)
C3 = ax3.contourf(X, Y, Error, 10, cmap='RdGy')
fig.colorbar(C3, shrink=1.0, orientation='horizontal', pad=0.08)
ax3.minorticks_on()
ax3.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
ax3.set_xlabel('x', fontsize=14, fontstyle='normal', color='black')
ax3.set_ylabel('y', fontsize=14, fontstyle='normal')
ax3.set_title('Error', fontsize=14)
ax3.set_box_aspect(1)
plt.tight_layout()
plt.show()
