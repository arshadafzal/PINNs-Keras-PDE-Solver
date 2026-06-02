# 2D Navier–Stokes Channel Flow Solver Using PINNs
import numpy as np
import tensorflow as tf
import keras
import scipy as sc
import pandas as pd
from matplotlib import pyplot as plt
print('Keras version: ', keras.__version__)
print('Numpy version: ', np.__version__)
print('Tensorflow version: ', tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# Reading grid data from Excel file
with pd.ExcelFile("NS_2DFLOW.xlsx") as xls:
    df1 = pd.read_excel(xls, 'bdry_bottom')
    df2 = pd.read_excel(xls, 'bdry_top')
    df3 = pd.read_excel(xls, 'bdry_left')
    df4 = pd.read_excel(xls, 'bdry_right')
    df5 = pd.read_excel(xls, 'domain_data')
bdry_b = df1.to_numpy(); bdry_t = df2.to_numpy(); bdry_l = df3.to_numpy(); bdry_r = df4.to_numpy()
domain = df5.to_numpy()

# Data Preparation - From Numpy arrays to tensors
# Bottom Wall
x_b = tf.convert_to_tensor(bdry_b[:, 0:1], dtype=tf.float32)
y_b = tf.convert_to_tensor(bdry_b[:, 1:2], dtype=tf.float32)
utrain_b = tf.convert_to_tensor(bdry_b[:, 2:3], dtype=tf.float32)

# Top wall
x_t = tf.convert_to_tensor(bdry_t[:, 0:1], dtype=tf.float32)
y_t = tf.convert_to_tensor(bdry_t[:, 1:2], dtype=tf.float32)
utrain_t = tf.convert_to_tensor(bdry_t[:, 2:3], dtype=tf.float32)

# Inlet
x_l = tf.convert_to_tensor(bdry_l[:, 0:1], dtype=tf.float32)
y_l = tf.convert_to_tensor(bdry_l[:, 1:2], dtype=tf.float32)
utrain_l = tf.convert_to_tensor(bdry_l[:, 2:3], dtype=tf.float32)

# Outlet
x_r = tf.convert_to_tensor(bdry_r[:, 0:1], dtype=tf.float32)
y_r = tf.convert_to_tensor(bdry_r[:, 1:2], dtype=tf.float32)
ptrain_r = tf.convert_to_tensor(bdry_r[:, 2:3], dtype=tf.float32)

# Domain Points
x_d = tf.convert_to_tensor(domain[:, 0:1], dtype=tf.float32)
y_d = tf.convert_to_tensor(domain[:, 1:2], dtype=tf.float32)

# Build Network Architecture
model = keras.Sequential()
model.add(keras.Input(shape=(2, )))
for i in range(5):
    model.add(tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, activity_regularizer=None))
    tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
                                       gamma_initializer="ones", beta_regularizer=None, gamma_regularizer=None,
                                       beta_constraint=None, gamma_constraint=None)
model.add(tf.keras.layers.Dense(3))
model.summary()


# Calculate global loss


def global_loss(b_x, b_y, t_x, t_y, l_x, l_y, r_x, r_y, d_x, d_y):
    b_pred = model(tf.concat([b_x, b_y], 1))
    t_pred = model(tf.concat([t_x, t_y], 1))
    l_pred = model(tf.concat([l_x, l_y], 1))
    r_pred = model(tf.concat([r_x, r_y], 1))

    u_b = b_pred[:, 0:1]
    v_b = b_pred[:, 1:2]

    u_t = t_pred[:, 0:1]
    v_t = t_pred[:, 1:2]

    u_l = l_pred[:, 0:1]
    v_l = l_pred[:, 1:2]

    p_r = r_pred[:, 2:3]

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(d_x); tape2.watch(d_y)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(d_x); tape1.watch(d_y)

            u_d = model(tf.concat([d_x, d_y], 1))
            u = u_d[:, 0:1]; v = u_d[:, 1:2]; p = u_d[:, 2:3]
        u_x = tape1.gradient(u, d_x)
        u_y = tape1.gradient(u, d_y)

        v_x = tape1.gradient(v, d_x)
        v_y = tape1.gradient(v, d_y)

        p_x = tape1.gradient(p, d_x)
        p_y = tape1.gradient(p, d_y)

    u_xx = tape2.gradient(u_x, d_x); v_xx = tape2.gradient(v_x, d_x)
    u_yy = tape2.gradient(u_y, d_y); v_yy = tape2.gradient(v_y, d_y)

    del tape1
    del tape2

    bdry_loss_u = (
            tf.reduce_mean((u_l - utrain_l) ** 2) +
            tf.reduce_mean((u_b - utrain_b) ** 2) +
            tf.reduce_mean((u_t - utrain_t) ** 2)
    )

    bdry_loss_v = (
            tf.reduce_mean(v_b ** 2) +
            tf.reduce_mean(v_t ** 2) +
            tf.reduce_mean(v_l ** 2)
    )

    bdry_loss_p = tf.reduce_mean((p_r - ptrain_r) ** 2)

    loss_u = tf.reduce_mean(((u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy)) ** 2)
    loss_v = tf.reduce_mean(((u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy)) ** 2)
    loss_c = tf.reduce_mean((u_x + v_y) ** 2)
    loss = bdry_loss_u + bdry_loss_v + bdry_loss_p + loss_u + loss_v + loss_c
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
        loss_fun = global_loss(x_b, y_b, x_t, y_t, x_l, y_l, x_r, y_r, x_d, y_d)
        model_grads = tp.gradient(loss_fun, model.trainable_variables)
        # Flatten and concatenate into a single vector
        grad_vector = np.array([])
        for lst in model_grads:
            lst_vector = lst.numpy().flatten()
            grad_vector = np.append(grad_vector, lst_vector)
    return tf.reshape(loss_fun, ()).numpy(), grad_vector


def det_print(_):
    global it
    loss_fun = global_loss(x_b, y_b, x_t, y_t, x_l, y_l, x_r, y_r, x_d, y_d)
    print(f"Epochs {it:d} ---- Loss: {loss_fun: 3.6f}")
    it += 1


res = sc.optimize.minimize(fun=val_grad, x0=weights_vector,
                           method='L-BFGS-B',
                           jac=True,
                           callback=det_print,
                           options={
                               'maxiter': 10000,
                               'ftol': 1e-11
                           })

print(res)
model.save('pinn_2D_FLOW.keras')
# Contour Plot
x = tf.linspace(0, 5, 40)
y = tf.linspace(0, 1, 40)
X, Y = tf.meshgrid(x, y)
z = model(tf.concat([tf.reshape(X, (1600, 1)), tf.reshape(Y, (1600, 1))], 1))[:, 0:1]
Z = tf.reshape(z, (40, 40))
fig, ax1 = plt.subplots(nrows=1, ncols=1)
C1 = ax1.contourf(X, Y, Z, 10, cmap='RdGy')
fig.colorbar(C1, shrink=0.5)
ax1.minorticks_on()
ax1.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
ax1.set_xlabel('x', fontsize=14, fontstyle='normal', color='black')
ax1.set_ylabel('y', fontsize=14, fontstyle='normal')
ax1.set_title('Predicted', fontsize=14)
# ax1.set_box_aspect()
plt.tight_layout()
plt.show()
