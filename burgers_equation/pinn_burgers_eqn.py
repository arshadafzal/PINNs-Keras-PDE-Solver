import xlrd
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import scipy as sc
from matplotlib import pyplot as plt
print('Keras version: ', keras.__version__)
print('Numpy version: ', np.__version__)
print('Tensorflow version: ', tf.__version__)

# Reading grid data from Excel file
wb = xlrd.open_workbook('Grid.xls')
# Initial Data
sheet = wb.sheet_by_index(0)
xtrain_0 = np.zeros([sheet.nrows, 2])
utrain_0 = np.zeros([sheet.nrows, 1])

for i in range(sheet.nrows):
    for j in range(2):
        xtrain_0[i, j] = sheet.cell_value(rowx=i, colx=j)
    utrain_0[i] = sheet.cell_value(rowx=i, colx=(sheet.ncols - 1))

# Boundary Data
sheet = wb.sheet_by_index(1)
xtrain_b = np.zeros([sheet.nrows, 2])
utrain_b = np.zeros([sheet.nrows, 1])
for i in range(sheet.nrows):
    for j in range(2):
        xtrain_b[i, j] = sheet.cell_value(rowx=i, colx=j)
    utrain_b[i] = sheet.cell_value(rowx=i, colx=(sheet.ncols - 1))

# Domain Data
sheet = wb.sheet_by_index(2)
xtrain_d = np.zeros([sheet.nrows, 2])
for i in range(sheet.nrows):
    for j in range(2):
        xtrain_d[i, j] = sheet.cell_value(rowx=i, colx=j)

# Grid Visualization
# plt.plot(xtrain_0[:, 0], xtrain_0[:, 1], color='blue', linestyle='none', marker='o',
#          markersize=4, markerfacecolor='none')
# plt.plot(xtrain_b[:, 0], xtrain_b[:, 1], color='red', linestyle='none', marker='s',
#          markersize=4, markerfacecolor='none')
# plt.plot(xtrain_0[:, 0], xtrain_0[:, 1], color='blue', linestyle='none', marker='o',
#          markersize=6, markerfacecolor='none')
# plt.plot(xtrain_d[:, 0], xtrain_d[:, 1], color='red', linestyle='none', marker='.',
#          markersize=6, markerfacecolor='none')
# plt.show()

# Data Preparation - From Numpy arrays to tensors
row1, col1 = xtrain_0.shape
t_0 = xtrain_0[:, 0]; t_0.shape = (row1, 1); t_0 = tf.convert_to_tensor(t_0, dtype=tf.float32)
x_0 = xtrain_0[:, 1]; x_0.shape = (row1, 1); x_0 = tf.convert_to_tensor(x_0, dtype=tf.float32)
utrain_0 = tf.convert_to_tensor(utrain_0, dtype=tf.float32)
row2, col2 = xtrain_b.shape
t_b = xtrain_b[:, 0]; t_b.shape = (row2, 1); t_b = tf.convert_to_tensor(t_b, dtype=tf.float32)
x_b = xtrain_b[:, 1]; x_b.shape = (row2, 1); x_b = tf.convert_to_tensor(x_b, dtype=tf.float32)
utrain_b = tf.convert_to_tensor(utrain_b, dtype=tf.float32)
row, col = xtrain_d.shape
t_d = xtrain_d[:, 0]; t_d.shape = (row, 1); t_d = tf.convert_to_tensor(t_d, dtype=tf.float32)
x_d = xtrain_d[:, 1]; x_d.shape = (row, 1); x_d = tf.convert_to_tensor(x_d, dtype=tf.float32)

# Build Network Architecture
model = keras.Sequential()
model.add(keras.Input(shape=(2, )))
for i in range(6):
    model.add(tf.keras.layers.Dense(32, activation='tanh', kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, activity_regularizer=None))
    # tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
    #                                    gamma_initializer="ones", beta_regularizer=None, gamma_regularizer=None,
    #                                    beta_constraint=None, gamma_constraint=None)
model.add(tf.keras.layers.Dense(1))
model.summary()

# Calculate global loss


def global_loss(t_i, x_i, t_bndry, x_bndry, t_dom, x_dom):
    u_i = model(tf.concat([t_i, x_i], 1))
    u_b = model(tf.concat([t_bndry, x_bndry], 1))

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_dom); tape2.watch(x_dom)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_dom); tape1.watch(x_dom)

            u_d = model(tf.concat([t_dom, x_dom], 1))
            u_t = tape1.gradient(u_d, t_dom)
            u_x = tape1.gradient(u_d, x_dom)
    u_xx = tape2.gradient(u_x, x_dom)
    del tape1
    del tape2

    loss_1 = tf.reduce_mean((u_i - utrain_0)**2)
    loss_2 = tf.reduce_mean((u_b - utrain_b)**2)
    loss_3 = tf.reduce_mean((u_t + u_d * u_x - (0.01/np.pi)*u_xx)**2)
    loss = loss_1 + loss_2 + loss_3
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
        loss_fun = global_loss(t_0, x_0, t_b, x_b,t_d, x_d)
        model_grads = tp.gradient(loss_fun, model.trainable_variables)
        # Flatten and concatenate into a single vector
        grad_vector = np.array([])
        for lst in model_grads:
            lst_vector = lst.numpy().flatten()
            grad_vector = np.append(grad_vector, lst_vector)
    return tf.reshape(loss_fun, ()).numpy(), grad_vector


def det_print(_):
    global it
    loss_fun = global_loss(t_0, x_0, t_b, x_b,t_d, x_d)
    print(f"Epochs {it:d} ---- Loss: {loss_fun: 3.6f}")
    it += 1


res = sc.optimize.minimize(fun=val_grad, x0=weights_vector,
                           method='L-BFGS-B',
                           jac=True,
                           callback=det_print,
                           options={
                               'maxiter': 8000,
                               'ftol': 1e-11,
                               'gtol': 1e-8
                           })

print(res)

model.save('burger_pinn.keras')
# Create time instances for plotting
t_p1 = 0.25*np.ones([100, 1]); t_p2 = 0.50*np.ones([100, 1]); t_p3 = 0.75*np.ones([100, 1])
x_p = tf.reshape(tf.linspace(-1.0, 1.0, 100), (100, 1))
y1 = model(tf.concat([t_p1, x_p], 1))
y2 = model(tf.concat([t_p2, x_p], 1))
y3 = model(tf.concat([t_p3, x_p], 1))


# Reading exact solution
with pd.ExcelFile("burgers_eqn.xlsx.") as xls:
    df1 = pd.read_excel(xls, 'Time_0.25')
    df2 = pd.read_excel(xls, 'Time_0.5')
    df3 = pd.read_excel(xls, 'Time_0.75')
u_snap25 = df1.to_numpy(); u_snap50 = df2.to_numpy(); u_snap75 = df3.to_numpy()
# Plots
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(131)
ax1.set_box_aspect(1)
ax1.plot(x_p, y1, color='blue', linewidth=2, linestyle='-',
         marker='o', markerfacecolor='none', markersize=2, label='u_PINN')
ax1.plot(u_snap25[:, 0], u_snap25[:, 1], color='red', linewidth=2, linestyle=None,
         marker='o', markerfacecolor='none', markersize=2, label='u')
# ax1.plot(x, u_snap[:, 0], color='red', linewidth=2, linestyle=None,
#          marker='o', markerfacecolor='none', markersize=3, label='u')
ax1.minorticks_on()
ax1.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
plt.xlabel('x', fontsize=14, fontstyle='normal', color='black')
plt.ylabel('y', fontsize=14, fontstyle='normal')
plt.title('t = 0.25', fontsize=14)
plt.legend(fontsize=14, edgecolor='none', loc='upper right')
plt.grid(False)
ax2 = fig.add_subplot(132)
ax2.set_box_aspect(1)
ax2.plot(x_p, y2, color='blue', linewidth=2, linestyle='-',
         marker='o', markerfacecolor='none', markersize=2, label='u_PINN')
ax2.plot(u_snap50[:, 0], u_snap50[:, 1], color='red', linewidth=2, linestyle=None,
         marker='o', markerfacecolor='none', markersize=2, label='u')
# ax2.plot(x, u_snap[:, 1], color='red', linewidth=2, linestyle=None,
#          marker='o', markerfacecolor='none', markersize=3, label='u')
ax2.minorticks_on()
ax2.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
plt.xlabel('x', fontsize=14, fontstyle='normal', color='black')
plt.ylabel('y', fontsize=14, fontstyle='normal')
plt.title('t = 0.50', fontsize=14)
plt.legend(fontsize=14, edgecolor='none', loc='upper right')
plt.grid(False)
ax3 = fig.add_subplot(133)
ax3.set_box_aspect(1)
ax3.plot(x_p, y3, color='blue', linewidth=2, linestyle='-',
         marker='o', markerfacecolor='none', markersize=2, label='u_PINN')
ax3.plot(u_snap75[:, 0], u_snap75[:, 1], color='red', linewidth=2, linestyle=None,
         marker='o', markerfacecolor='none', markersize=2, label='u')
# ax3.plot(x, u_snap[:, 2], color='red', linewidth=2, linestyle=None,
#          marker='o', markerfacecolor='none', markersize=3, label='u')
ax3.minorticks_on()
ax3.tick_params(labelsize=12, labelcolor='black', labelfontfamily='Monospace', width=2)
plt.xlabel('x', fontsize=14, fontstyle='normal', color='black')
plt.ylabel('y', fontsize=14, fontstyle='normal')
plt.title('t = 0.75', fontsize=14)
plt.legend(fontsize=14, edgecolor='none', loc='upper right')
plt.grid(False)
fig.tight_layout()
plt.show()
