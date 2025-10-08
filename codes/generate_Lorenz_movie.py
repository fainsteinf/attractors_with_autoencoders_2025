# F. Fainstein (1,2), G. B. Mindlin (1,2), P. Groisman (3). Reconstructing attractors with autoencoders. Chaos, 35(1) (2025).

# 1. Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Ciudad Universitaria, 1428 Buenos Aires, Argentina.
# 2. CONICET - Universidad de Buenos Aires, Instituto de Física Interdisciplinaria y Aplicada (INFINA), Ciudad Universitaria, 1428 Buenos Aires, Argentina.
# 3. IMAS-CONICET and Departamento de Matemática, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires, Ciudad Universitaria, 1428 Buenos Aires, Argentina


import numpy as np
import matplotlib.pyplot as plt

#We start integrating lorenz dyn system

#RK4
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

#Vector field
def f(v, t):
    sigma, beta, rho =10, 8/3 , 28

    x,y, z = v[0],v[1],v[2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta * z

    return [dxdt, dydt, dzdt]

dt = 0.01
time = np.arange(0, 400, dt)
x = np.zeros_like(time)
y = np.zeros_like(time)
z = np.zeros_like(time)

#initial condition
x_0, y_0, z_0 = 0.1, 0.1, 0.1
x[0] = x_0
y[0] = y_0
z[0] = z_0


for ix, tt in enumerate(time[:-1]):
    x[ix+1], y[ix+1], z[ix+1] = rk4(f, [x[ix], y[ix], z[ix]], tt, dt)

x = x[int(10/dt):]
y = y[int(10/dt):]
z = z[int(10/dt):]
time = time[int(10/dt):] - time[int(10/dt)]

#%% We create the spatial modes
#create modes
X =  np.linspace(0, 1, 40)
Z = np.linspace(0, 1, 40)
X, Z = np.meshgrid(X, Z)

a = 1 #1 / np.sqrt(2)

#Spatial modes for the temperature
Mode_1 = 1 * np.cos(np.pi * a * X) * np.sin(np.pi * Z)
Mode_2 = 1 * np.sin(2 * np.pi * Z)

#Modes for the stream function
Mode_3 = np.sin(np.pi * a * X) * np.sin(np.pi * Z)

# %%
plot modes
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf1 = ax.plot_surface(X, Z, Mode_1, alpha=.5)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel(r"$\theta$", rotation=90)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf2 = ax.plot_surface(X, Z, Mode_2, alpha=.5)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel(r"$\theta$", rotation=90)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf2 = ax.plot_surface(X, Z, Mode_3, alpha=.5)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel(r"$\theta$", rotation=90)
plt.show()

# %% We create the images of the temperature and the stream function

#Temperature images
ims = []
for k in range(len(x)):
    ims.append( (y[k]/20) * Mode_1 - (z[k]/40) * Mode_2 )

#Stream images
ims_velocidad = []
Mode_3 = np.sin(np.pi * a * X) + np.sin(np.pi * Z)
for k in range(len(x)):
  ims_velocidad.append( (x[k]/20) * Mode_3  )

ims = np.array(ims)
ims_velocidad = np.array(ims_velocidad)

#%% We create the full data, both frames put together
x_data = []
for k in range(len(ims)):
  x_data.append( np.concatenate((ims[k], ims_velocidad[k])) )
x_data = np.array(x_data)
print(x_data.shape)
#%%
save_folder = '.../'
# np.save(save_folder+"lorenz_movie_XYZ.npy", x_data)    

