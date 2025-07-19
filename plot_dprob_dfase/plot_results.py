import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

# Carga los datos
D = np.loadtxt("density.dat")
P = np.loadtxt("phase.dat")
B = np.loadtxt("boundary.dat")

# Reconstruye la grilla física
N = D.shape[0]
xmin, xmax = -8, 8
ymin, ymax = -8, 8
x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
X, Y = np.meshgrid(x, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1) Densidad (normalizada a [0,1])
Dnorm = D / D.max()

im1 = ax1.pcolormesh(
    X, Y, Dnorm,
    shading='auto',
    cmap='inferno',
    norm=colors.Normalize(vmin=0, vmax=1)
)
ax1.plot(B[:,0], B[:,1], 'w-', lw=1)
ax1.set_aspect('equal')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_title('Densidad de probabilidad')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 0.25, 0.5, 0.75, 1.0])
cbar1.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
cbar1.set_label(r'$|\psi|^2$')

# 2) Fase (rango [-π,π])
im2 = ax2.pcolormesh(
    X, Y, P,
    shading='auto',
    cmap='hsv',
    norm=colors.Normalize(vmin=-np.pi, vmax=np.pi)
)
ax2.plot(B[:,0], B[:,1], 'k-', lw=1)
ax2.set_aspect('equal')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_title('Distribución de fase')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

cbar2 = fig.colorbar(
    im2, ax=ax2,
    ticks=[-np.pi, 0, np.pi]
)
cbar2.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
cbar2.set_label(r'Fase $\psi$')

plt.tight_layout()
plt.show()
