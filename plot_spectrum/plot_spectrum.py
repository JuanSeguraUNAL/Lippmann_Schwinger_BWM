import numpy as np
import matplotlib.pyplot as plt

# 1) cargar datos
spec = np.loadtxt("spectrum.dat")
res  = np.loadtxt("resonances.dat")

k_vals = spec[:,0]
norms  = spec[:,1]

# 2) graficar
plt.figure(figsize=(10,6))
# curva ||T|| vs k^2
plt.plot(k_vals**2, norms, '-b', label=r'$||T(k)||$')
# puntos de resonancia
plt.scatter(
    res**2,
    np.interp(res, k_vals, norms),
    c='red', s=100, label='Resonancias')

for i, r in enumerate(res):
    ks2 = r**2
    plt.annotate(
        f'{ks2:.3f}',
        (ks2, np.interp(r, k_vals, norms)),
        textcoords="offset points",
        xytext=(0,10),
        ha='center')

plt.xlabel(r'$k^2$')
plt.ylabel(r'$||T(k)||$')
plt.title(r'Espectro de $||T||$ en función de $k^2$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
