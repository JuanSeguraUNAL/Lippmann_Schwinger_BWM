import numpy as np
import matplotlib.pyplot as plt

x, y = np.loadtxt('boundary.dat').T

#Dibuja frontera billar
plt.plot(x, y, 'k-', lw=1)
plt.axis('equal')
plt.title("Frontera del billar parabólico confocal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
