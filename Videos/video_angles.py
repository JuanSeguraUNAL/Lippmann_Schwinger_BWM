import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, animation
from matplotlib.animation import FFMpegWriter

# Recoge parámetros desde la línea de comandos
xi0 = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0  # Valores por defecto
eta0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

step = 0.05
angles = np.arange(0.0, 90 + step, step)
K2 = [0.408, 0.818]
B = np.loadtxt("resultados/boundary.dat")

for k2 in K2:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    D = np.loadtxt("resultados/density_0.408000_0.000000.dat")
    P = np.loadtxt("resultados/phase_0.408000_0.000000.dat")
    N = D.shape[0]
    x = np.linspace(-8, 8, N)
    y = np.linspace(-8, 8, N)
    X, Y = np.meshgrid(x, y)
    fig.suptitle(rf"Billar ({xi0},{eta0}) | $k^2$ = {k2} | $\beta$ = 0.00°")

    Dnorm = D / D.max()

    im1 = ax1.pcolormesh(X, Y, Dnorm, shading='auto', cmap='inferno', norm=colors.Normalize(vmin=0, vmax=1))

    ax1.plot(B[:,0], B[:,1], 'w-', lw=1)
    ax1.set_aspect('equal')
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)
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
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_title('Distribución de fase')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    cbar2 = fig.colorbar(
        im2, ax=ax2,
        ticks=[-np.pi, 0, np.pi]
    )
    cbar2.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    cbar2.set_label(r'Fase $\psi$')
    
    # Funcion de actualizacion
    def update(angle):
        D = np.loadtxt(f"resultados/density_{k2:.3f}000_{angle:.2f}0000.dat")
        P = np.loadtxt(f"resultados/phase_{k2:.3f}000_{angle:.2f}0000.dat")

        N = D.shape[0]
        x = y = np.linspace(-8, 8, N)
        X, Y = np.meshgrid(x, y)
        
        Dnorm = D / D.max()
        
        im1.set_array(Dnorm.ravel())
        im2.set_array(P.ravel())
        
        fig.suptitle(f'Billar({xi0},{eta0}) | $k^2$={k2:.3f} | Ángulo: {angle:.1f}°')

        return im1, im2

    ani = animation.FuncAnimation(fig, update, frames=angles, blit=True, interval=50, repeat=False)

    writer = FFMpegWriter(fps=180, metadata=dict(title=rf'Simulacion de Billar $k^2$={k2:.2f}'), codec='mpeg4', extra_args=['-q:v', '2'])
    ani.save(f"Billar_{k2:.2f}.mp4", writer=writer)

    plt.close()
    print(f"Video creado: k2 = {k2}")
    