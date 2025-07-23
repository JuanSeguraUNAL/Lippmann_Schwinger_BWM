import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("paper")
sns.set_palette("colorblind")

def main():
    datos = np.loadtxt("tiempos_OMP_threads.txt")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(datos[:, 0], datos[:, 1], marker='o', ms=6, mec='black')
    ax.set_xlabel("Número de threads", fontsize=15)
    ax.set_ylabel("Wall time (seg)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"Tiempos_OMP.pdf")

main()