import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set()
sns.set_context("paper")
sns.set_palette("colorblind")

def main():
    cadena_opt = sys.argv[1]
    datCpp = np.loadtxt("tiempos_cpp.txt")
    datCuda = np.loadtxt("tiempos_cuda.txt")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(datCpp[:, 0], datCpp[:, 1], label="C++", marker='o', ms=6, mec='black')
    ax.plot(datCuda[:, 0], datCuda[:, 1], label="CUDA", marker='o', ms=6, mec='black')
    ax.set_xlabel("Tamaño de la malla", fontsize=15)
    ax.set_ylabel("Wall time (seg)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"Tiempos_{cadena_opt}.pdf")

main()