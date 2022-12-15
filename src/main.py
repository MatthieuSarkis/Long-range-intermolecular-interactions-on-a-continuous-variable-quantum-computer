import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.vqe import VQE

tf.random.set_seed(137)
np.random.seed(137)

def main():

    modes = 2
    layers = 8
    cutoff_dim = 6
    active_sd=0.0001
    passive_sd=0.1
    epochs = 10

    vqe = VQE(modes=modes, layers=layers, active_sd=active_sd, passive_sd=passive_sd, cutoff_dim=cutoff_dim)
    vqe.train(epochs=epochs)

    fig = plt.figure()
    X = np.linspace(-5, 5, 100)
    P = np.linspace(-5, 5, 100)
    Z = vqe.state.wigner(0, X, P)
    X, P = np.meshgrid(X, P)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
    fig.set_size_inches(4.8, 5)
    ax.set_axis_off()
    plt.show()

main()