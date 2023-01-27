# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import matplotlib.pyplot as plt
import numpy as np
from strawberryfields.backends.tfbackend.states import FockStateTF
from typing import List
from dataclasses import dataclass
import string
from scipy.special import hermite
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _einsum_v1 as einsum

@dataclass
class Atom:
    m: float
    omega: float
    q: float


def plot_partial_wigner_function(
    state: FockStateTF,
    mode: int
) -> None:
    """
    Plots the Wigner function of a given quantum state in a specific mode.

    Parameters:
    state (object): The quantum state whose Wigner function is to be plotted.
    mode (int): The mode in which the Wigner function is to be plotted.
    """

    fig = plt.figure()
    X = np.linspace(-5, 5, 100)
    P = np.linspace(-5, 5, 100)
    Z = state.wigner(mode, X, P)
    X, P = np.meshgrid(X, P)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
    fig.set_size_inches(4.8, 5)
    ax.set_axis_off()
    plt.show()

def plot_loss_history(
    loss_history: List[float],
    save_path: str
) -> None:
    """
    Plots the evolution of the loss over time.

    Parameters:
    loss_history (list): A list of floats representing the loss at each epoch.
    save_path (str): The path where the plot will be saved.
    """

    plt.style.use('./src/plots.mplstyle')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(loss_history)
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.grid(True)
    axes.set_title('Evolution of the loss')
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')

def plot_potential_energy_surface(
    distance_array: np.ndarray,
    binding_energy_array: np.ndarray,
    save_path: str
) -> None:
    """
    Plot the potential energy surface for a system based on the given distance array and binding energy array.

    Parameters
    ----------
    distance_array : np.ndarray
        An array of interatomic distances.
    binding_energy_array : np.ndarray
        An array of binding energies corresponding to the interatomic distances in `distance_array`.
    save_path : str
        The file path where the plot should be saved.

    Returns
    -------
    None
    """

    plt.style.use('./src/plots.mplstyle')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(distance_array, binding_energy_array)
    axes.set_xlabel('Interatomic distance')
    axes.set_ylabel('Binding energy')
    axes.grid(True)
    axes.set_title('Potential energy surface')
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')

def amplitude(
    x: tf.Tensor,
    alpha: tf.Tensor,
    num_modes: int,
    cutoff: int
) -> tf.Tensor:

    num_points = x.shape[0]
    einsum_rule = ''.join(
        [string.ascii_lowercase[: num_modes]] \
        + [',' + string.ascii_lowercase[i] + string.ascii_lowercase[num_modes+i] for i in range(num_modes)] \
        + ['->'] + [string.ascii_lowercase[num_modes+i] for i in range(num_modes)]
    )
    hermite_tensor = tf.Variable(np.zeros(shape=(cutoff, num_points)), dtype=tf.complex128)

    for n in range(cutoff):

        h = hermite(n)
        wave = tf.Variable(h(x)) * tf.exp(-x**2 / 2)/ (np.sqrt(np.sqrt(np.pi)*(2**n)*np.math.factorial(n)))
        wave = tf.cast(wave, tf.complex128)
        hermite_tensor[n].assign(wave)

    amp = einsum(einsum_rule, alpha, *(hermite_tensor for _ in range(num_modes)))

    return amp

def quadratures_density(
    x: tf.Tensor,
    alpha: tf.Tensor,
    num_modes: int,
    cutoff: int
) -> tf.Tensor:

    amp = amplitude(x, alpha, num_modes, cutoff)
    density = tf.abs(amp)**2

    return density
