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

from math import log
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from strawberryfields.backends.tfbackend.states import FockStateTF
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
    state: np.ndarray,
    mode: int
) -> None:
    """
    Plots the Wigner function of a given quantum state in a specific mode.

    Parameters:
    state (object): The quantum state whose Wigner function is to be plotted.
    mode (int): The mode in which the Wigner function is to be plotted.
    """

    state = FockStateTF(
        state_data=state,
        num_modes=len(state.shape),
        pure=True,
        cutoff_dim=state.shape[0]
    )

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


def plot_potential_energy_surface(
    distance_array: np.ndarray,
    theta: float,
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
    axes.set_title('Potential energy surface, theta={:.4f}'.format(theta))
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

def plot_entropy(
    distance_array: np.ndarray,
    theta: float,
    entropy_array: np.ndarray,
    save_path: str
) -> None:
    """
    Plot the potential energy surface for a system based on the given distance array and binding energy array.

    Parameters
    ----------
    distance_array : np.ndarray
        An array of interatomic distances.
    entropy_array : np.ndarray
        An array of entropy values.
    save_path : str
        The file path where the plot should be saved.

    Returns
    -------
    None
    """

    plt.style.use('./src/plots.mplstyle')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(distance_array, entropy_array)
    axes.set_xlabel('Interatomic distance')
    axes.set_ylabel('Entanglement entropy')
    axes.grid(True)
    axes.set_title('Entanglement entropy, theta={:.4f}'.format(theta))
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

def amplitude(
    x: tf.Tensor,
    alpha: tf.Tensor,
    num_modes: int,
    cutoff: int
) -> tf.Tensor:
    r'''
    '''

    num_points = x.shape[0]
    alpha = tf.cast(alpha, tf.complex128)

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
    r'''
    '''

    amp = amplitude(x, alpha, num_modes, cutoff)
    density = tf.abs(amp)**2

    return density

def marginal_densities(
    rho: np.ndarray,
    dx: float
) -> np.ndarray:
    r'''Given the joint density of the quadratures, compute the list of the marginals.

    Args:
        rho (np.ndarray): joint density of the position quadratures
        dx (float): step used in the position quadrature grid.
        Necessary to compute discretized integrals.

    Returns:
        (np.ndarray): Collection of the marginals for each mode.
        The returned array is of shape (num_modes, size_of_x_grid).
    '''

    marginals = []
    num_modes = len(rho.shape)

    for mode in range(num_modes):
        einsum_rule = ''.join([string.ascii_lowercase[: num_modes]] + ['->'] + [string.ascii_lowercase[mode]])
        marginal = np.einsum(einsum_rule, dx**(num_modes - 1) * rho)
        marginals.append(marginal)

    marginals = np.array(marginals)

    return marginals

#def von_neumann_entropy(alpha: np.ndarray) -> np.ndarray:
#    r""" Computes the von neumann entropy of a the partial density matrix
#    of the first subsystem of the total system described by state `alpha`.
#    Note that this function does not support more than a two-mode system for now.
#
#    Args:
#        alpha (np.ndarray): The coefficients of the state of the total system expressed in the Fock basis.
#    Returns:
#        (float): The von neumann entropy of the first subsystem.
#    """
#
#    # Let us compute the partial density matrix of the first
#    # subsystem, expressed in the Fock basis
#    rho = np.einsum('ml,nl->nm', alpha.conjugate(), alpha)
#
#    # We finally compute the von Neumann entropy (log base 2)
#    entropy = - (1 / log(2)) * np.trace(rho @ linalg.logm(rho))
#
#    return entropy.item()

def von_neumann_entropy(states: np.ndarray) -> np.ndarray:
    r""" Computes the von neumann entropy of a the partial density matrix
    of the first subsystem of the total system described by state `alpha`.
    Note that this function does not support more than a two-mode system for now.

    Args:
        alpha (np.ndarray): The coefficients of the state of the total system expressed in the Fock basis.
    Returns:
        (float): The von neumann entropy of the first subsystem.
    """

    # Let us compute the partial density matrix of the first
    # subsystem, expressed in the Fock basis
    rho = np.einsum('abml,abnl->abnm', states.conjugate(), states)

    entropy_array = np.zeros(shape=(states.shape[0], states.shape[1]))

    # Unfortunately the log of a matrix is computed with `linalg.logm`,
    # which doesn't have a vectorized implementation, so one should loop.
    for i in range(entropy_array.shape[0]):
        for j in range(entropy_array.shape[1]):

            # We finally compute the von Neumann entropy (log base 2)
            entropy = - (1 / log(2)) * np.trace(rho[i, j] @ linalg.logm(rho[i, j]))
            entropy_array[i, j] = entropy

    return entropy_array