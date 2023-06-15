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
import numpy as np
from scipy import linalg
from dataclasses import dataclass
import string
from scipy.special import hermite
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _einsum_v1 as einsum
import glob
from PIL import Image
import os
import moviepy.editor as mp


@dataclass
class Atom:
    m: float
    omega: float
    q: float

def amplitude(
    x: tf.Tensor,
    alpha: tf.Tensor,
    num_modes: int,
    cutoff: int
) -> tf.Tensor:
    r'''Computes the probability amplitudes on the quadrature position grid.

    Args:
        x (tf.Tensor): Position quadrature grid.
        alpha (tf.tensor): amplitudes of the state in the Fock tensor product basis.
        num_modes (int): total number of photon modes.
        cutoff (int): cutoff level for single-mode Fock spaces.

    Returns:
        (tf.Tensor): Amplitudes in position basis on the discrete grid `x`.
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
        wave = tf.Variable(h(x)) * tf.exp(-x**2 / 2) / (np.sqrt(np.sqrt(np.pi)*(2**n)*np.math.factorial(n)))
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
    r'''Computes the position probability density.

    Args:
        x (tf.Tensor): Position quadrature grid.
        alpha (tf.tensor): amplitudes of the state in the Fock tensor product basis.
        num_modes (int): total number of photon modes.
        cutoff (int): cutoff level for single-mode Fock spaces.

    Returns:
        (tf.Tensor): Position quadrature probability density on the grid `x`.
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

def pair_density(
    grid: tf.Tensor,
    alpha: tf.Tensor,
    cutoff: int,
    type: str='xx'
) -> tf.Tensor:
    r'''Those joint densities are just used to compute the Bell inequality.
    It also only supports 2 modes (1d model).

    Args:
        grid (tf.Tensor): Position quadrature grid.
        alpha (tf.tensor): amplitudes of the state in the Fock tensor product basis.
        cutoff (int): cutoff level for single-mode Fock spaces.
        type (str): specifies which correlation coefficient to compute.

    Returns:
        (tf.Tensor): expectation values of the product of quadratures.
    '''

    num_points = grid.shape[0]
    alpha = tf.cast(alpha, tf.complex128)

    einsum_rule = ''.join(
        [string.ascii_lowercase[: 2]] \
        + [',' + string.ascii_lowercase[i] + string.ascii_lowercase[2+i] for i in range(2)] \
        + ['->'] + [string.ascii_lowercase[2+i] for i in range(2)]
    )
    pos_hermite_tensor = tf.Variable(np.zeros(shape=(cutoff, num_points)), dtype=tf.complex128)
    mom_hermite_tensor = tf.Variable(np.zeros(shape=(cutoff, num_points)), dtype=tf.complex128)

    for n in range(cutoff):

        h = hermite(n)
        pos_wave = tf.Variable(h(grid)) * tf.exp(-grid**2 / 2) / (np.sqrt(np.sqrt(np.pi)*(2**n)*np.math.factorial(n)))
        pos_wave = tf.cast(pos_wave, tf.complex128)
        mom_wave = (-1j)**n * pos_wave
        pos_hermite_tensor[n].assign(pos_wave)
        mom_hermite_tensor[n].assign(mom_wave)

    if type == 'xx':
        amp = einsum(einsum_rule, alpha, pos_hermite_tensor, pos_hermite_tensor)
    elif type == 'xp':
        amp = einsum(einsum_rule, alpha, pos_hermite_tensor, mom_hermite_tensor)
    elif type == 'px':
        amp = einsum(einsum_rule, alpha, mom_hermite_tensor, pos_hermite_tensor)
    elif type == 'pp':
        amp = einsum(einsum_rule, alpha, mom_hermite_tensor, mom_hermite_tensor)

    density = tf.abs(amp)**2

    return density

def bell(
    grid: tf.Tensor,
    alpha: tf.Tensor,
    cutoff: int
) -> float:
    r'''Difference between the two sides of the Bell-type
    inequality, as defined in arXiv:1005.2208.

    Args:
        grid (tf.Tensor): Position quadrature grid.
        alpha (tf.tensor): amplitudes of the state in the Fock tensor product basis.
        cutoff (int): cutoff level for single-mode Fock spaces.

    Returns:
        (float): Continuous-variable Bell inequality.
    '''

    xx_density = pair_density(grid, alpha, cutoff, 'xx')
    xp_density = pair_density(grid, alpha, cutoff, 'xp')
    px_density = pair_density(grid, alpha, cutoff, 'px')
    pp_density = pair_density(grid, alpha, cutoff, 'pp')

    dx = grid[1] - grid[0]

    temp1 = np.einsum('ab,a,b->', dx**2 * xx_density, grid**2, grid**2)
    temp2 = np.einsum('ab,a,b->', dx**2 * xp_density, grid**2, grid**2)
    temp3 = np.einsum('ab,a,b->', dx**2 * px_density, grid**2, grid**2)
    temp4 = np.einsum('ab,a,b->', dx**2 * pp_density, grid**2, grid**2)

    lhs = temp1 + temp2 + temp3 + temp4

    temp1 = np.einsum('ab,a,b->', dx**2 * xx_density, grid, grid)
    temp2 = np.einsum('ab,a,b->', dx**2 * xp_density, grid, grid)
    temp3 = np.einsum('ab,a,b->', dx**2 * px_density, grid, grid)
    temp4 = np.einsum('ab,a,b->', dx**2 * pp_density, grid, grid)

    rhs = np.abs(temp1 - temp4 + 1j * (temp2 + temp3))**2

    return (lhs - rhs).item()

def correlation_quadratures(
    x: tf.Tensor,
    states: tf.Tensor,
    cutoff: int
) -> np.ndarray:
    r'''

    Args:
        x (tf.Tensor): Position quadrature grid.
        states (tf.tensor): amplitudes of the states in the Fock tensor product basis.
        cutoff (int): cutoff level for single-mode Fock spaces.

    Returns:
        (np.ndarray): Correlation coefficient of the single-mode position quadratures.
    '''

    dx = x[1] - x[0]

    corr_array = np.zeros(shape=(states.shape[0], states.shape[1]))

    for i in range(corr_array.shape[0]):
        for j in range(corr_array.shape[1]):

            alpha = states[i, j]

            density = quadratures_density(
                x=x,
                alpha=alpha,
                num_modes=2,
                cutoff=cutoff
            )

            marginals = marginal_densities(
                rho=density,
                dx=dx
            )

            marginal1 = marginals[0]
            marginal2 = marginals[1]

            mu1 = np.einsum('a,a->', dx * marginal1, x)
            mu2 = np.einsum('a,a->', dx * marginal2, x)
            sigma1 = np.einsum('a,a->', dx * marginal1, x**2)**0.5
            sigma2 = np.einsum('a,a->', dx * marginal2, x**2)**0.5
            cov = np.einsum('ab,a,b->', dx**2 * density, x, x)

            corr = (cov - mu1 * mu2) / (sigma1 * sigma2)

            corr_array[i, j] = corr

    return corr_array

def von_neumann_entropy(states: np.ndarray) -> np.ndarray:
    r""" Computes the von neumann entropy of a the partial density matrix
    of the first subsystem of the total system described by state `alpha`.
    Note that this function does not support more than a two-mode system for now.

    Args:
        states (np.ndarray): The coefficients of the state of the total system expressed in the Fock basis.

    Returns:
        (np.ndarray): The von neumann entropy of the first subsystem.
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

def renyi_entropy(
    states: np.ndarray,
    n: int
) -> np.ndarray:
    r""" Computes the renyi entropy $\frac{1}{1-n} \log_2 (Tr(\rho^n))$
    of a the partial density matrix of the first subsystem of the total system
    described by state `alpha`.
    Note that this function does not support more than a two-mode system for now.

    Args:
        states (np.ndarray): The coefficients of the state of the total system expressed in the Fock basis.
        n (int): Integer entering in the definition of the renyi entropy.

    Returns:
        (np.ndarray): The renyi entropy of the first subsystem.
    """

    # Let us compute the partial density matrix of the first
    # subsystem, expressed in the Fock basis
    rho = np.einsum('abml,abnl->abnm', states.conjugate(), states)

    # n-th power of the partial density matrix
    einsum_rule = ','.join([string.ascii_lowercase[n+1] + string.ascii_lowercase[n+2] + string.ascii_lowercase[i: i+2] for i in range(n)]) + '->' + string.ascii_lowercase[n+1] + string.ascii_lowercase[n+2] + string.ascii_lowercase[0] + string.ascii_lowercase[n]
    rho_n = np.einsum(einsum_rule, *(rho for _ in range(n)))

    # trace of the n-th power
    tr_rho_n = np.einsum('abcc->ab', rho_n)

    # Entrywise log
    log_tr_rho_n = np.log2(tr_rho_n)

    return (1 / (n - 1)) * log_tr_rho_n

# Plotting functions

def make_gif(
    frames_dir: str,
    duration: int
) -> None:
    r'''
    Args:
        frames_dir (str): path of the folder containing the images to stack.
        duration (int): total duration of the video in seconds.

    Returns:
        None
    '''

    frames = [Image.open(image) for image in sorted(glob.glob(os.path.join(frames_dir, '*.png')))][::-1]
    frame_one = frames[0]

    frame_one.save(
        fp=os.path.join(frames_dir, 'animation.gif'),
        format="GIF",
        optimize=False,
        append_images=frames,
        save_all=True,
        duration=duration*1000/len(frames),
        loop=0
    )

    clip = mp.VideoFileClip(os.path.join(frames_dir, 'animation.gif'))
    clip.write_videofile(os.path.join(frames_dir, 'animation.mp4'))