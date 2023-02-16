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

import string
from scipy import linalg
from math import log
import numpy as np
import os
import tensorflow as tf

from src.utils import amplitude

class PostProcessor():

    def __init__(
        self,
        log_dir: str,
        quadrature_grid
    ) -> None:

        self.log_dir = log_dir

        self.data = np.load(os.path.join(log_dir, 'results.npy'))

        self.state = self.data[:-1, :-1]
        self.energy = self.data[-1, -1]

        self.x_grid = quadrature_grid
        self.dx = self.x_grid[1] - self.x_grid[0]

        self.cutoff = self.state.shape[0]

        self.joint_density = None
        self.marginals = None
        self.entropy = None

    def quadratures_density(self) -> None:

        alpha = tf.Tensor(self.state)
        x = tf.Tensor(self.x_grid)
        amp = amplitude(x, alpha, 2, self.cutoff)
        density = tf.abs(amp)**2

        self.joint_density = density.numpy()

    def marginal_densities(self) -> None:
        r'''Given the joint density of the quadratures, compute the list of the marginals.

        Args:
            rho (np.ndarray): joint density of the position quadratures
            dx (float): step used in the position quadrature grid.
            Necessary to compute discretized integrals.

        Returns:
            (np.ndarray): Collection of the marginals for each mode.
            The returned array is of shape (num_modes, size_of_x_grid).
        '''

        if self.joint_density == None:
            self.joint_density()

        marginals = []
        num_modes = len(self.joint_density.shape)

        for mode in range(num_modes):
            einsum_rule = ''.join([string.ascii_lowercase[: num_modes]] + ['->'] + [string.ascii_lowercase[mode]])
            marginal = np.einsum(einsum_rule, self.dx**(num_modes - 1) * self.joint_density)
            marginals.append(marginal)

        self.marginals = np.array(marginals)

    def von_neumann_entropy(self) -> None:
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
        rho = np.einsum('ml,nl->nm', self.state.conjugate(), self.state)

        # We finally compute the von Neumann entropy (log base 2)
        self.entropy = - (1 / log(2)) * np.trace(rho @ linalg.logm(rho))
