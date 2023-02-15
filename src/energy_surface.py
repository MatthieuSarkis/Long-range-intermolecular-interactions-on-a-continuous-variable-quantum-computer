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

import os
import numpy as np
from typing import List

from src.vqe import VQE
from src.utils import plot_potential_energy_surface, Atom, plot_entropy
from src.constants import HBAR

class EnergySurface():
    r"""
    This class is a simple wrapper over the VQE class.
    Just here to facilitate the generation of binding curves.
    """

    def __init__(
        self,
        layers: int,
        distance_list: List[float],
        theta_list: List[float],
        x_quadrature_grid: np.ndarray,
        atoms: List[Atom] = [],
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        learning_rate: float = 0.001,
        save_dir: str = 'logs/',
        verbose: bool = True
    ) -> None:
        """Constructor of the EnergySurface class.
        Takes a list of distances between a pair of QDOs
        and performs the loop over the distances.

        Args:
            layers (int): The number of layers in the quantum neural network.
            distance_list (List[float]): List of distances between the two QDOs.
            order (str): Order in the multipolar expansion: `quadratic`, `quartic` or `full`.
            theta_list (List[float]): List of angle, one per model.
            atoms (List[Atom]): List of atoms, characterized by their mass, frequency and charge.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.
            learning_rate (float): The learning rate for the tensorflow optimizer.
            save_dir (str): Directory where to save the logs.

        Returns:
            None
        """

        self.dimension = 1
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance_list = distance_list
        self.theta_list = theta_list
        self.atoms = atoms
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.learning_rate = learning_rate
        self.x_quadrature_grid = x_quadrature_grid
        self.verbose = verbose

        self.save_dir = save_dir

    def construct_energy_surface(
        self,
        epsilon=1e-3,
        patience=20
    ) -> None:
        """
        Calculate the energy surface of the system by training a Variational Quantum Eigensolver (VQE) model
        for each distance in `self.distance_list`. The energy surface is stored in `self.energy_surface`.

        Args:
            epsilon (float): Tolerance for the training loop stopping criterium.
            alpha (float): Rate for the moving average in the training loop.
            patience (int): Impose lack of improvement in loss for at least that number of epochs.

        Returns:
            None
        """

        # Read out the frequency of the two QDOs to compute
        # the ground state energy of the uninteracting system.
        omega1 = self.atoms[0].omega
        omega2 = self.atoms[1].omega

        energy_surface = []
        entanglement_entropy = []

        for j in range(len(self.theta_list)):

            energy_surface_theta = []
            entanglement_entropy_theta = []

            for i in range(len(self.distance_list)):

                print('Distance {}/{}'.format(i+1, len(self.distance_list)))

                # Instanciate a VQE object
                vqe = VQE(
                    layers=self.layers,
                    distance=self.distance_list[i],
                    theta=self.theta_list[j],
                    x_quadrature_grid=self.x_quadrature_grid,
                    atoms=self.atoms,
                    active_sd=self.active_sd,
                    passive_sd=self.passive_sd,
                    cutoff_dim=self.cutoff_dim,
                    learning_rate=self.learning_rate,
                    save_dir=self.save_dir,
                    verbose=self.verbose
                )

                # Run the VQE algorithm
                vqe.train(
                    epsilon=epsilon,
                    patience=patience
                )

                # Compute the ground state energy of the uninteracting system
                # to substract out to get the binding energy
                energy_free = 0.5 * self.dimension * HBAR * (omega1 + omega2)

                # Append the obtained binding energy to the list
                energy_surface_theta.append(vqe.best_loss - energy_free)

                entanglement_entropy_theta.append(vqe.partial_entropy)

                self.save_logs(
                    theta=self.theta_list[j],
                    energy_surface=energy_surface_theta,
                    state=vqe.state.ket(),
                    density=vqe.density,
                    marginals=vqe.marginals,
                    entanglement_entropy=entanglement_entropy_theta,
                    distance='{:.4f}'.format(self.distance_list[i])
                )

            energy_surface.append(energy_surface_theta)
            entanglement_entropy.append(entanglement_entropy_theta)

    def save_logs(
        self,
        theta: float,
        energy_surface: list,
        state: np.ndarray,
        density: np.ndarray,
        marginals: np.ndarray,
        entanglement_entropy: list,
        distance: str
    ) -> None:
        r"""Saves various logs for post analysis.

        The logs are:
            distance_list (np.ndarray): shape (num_distances,)
            energy_surface (list): shape (num_distances,)
            states (np.ndarray): shape (num_distances, fock_cutoff, ..., fock_cutoff)   (depending on the number of modes, 2 in 1d, 6 in 3d)
            densities (np.ndarray): shape (num_distances, x_grid_size, ..., x_grid_size)
            marginals (np.ndarray): shape (num_distances, num_modes, x_grid_size)
            entanglement_entropy (list): shape (num_distances,)
        """

        save_dir_energy_surface = os.path.join(self.save_dir, 'theta={:.4f}'.format(theta), 'energy_surface')
        save_dir_states = os.path.join(self.save_dir, 'theta={:.4f}'.format(theta), 'states')
        save_dir_quad_density = os.path.join(self.save_dir, 'theta={:.4f}'.format(theta), 'quad_density')
        save_dir_quad_marginals = os.path.join(self.save_dir, 'theta={:.4f}'.format(theta), 'quad_marginals')
        save_dir_entropy = os.path.join(self.save_dir, 'theta={:.4f}'.format(theta), 'entropy')

        os.makedirs(save_dir_energy_surface, exist_ok=True)
        os.makedirs(save_dir_states, exist_ok=True)
        os.makedirs(save_dir_quad_density, exist_ok=True)
        os.makedirs(save_dir_quad_marginals, exist_ok=True)
        os.makedirs(save_dir_entropy, exist_ok=True)

        distance_list = np.array(self.distance_list)
        energy_surface = np.array(energy_surface)
        entanglement_entropy = np.array(entanglement_entropy)

        np.save(os.path.join(save_dir_energy_surface, 'distance_list'), distance_list)
        np.save(os.path.join(save_dir_energy_surface, 'energy_surface'), energy_surface)
        np.save(os.path.join(save_dir_states, 'distance={}'.format(distance)), state)
        np.save(os.path.join(save_dir_quad_density, 'distance={}'.format(distance)), density)
        np.save(os.path.join(save_dir_quad_marginals, 'distance={}'.format(distance)), marginals)
        np.save(os.path.join(save_dir_entropy, 'entanglement_entropy'), entanglement_entropy)

        plot_potential_energy_surface(
            distance_array=distance_list[:energy_surface.shape[0]],
            theta=theta,
            binding_energy_array=energy_surface,
            save_path=os.path.join(save_dir_energy_surface, 'binding_energy_plot')
        )

        plot_entropy(
            distance_array=distance_list[:energy_surface.shape[0]],
            theta=theta,
            entropy_array=entanglement_entropy,
            save_path=os.path.join(save_dir_entropy, 'entropy_plot')
        )

