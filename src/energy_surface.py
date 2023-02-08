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
from src.utils import plot_potential_energy_surface, Atom
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
        model: str,
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
            model (str): One of the nine models defined in the paper ('11', '12', ..., '33')
            atoms (List[Atom]): List of atoms, characterized by their mass, frequency and charge.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.
            learning_rate (float): The learning rate for the tensorflow optimizer.
            save_dir (str): Directory where to save the logs.

        Returns:
            None
        """

        self.dimension = 3 if model[1]=='3' else 1
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance_list = distance_list
        self.model = model
        self.atoms = atoms
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.learning_rate = learning_rate
        self.x_quadrature_grid = x_quadrature_grid
        self.verbose = verbose

        self.save_dir = save_dir
        self.save_dir_energy_surface = os.path.join(save_dir, 'energy_surface')
        self.save_dir_states = os.path.join(save_dir, 'states')
        self.save_dir_quad_density = os.path.join(save_dir, 'quad_density')
        self.save_dir_quad_marginals = os.path.join(save_dir, 'quad_marginals')

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_energy_surface, exist_ok=True)
        os.makedirs(self.save_dir_states, exist_ok=True)
        os.makedirs(self.save_dir_quad_density, exist_ok=True)
        os.makedirs(self.save_dir_quad_marginals, exist_ok=True)

    def construct_energy_surface(
        self,
        epsilon=1e-3,
        alpha=0.95,
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

        for i in range(len(self.distance_list)):

            print('Distance {}/{}'.format(i+1, len(self.distance_list)))

            # Instanciate a VQE object
            vqe = VQE(
                layers=self.layers,
                distance=self.distance_list[i],
                model=self.model,
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
            energy_surface.append(vqe.best_loss - energy_free)

            self.save_logs(
                energy_surface=energy_surface,
                state=vqe.state.ket(),
                density=vqe.density,
                marginals=vqe.marginals,
                distance='{:.4f}'.format(self.distance_list[i])
            )

    def save_logs(
        self,
        energy_surface: list,
        state: np.ndarray,
        density: np.ndarray,
        marginals: np.ndarray,
        distance: str
    ) -> None:
        r"""Saves various logs for post analysis.

        The logs are:
            distance_list (np.ndarray): shape (num_distances,)
            energy_surface (np.ndarray): shape (num_distances,)
            states (np.ndarray): shape (num_distances, fock_cutoff, ..., fock_cutoff)   (depending on the number of modes, 2 in 1d, 6 in 3d)
            densities (np.ndarray): shape (num_distances, x_grid_size, ..., x_grid_size)
            marginals (np.ndarray): shape (num_distances, num_modes, x_grid_size)
        """

        distance_list = np.array(self.distance_list)
        energy_surface = np.array(energy_surface)

        np.save(os.path.join(self.save_dir_energy_surface, 'distance_list'), distance_list)
        np.save(os.path.join(self.save_dir_energy_surface, 'energy_surface'), energy_surface)
        np.save(os.path.join(self.save_dir_states, 'state_d={}'.format(distance)), state)
        np.save(os.path.join(self.save_dir_quad_density, 'density_d={}'.format(distance)), density)
        np.save(os.path.join(self.save_dir_quad_marginals, 'marginals_d={}'.format(distance)), marginals)

        plot_potential_energy_surface(
            distance_array=distance_list[:energy_surface.shape[0]],
            binding_energy_array=energy_surface,
            save_path=os.path.join(self.save_dir_energy_surface, 'binding_energy_plot')
        )

