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
        order: str,
        model: str,
        atoms: List[Atom] = [],
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        learning_rate: float = 0.001,
        save_dir: str = 'logs/'
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
        self.order = order
        self.model = model
        self.atoms = atoms
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.save_dir = save_dir
        self.learning_rate = learning_rate

        self.energy_surface = []

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

        for i in range(len(self.distance_list)):

            print('Distance {}/{}'.format(i+1, len(self.distance_list)))

            # Instanciate a VQE object
            vqe = VQE(
                layers=self.layers,
                distance=self.distance_list[i],
                order=self.order,
                model=self.model,
                atoms=self.atoms,
                active_sd=self.active_sd,
                passive_sd=self.passive_sd,
                cutoff_dim=self.cutoff_dim,
                learning_rate=self.learning_rate,
                save_dir=self.save_dir
            )

            # Run the VQE algorithm
            vqe.train(
                epsilon=epsilon,
                alpha=alpha,
                patience=patience
            )

            # Compute the ground state energy of the uninteracting system
            # to substract out to get the binding energy
            energy_free = 0.5 * self.dimension * HBAR * (omega1 + omega2)

            # Append the obtained binding energy to the list
            self.energy_surface.append(vqe.best_loss - energy_free)

    def save_logs(self) -> None:
        """
        Save the distance list and energy surface to the specified directory.

        Parameters:
            save_dir (str): The directory where the distance list and energy surface should be saved.
        """

        save_dir = os.path.join(self.save_dir, 'energy_surface')
        os.makedirs(save_dir, exist_ok=True)

        distance_list = np.array(self.distance_list)
        energy_surface = np.array(self.energy_surface)

        np.save(os.path.join(save_dir, 'distance_list'), distance_list)
        np.save(os.path.join(save_dir, 'energy_surface'), energy_surface)

        plot_potential_energy_surface(
            distance_array=distance_list,
            binding_energy_array=energy_surface,
            save_path=os.path.join(save_dir, 'binding_energy')
        )

