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

class EnergySurface():

    def __init__(
        self,
        layers: int,
        distance_list: List[float],
        order: str,
        direction: str,
        dimension: int,
        atoms: List[Atom] = [],
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        epochs: int = 100,
        save_dir: str = 'logs/'
    ) -> None:
        """Constructor of the EnergySurface class.
        Takes a list of distances between a pair of QDOs
        and performs the loop over the distances.

        Args:
            layers (int): The number of layers in the quantum neural network.
            distance_list (List[float]): List of distances between the two QDOs.
            order (str): Order in the multipolar expansion: `quadratic`, `quartic` or `full`.
            direction (str): Axis along which the electrons move: parallel or perpendicular.
            dimension (int): Dimension of space (1d or 3d).
            atoms (List[Atom]): List of atoms, characterized by their mass, frequency and charge.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.
            epochs (int): Number of epochs for the training procedure.
            save_dir (str): Directory where to save the logs.

        Returns:
            None
        """

        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance_list = distance_list
        self.order = order
        self.direction = direction
        self.dimension = dimension
        self.atoms = atoms
        self.epochs = epochs
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.save_dir = save_dir

        self.energy_surface = []

    def construct_energy_surface(self) -> None:
        """
        Calculate the energy surface of the system by training a Variational Quantum Eigensolver (VQE) model
        for each distance in `self.distance_list`. The energy surface is stored in `self.energy_surface`.
        """

        for i in range(len(self.distance_list)):

            print('Distance {}/{}'.format(i+1, len(self.distance_list)))

            vqe = VQE(
                layers=self.layers,
                distance=self.distance_list[i],
                order=self.order,
                direction=self.direction,
                dimension=self.dimension,
                atoms=self.atoms,
                active_sd=self.active_sd,
                passive_sd=self.passive_sd,
                cutoff_dim=self.cutoff_dim,
                save_dir=self.save_dir
            )

            vqe.train(epochs=self.epochs)

            self.energy_surface.append(vqe.best_loss - 0.5 * len(self.atoms) * self.dimension)
            # The shift simply corresponds to the removal of
            # the ground state energy of a pair of free quantum
            # harmonic oscillators.

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

