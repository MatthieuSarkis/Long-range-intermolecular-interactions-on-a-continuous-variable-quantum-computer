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

class EnergySurface():

    def __init__(
        self,
        modes: int,
        layers: int,
        distance_list: List[float],
        order: str,
        direction: str,
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        epochs: int = 100
    ) -> None:
        """Constructor of the EnergySurface class.
        Takes a list of distances between a pair of QDOs
        and performs the loop over the distances.

        Args:
            modes (int): The number of modes in the quantum neural network.
            layers (int): The number of layers in the quantum neural network.
            distance_list (List[float]): List of distances between the two QDOs.
            order (str): Order in the multipolar expansion: `quadratic`, `quartic` or `full`.
            direction (str): Axis along which the electrons move: parallel or perpendicular.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.
            epochs (int): Number of epochs for the training procedure.

        Returns:
            None
        """

        self.modes = modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance_list = distance_list
        self.order = order
        self.direction = direction
        self.epochs = epochs
        self.active_sd = active_sd
        self.passive_sd = passive_sd

        self.energy_surface = []

    def construct_energy_surface(self) -> None:
        """
        Calculate the energy surface of the system by training a Variational Quantum Eigensolver (VQE) model
        for each distance in `self.distance_list`. The energy surface is stored in `self.energy_surface`.
        """

        for i in range(len(self.distance_list)):

            vqe = VQE(
                modes=self.modes,
                layers=self.layers,
                distance=self.distance_list[i],
                order=self.order,
                direction=self.direction,
                active_sd=self.active_sd,
                passive_sd=self.passive_sd,
                cutoff_dim=self.cutoff_dim
            )

            vqe.train(epochs=self.epochs)

            self.energy_surface.append(vqe.best_loss - 1.0)
            # The -1.0 simply corresponds to the removal of
            # the ground state energy of a pair of free quantum
            # harmonic oscillators.

    def save_logs(
        self,
        save_dir: str
    ) -> None:
        """
        Save the distance list and energy surface to the specified directory.

        Parameters:
            save_dir (str): The directory where the distance list and energy surface should be saved.
        """

        distance_list = np.array(self.distance_list)
        energy_surface = np.array(self.energy_surface)

        np.save(os.path.join(save_dir, 'distance_list'), distance_list)
        np.save(os.path.join(save_dir, 'energy_surface'), energy_surface)

