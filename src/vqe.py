# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
# The code contains modification of code by Xanadu that can be found at
# https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import os
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.backends.tfbackend.states import FockStateTF
from typing import List

from src.utils import plot_loss_history, Atom, quadratures_density
from src.circuit import Circuit
from src.constants import XMIN, XMAX, NUM_POINTS

class VQE():

    def __init__(
        self,
        layers: int,
        distance: float,
        order: str,
        model: str,
        atoms: List[Atom],
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        save_dir: str = 'logs/'
    ) -> None:
        """
        Initializes a new instance of the QuantumNeuralNetwork class.

        Args:
            layers (int): The number of layers in the quantum neural network.
            distance (float): Distance between the two QDOs.
            order (str): Order in the multipolar expansion: `quadratic`, `quartic` or `full`.
            model (str): One of the nine models defined in the paper ('11', '12', ..., '33')
            atoms (List[Atom]): List of atoms, characterized by their mass, frequency and charge.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.
            save_dir (str): Directory where to save the logs.

        Returns:
            None
        """

        self.dimension = 3 if model[1]=='3' else 1
        self.modes = len(atoms) * self.dimension
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance = distance
        self.order = order
        self.model = model
        self.atoms = atoms
        self.save_dir = save_dir

        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        self.qnn = sf.Program(self.modes)

        self.weights = self.init_weights(active_sd=active_sd, passive_sd=passive_sd) # our TensorFlow weights
        num_params = np.prod(self.weights.shape)   # total number of parameters in our model

        self.sf_params = np.arange(num_params).reshape(self.weights.shape).astype(np.str)
        self.sf_params = np.array([self.qnn.params(*i) for i in self.sf_params])

        self.circuit = Circuit(self.qnn, self.sf_params, self.layers)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_history = None
        self.best_loss = None
        self.state = None

    def init_weights(
        self,
        active_sd: float = 0.0001,
        passive_sd: float = 0.1
    ) -> None:
        """Initialize a 2D TensorFlow Variable containing normally-distributed
        random weights for an ``N`` mode quantum neural network with ``L`` layers.

        Args:
            active_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the active parameters
                (displacement, squeezing, and Kerr magnitude)
            passive_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the passive parameters
                (beamsplitter angles and all gate phases)

        Returns:
            tf.Variable[tf.float32]: A TensorFlow Variable of shape
            ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
            row represents the layer parameters for the Lth layer.
        """

        # Number of interferometer parameters:
        M = int(self.modes * (self.modes - 1)) + max(1, self.modes - 1)

        # Create the TensorFlow variables
        int1_weights = tf.random.normal(shape=[self.layers, M], stddev=passive_sd)
        s_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)
        int2_weights = tf.random.normal(shape=[self.layers, M], stddev=passive_sd)
        dr_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)
        dp_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=passive_sd)
        k_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)

        weights = tf.concat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], axis=1
        )

        weights = tf.Variable(weights)

        return weights

    def cost(
        self,
        state: FockStateTF
    ) -> tf.Tensor:
        """
        Calculates the cost of a given Fock state using the Hamiltonian function.
        We treat either the full Coulomb potential Hamiltonian, or the Hamiltonian
        at some fixed order in the multipolar expansion.
        In all cases we treat the one-dimensional toy model, in which the electrons
        are constrained to move along an axis which is either parallel or perpendicular
        to the axis on which the nuclei are sitting.
        This implies that we therefore assume the nuclei are all sitting along a common axis,
        that we take to be the z-axis in a Cartesian basis.

        Args:
            state (FockStateTF): The Fock state for which to calculate the cost.

        Returns:
            tf.Tensor: The cost of the given Fock state.
        """

        x = tf.linspace(XMIN, XMAX, NUM_POINTS)
        dx = (x[1] - x[0]).numpy()
        x = tf.cast(x, tf.double)

        density = quadratures_density(
            x=x,
            alpha=state.ket(),
            num_modes=self.modes,
            cutoff=self.cutoff_dim
        )

        density = tf.cast(density, tf.double)

        n = tf.reshape(
            tf.stack([state.mean_photon(mode=i, cutoff=self.cutoff_dim)[0] for i in range(self.modes)]),
            shape=(self.modes,)
        )

        n = tf.cast(n, tf.double)

        m1 = self.atoms[0].m
        m2 = self.atoms[1].m
        q1 = self.atoms[0].q
        q2 = self.atoms[1].q
        omega1 = self.atoms[0].omega
        omega2 = self.atoms[1].omega

        if self.model == '11':

            potential = -2 * q1 * q2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '12':

            potential = q1 * q2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '13':
            pass

        elif self.model == '21':

            term1 = -2 * q1 * q2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            term2 = 3 * q1 * q2 * (tf.einsum('a,b,a->ab', x, x, x) - tf.einsum('a,b,b->ab', x, x, x)) / self.distance**4
            term3 =  -2 * q1 * q2 * (2 * tf.einsum('a,b,a,a->ab', x, x, x, x) - 3 * tf.einsum('a,b,a,b->ab', x, x, x, x) + 2 * tf.einsum('a,b,b,b->ab', x, x, x, x)) / self.distance**5
            potential = term1 + term2 + term3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '22':

            term1 = q1 * q2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            term2 =  -0.75 * q1 * q2 * (2 * tf.einsum('a,b,a,a->ab', x, x, x, x) - 3 * tf.einsum('a,b,a,b->ab', x, x, x, x) + 2 * tf.einsum('a,b,b,b->ab', x, x, x, x)) / self.distance**5
            potential = term1 + term2
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '23':
            pass

        elif self.model == '31':
            pass

        elif self.model == '32':
            pass

        elif self.model == '33':
            pass

        return cost

    def train(
        self,
        epochs: int
    ) -> None:
        """
        Trains the quantum neural network using the specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the network for.

        Returns:
            None
        """

        self.best_loss = float('inf')
        self.loss_history = []

        for i in range(epochs):

            if self.eng.run_progs:
                self.eng.reset()

            with tf.GradientTape() as tape:
                mapping = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1]))}
                state = self.eng.run(self.qnn, args=mapping).state
                loss = self.cost(state)


            #if loss < self.best_loss:
            #    self.best_loss = loss
            #    self.state = state

            gradients = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip([gradients], [self.weights]))
            self.loss_history.append(float(loss))

            if i%10==0:
                print("Epoch: {}/{} | Energy: {:.20f}".format(i+1, epochs, loss))

        self.best_loss = self.loss_history[-1]
        self.state = state

    def save_logs(self) -> None:

        loss_history = np.array(self.loss_history)
        state = self.state.ket()

        os.makedirs(os.path.join(self.save_dir, 'plots', 'loss_history'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'statevectors'), exist_ok=True)

        np.save(os.path.join(self.save_dir, 'plots', 'loss_history'), loss_history)
        np.save(os.path.join(self.save_dir, 'statevectors'), state)

        plot_loss_history(
            loss_history=self.loss_history,
            save_path=os.path.join(os.path.join(self.save_dir, 'plots', 'loss_history', 'loss'))
        )

