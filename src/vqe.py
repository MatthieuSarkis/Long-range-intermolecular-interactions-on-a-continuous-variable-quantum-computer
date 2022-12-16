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

from src.utils import plot_loss_history
from src.circuit import Circuit

class VQE():

    def __init__(
        self,
        modes: int,
        layers: int,
        distance: float,
        order: int,
        direction: str,
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6
    ) -> None:
        """
        Initializes a new instance of the QuantumNeuralNetwork class.

        Args:
            modes (int): The number of modes in the quantum neural network.
            layers (int): The number of layers in the quantum neural network.
            distance (float): Distance between the two QDOs.
            order (int): Order in the multipolar expansion.
            direction (str): Axis along which the electrons move: parallel or perpendicular.
            active_sd (float): The standard deviation of the active weights.
            passive_sd (float): The standard deviation of the passive weights.
            cutoff_dim (int): The cutoff dimension of the quantum engine.

        Returns:
            None
        """

        self.modes = modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance = distance
        self.order = order
        self.direction = direction

        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        self.qnn = sf.Program(self.modes)

        self.weights = self.init_weights(active_sd=active_sd, passive_sd=passive_sd) # our TensorFlow weights
        num_params = np.prod(self.weights.shape)   # total number of parameters in our model

        self.sf_params = np.arange(num_params).reshape(self.weights.shape).astype(np.str)
        self.sf_params = np.array([self.qnn.params(*i) for i in self.sf_params])

        self.circuit = Circuit(self.qnn, self.sf_params, self.layers)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_history = None
        self.state = None

    def init_weights(
        self,
        active_sd: float = 0.0001,
        passive_sd: float = 0.1
    ):
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
        Calculates the cost of a given Fock state using the quadratic Hamiltonian function.

        Args:
            state (FockStateTF): The Fock state for which to calculate the cost.

        Returns:
            tf.Tensor: The cost of the given Fock state.
        """

        x = tf.reshape(
            tf.stack([state.quad_expectation(mode=i, phi=0.0)[0] for i in range(self.modes)]),
            shape=(self.modes, 1)
        )

        p = tf.reshape(
            tf.stack([state.quad_expectation(mode=i, phi=0.5*np.pi)[0] for i in range(self.modes)]),
            shape=(self.modes, 1)
        )

        if self.order==2:

            g2 = -2 if self.direction=='parallel' else 1

            gamma = tf.Variable(
                np.full(shape=(self.modes, self.modes), fill_value=g2/self.distance**3) - np.eye(self.modes),
                dtype=tf.float32
            )

            H = 0.5 * tf.reduce_sum(x**2 + p**2) + 1.0 + 0.5 * tf.matmul(tf.transpose(x), tf.matmul(gamma, x))
            # the +1.0 in the above Hamiltonian corresponds to the vacuum energy
            # coming from symmetrizing the classical Hamiltonian and then using the
            # canonical commutation relation.

            return H[0][0]

        elif self.order==4:

            g2 = -2 if self.direction=='parallel' else 1
            g3 = 3 if self.direction=='parallel' else 0
            g4 = -2 if self.direction=='parallel' else -0.75

            H = 0.5 * tf.reduce_sum(x**2 + p**2) + 1.0 \
                + (g2 / self.distance**3) * x[0] * x[1] \
                + (g3 / self.distance**4) * x[0] * x[1] * (x[0] - x[1]) \
                + (g4 / self.distance**5) * x[0] * x[1] * (2 * x[0]**2 - 3 * x[0] * x[1] + 2 * x[1]**2)

            return H[0]

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

            if loss < self.best_loss:
                self.best_loss = loss
                self.state = state

            gradients = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip([gradients], [self.weights]))
            self.loss_history.append(float(loss))

            print("Epoch: {} | Energy: {:.20f}".format(i+1, loss))

    def save_logs(
        self,
        save_dir: str
    ) -> None:

        loss_history = np.array(self.loss_history)
        state = self.state.ket()

        np.save(os.path.join(save_dir, 'loss_history'), loss_history)
        np.save(os.path.join(save_dir, 'statevector'), state)

        plot_loss_history(
            loss_history=self.loss_history,
            save_path=os.path.join(save_dir, 'loss')
        )