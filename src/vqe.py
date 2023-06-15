# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
# The method `init_weights` was written by Xanadu and can be found at
# https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from math import sqrt, cos
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.backends.tfbackend.states import FockStateTF
from typing import List

from src.utils import Atom, quadratures_density
from src.circuit import Circuit

class VQE():

    def __init__(
        self,
        layers: int,
        distance: float,
        theta: float,
        x_quadrature_grid: np.ndarray,
        atoms: List[Atom],
        active_sd: float = 0.0001,
        passive_sd: float = 0.1,
        cutoff_dim: int = 6,
        learning_rate: float = 0.001,
        save_dir: str = 'logs/',
        verbose: bool = True
    ) -> None:
        """
        Initializes a new instance of the QuantumNeuralNetwork class.

        Args:
            layers (int): The number of layers in the quantum neural network.
            distance (float): Distance between the two QDOs.
            order (str): Order in the multipolar expansion: `quadratic`, `quartic` or `full`.
            theta (float): Angle with respect to the axis connecting the two nuclei. Defines the 1d model.
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
        self.modes = len(atoms) * self.dimension
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.distance = distance
        self.theta = theta
        self.atoms = atoms
        self.save_dir = save_dir
        self.verbose = verbose

        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        self.qnn = sf.Program(self.modes)

        self.weights = self.init_weights(active_sd=active_sd, passive_sd=passive_sd) # our TensorFlow weights
        num_params = np.prod(self.weights.shape)   # total number of parameters in our model

        self.sf_params = np.arange(num_params).reshape(self.weights.shape).astype(np.str)
        self.sf_params = np.array([self.qnn.params(*i) for i in self.sf_params])

        self.circuit = Circuit(self.qnn, self.sf_params, self.layers)

        self.best_loss = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_history = None
        self.state = None
        self.density = None
        self.marginals = None
        self.partial_entropy = None

        # Define the discretize position quadrature line/grid.
        self.x = tf.cast(tf.constant(x_quadrature_grid), tf.double)

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
            (tf.Variable[tf.float32]): A TensorFlow Variable of shape
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
        """Calculates the cost of a given Fock state using the Hamiltonian function.
        We treat either the full Coulomb potential Hamiltonian, or the Hamiltonian
        at some fixed order in the multipolar expansion.

        Args:
            state (FockStateTF): The Fock state for which to calculate the cost.

        Returns:
            (tf.Tensor): The cost of the given Fock state.
        """

        # Define the discretize position quadrature line/grid.
        x = self.x
        # Store the qudrature step playing the role of 'integration measure'.
        dx = (x[1] - x[0]).numpy()
        # Store the total number of values in the quadrature grid.
        L = x.shape[0]

        # Compute the joint probablity density of the quadratures
        # by calling `src.utils.quadrature_density`
        density = quadratures_density(
            x=x,
            alpha=state.ket(),
            num_modes=self.modes,
            cutoff=self.cutoff_dim
        )

        density = tf.cast(density, tf.double)

        # Compute the mean photon number for each photon channel
        # in the system and concatenate them.
        n = tf.reshape(
            tf.stack([state.mean_photon(mode=i, cutoff=self.cutoff_dim)[0] for i in range(self.modes)]),
            shape=(self.modes,)
        )

        n = tf.cast(n, tf.double)

        # Store the QDO parameters
        m1 = self.atoms[0].m
        m2 = self.atoms[1].m
        q1 = self.atoms[0].q
        q2 = self.atoms[1].q
        omega1 = self.atoms[0].omega
        omega2 = self.atoms[1].omega

        # Since the quadratures we are working with in Strawberry Fields are
        # dimensionless, the following dimensionful parameters have to appear explicitely
        # in the definition of the various potentials below.
        a1 = sqrt(sf.hbar / (m1 * omega1))
        a2 = sqrt(sf.hbar / (m2 * omega2))

        ct = cos(self.theta)

        potential = q1 * q2 * (
            1 / self.distance \
            - 1 / tf.sqrt(self.distance**2 + 2 * ct * a1 * self.distance * tf.repeat(x[:,tf.newaxis], L, 1) + a1**2 * tf.repeat((x**2)[:,tf.newaxis], L, 1)) \
            - 1 / tf.sqrt(self.distance**2 - 2 * ct * a2 * self.distance * tf.repeat(x[tf.newaxis,:], L, 0) + a2**2 * tf.repeat((x**2)[tf.newaxis,:], L, 0)) \
            + 1 / tf.sqrt(self.distance**2 - 2 * ct * self.distance * (a2 * tf.repeat(x[tf.newaxis,:], L, 0) - a1 * tf.repeat(x[:,tf.newaxis], L, 1)) + a1**2 * tf.repeat((x**2)[:,tf.newaxis], L, 1) + a2**2 * tf.repeat((x**2)[tf.newaxis,:], L, 0) - 2 * a1 * a2 * tf.einsum('a,b->ab', x, x))
        )
        potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
        cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        return cost

    def train(
        self,
        epsilon=1e-3,
        patience=20
    ) -> None:
        r"""
        Args:
            epsilon (float): Tolerance for the training loop stopping criterium.
            alpha (float): Rate for the moving average in the training loop.
            patience (int): Impose lack of improvement in loss for at least that number of epochs.

        Returns:
            None
        """

        prev_loss = float('inf')
        cpt = 0
        patience_cpt = 0

        self.loss_history = []
        self.loss_history_average = []

        while True:

            # Reset the engine
            if self.eng.run_progs:
                self.eng.reset()

            # Compute the loss
            with tf.GradientTape() as tape:
                mapping = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1]))}
                state = self.eng.run(self.qnn, args=mapping).state
                loss = self.cost(state)

            # Check if `epsilon`-improvement or not. If no improvement during
            # at least `patience` epochs, break the training loop.
            if np.abs(prev_loss - loss) < epsilon:
                if (cpt + 1) % 5 == 0:
                    patience_cpt += 1
            else:
                patience_cpt = 0
            if patience_cpt >= patience or cpt >= 500:
                break

            # Perform the classical optimization step
            gradients = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip([gradients], [self.weights]))
            self.loss_history.append(float(loss))

            cpt += 1

            if self.verbose:
                if (cpt + 1) % 10 == 0:
                    print("Epoch {:03d} | Loss {:2.6f} | Patience_cpt {} | diff {:2.6f} | Angle {} | Distance {}".format(cpt, loss, patience_cpt, np.abs(prev_loss - loss), self.theta, self.distance))

            prev_loss = loss
        # The value of the loss at the end of the training,
        # namely the ground state energy of the system.
        self.best_loss = self.loss_history[-1]

        # The ground state \alpha of the system, expressed as \sum_{mn}^cutoff \alpha_{mn} |m, n>
        # in the case of two modes, and obvious generalization for 6 modes.
        # Useful to keep it to maybe plot marginal Wigner functions.
        self.state = state