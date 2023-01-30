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

from math import sqrt
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
        learning_rate: float = 0.001,
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
            learning_rate (float): The learning rate for the tensorflow optimizer.
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

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
        L = x.shape[0]

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

        a1 = sqrt(sf.hbar / (m1 * omega1))
        a2 = sqrt(sf.hbar / (m2 * omega2))

        if self.model == 'debug':

            print(n)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5)

        elif self.model == '11':

            potential = -2 * q1 * q2 * a1 * a2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '12':

            potential = q1 * q2 * a1 * a2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '13':

            term01 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term02 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term03 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)
            potential = q1 * q2 * a1 * a2 * (term01 + term02 - term03) / self.distance**3
            potential_expectation = tf.einsum('abcdef,abcdef->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) \
                 + sf.hbar * omega1 * (n[1] + 0.5) \
                 + sf.hbar * omega1 * (n[2] + 0.5) \
                 + sf.hbar * omega2 * (n[3] + 0.5) \
                 + sf.hbar * omega2 * (n[4] + 0.5) \
                 + sf.hbar * omega2 * (n[5] + 0.5) \
                 + potential_expectation

        elif self.model == '21':

            term1 = -2 * q1 * q2 * a1 * a2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            term2 = 3 * q1 * q2 * (
                a1**2 * a2 * tf.einsum('a,b,a->ab', x, x, x) \
                - a1 * a2**2 * tf.einsum('a,b,b->ab', x, x, x)
            ) / self.distance**4
            term3 =  -2 * q1 * q2 * (
                2 * a1**3 * a2 * tf.einsum('a,b,a,a->ab', x, x, x, x) \
                - 3 * a1**2 * a2**2 * tf.einsum('a,b,a,b->ab', x, x, x, x) \
                + 2 * a1 * a2**3 * tf.einsum('a,b,b,b->ab', x, x, x, x)
            ) / self.distance**5
            potential = term1 + term2 + term3
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '22':

            term1 = q1 * q2 * a1 * a2 * tf.einsum('a,b->ab', x, x) / self.distance**3
            term2 = -0.75 * q1 * q2 * (
                2 * a1**3 * a2 * tf.einsum('a,b,a,a->ab', x, x, x, x) \
                - 3 * a1**2 * a2**2 * tf.einsum('a,b,a,b->ab', x, x, x, x) \
                + 2 * a1 * a2**3 * tf.einsum('a,b,b,b->ab', x, x, x, x)
            ) / self.distance**5
            potential = term1 + term2
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '23':

            temp0 = tf.einsum('a,b->ab', x, x)

            term00 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp0[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term01 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp0[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term02 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp0[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)

            temp10 = tf.einsum('a,a,b->ab', x, x, x)
            temp11 = tf.einsum('a,b,c->abc', x, x, x)
            temp12 = tf.einsum('a,b,b->ab', x, x, x)

            term10 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp10[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 1), L, 2), L, 3), L, 4)
            term11 = tf.repeat(tf.repeat(tf.repeat(temp11[:,tf.newaxis,:,:,tf.newaxis,tf.newaxis], L, 1), L, 4), L, 5)
            term12 = tf.repeat(tf.repeat(tf.repeat(temp11[:,tf.newaxis,tf.newaxis,:,tf.newaxis,:], L, 1), L, 2), L, 4)
            term13 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp12[tf.newaxis,tf.newaxis,:,:,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 4), L, 5)
            term14 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp10[tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 2), L, 3), L, 4)
            term15 = tf.repeat(tf.repeat(tf.repeat(temp11[tf.newaxis,:,:,tf.newaxis,:,tf.newaxis], L, 0), L, 3), L, 5)
            term16 = tf.repeat(tf.repeat(tf.repeat(temp11[tf.newaxis,:,tf.newaxis,tf.newaxis,:,:], L, 0), L, 2), L, 3)
            term17 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp12[tf.newaxis,tf.newaxis,:,tf.newaxis,:,tf.newaxis], L, 0), L, 1), L, 3), L, 5)
            term18 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp10[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)
            term19 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp12[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)

            temp20 = tf.einsum('a,a,a,b->ab', x, x, x, x)
            temp21 = tf.einsum('a,a,b,b->ab', x, x, x, x)
            temp22 = tf.einsum('a,a,b,c->abc', x, x, x, x)
            temp23 = tf.einsum('a,b,b,b->ab', x, x, x, x)
            temp24 = tf.einsum('a,b,b,c->abc', x, x, x, x)
            temp25 = tf.einsum('a,b,c,d->abcd', x, x, x, x)
            temp26 = tf.einsum('a,b,c,c->abc', x, x, x, x)

            term200 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp20[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term201 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term202 = tf.repeat(tf.repeat(tf.repeat(temp22[:,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 2), L, 3), L, 5)
            term203 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[:,tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 1), L, 2), L, 3), L, 5)
            term204 = tf.repeat(tf.repeat(tf.repeat(temp22[:,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 1), L, 3), L, 4)
            term205 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 1), L, 2), L, 3), L, 4)
            term206 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp23[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term207 = tf.repeat(tf.repeat(tf.repeat(temp24[:,:,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 2), L, 4), L, 5)
            term208 = tf.repeat(tf.repeat(temp25[:,:,tf.newaxis,:,:,tf.newaxis], L, 2), L, 5)
            term209 = tf.repeat(tf.repeat(tf.repeat(temp24[:,tf.newaxis,tf.newaxis,:,:,tf.newaxis], L, 1), L, 2), L, 5)
            term210 = tf.repeat(tf.repeat(tf.repeat(temp24[:,tf.newaxis,:,:,tf.newaxis,tf.newaxis], L, 1), L, 4), L, 5)
            term211 = tf.repeat(tf.repeat(temp25[:,tf.newaxis,:,:,tf.newaxis,:], L, 1), L, 4)
            term212 = tf.repeat(tf.repeat(tf.repeat(temp26[:,tf.newaxis,tf.newaxis,:,tf.newaxis,:], L, 1), L, 2), L, 4)
            term213 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,:,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 0), L, 2), L, 4), L, 5)
            term214 = tf.repeat(tf.repeat(tf.repeat(temp24[tf.newaxis,:,tf.newaxis,:,:,tf.newaxis], L, 0), L, 2), L, 5)
            term215 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,tf.newaxis,:,:,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 4), L, 5)
            term216 = tf.repeat(tf.repeat(tf.repeat(temp24[tf.newaxis,tf.newaxis,:,:,tf.newaxis,:], L, 0), L, 1), L, 4)
            term217 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp20[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term218 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term219 = tf.repeat(tf.repeat(tf.repeat(temp24[tf.newaxis,tf.newaxis,:,tf.newaxis,:,:], L, 0), L, 1), L, 3)
            term220 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 2), L, 3), L, 4)
            term221 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp23[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term222 = tf.repeat(tf.repeat(tf.repeat(temp24[tf.newaxis,:,:,tf.newaxis,:,tf.newaxis], L, 0), L, 3), L, 5)
            term223 = tf.repeat(tf.repeat(temp25[tf.newaxis,:,:,tf.newaxis,:,:], L, 0), L, 3)
            term224 = tf.repeat(tf.repeat(tf.repeat(temp26[tf.newaxis,:,tf.newaxis,tf.newaxis,:,:], L, 0), L, 2), L, 3)
            term225 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,tf.newaxis,:,tf.newaxis,:,tf.newaxis], L, 0), L, 1), L, 3), L, 5)
            term226 = tf.repeat(tf.repeat(tf.repeat(temp24[tf.newaxis,tf.newaxis,:,tf.newaxis,:,:], L, 0), L, 1), L, 3)
            term227 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp20[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)
            term228 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp21[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)
            term229 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(temp23[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)

            potential0 = q1 * q2 * a1 * a2 * (
                term00 \
                + term01 \
                - term02
            ) / self.distance**3

            potential1 = q1 * q2 * 0.5 * (
                -3 * a1**2 * a2**1 * term10 \
                -6 * a1**2 * a2**1 * term11 \
                +6 * a1**1 * a2**2 * term12 \
                +3 * a1**1 * a2**2 * term13 \
                -3 * a1**2 * a2**1 * term14 \
                -6 * a1**2 * a2**1 * term15 \
                +6 * a1**1 * a2**2 * term16 \
                +3 * a1**1 * a2**2 * term17 \
                +6 * a1**2 * a2**1 * term18 \
                -6 * a1**1 * a2**2 * term19
            ) / self.distance**4

            potential2 = q1 * q2 * 0.5 * (
                -6  * a1**3 * a2**1 * term200 \
                +9  * a1**2 * a2**2 * term201 \
                -6  * a1**3 * a2**1 * term202 \
                +3  * a1**2 * a2**2 * term203 \
                +24 * a1**3 * a2**1 * term204 \
                -12 * a1**2 * a2**2 * term205 \
                -6  * a1**1 * a2**3 * term206 \
                -6  * a1**3 * a2**1 * term207 \
                +12 * a1**2 * a2**2 * term208 \
                -6  * a1**1 * a2**3 * term209 \
                +24 * a1**3 * a2**1 * term210 \
                -48 * a1**2 * a2**2 * term211 \
                +24 * a1**1 * a2**3 * term212 \
                +3  * a1**2 * a2**2 * term213 \
                -6  * a1**1 * a2**3 * term214 \
                -12 * a1**2 * a2**2 * term215 \
                +24 * a1**1 * a2**3 * term216 \
                -6  * a1**3 * a2**1 * term217 \
                +9  * a1**2 * a2**2 * term218 \
                +24 * a1**3 * a2**1 * term219 \
                -12 * a1**2 * a2**2 * term220 \
                -6  * a1**1 * a2**3 * term221 \
                +24 * a1**3 * a2**1 * term222 \
                -48 * a1**2 * a2**2 * term223 \
                +24 * a1**1 * a2**3 * term224 \
                -12 * a1**2 * a2**2 * term225 \
                +24 * a1**1 * a2**3 * term226 \
                -16 * a1**3 * a2**1 * term227 \
                +24 * a1**2 * a2**2 * term228 \
                -16 * a1**1 * a2**3 * term229
            ) / self.distance**5

            potential = potential0 + potential1 + potential2

            potential_expectation = tf.einsum('abcdef,abcdef->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) \
                 + sf.hbar * omega1 * (n[1] + 0.5) \
                 + sf.hbar * omega1 * (n[2] + 0.5) \
                 + sf.hbar * omega2 * (n[3] + 0.5) \
                 + sf.hbar * omega2 * (n[4] + 0.5) \
                 + sf.hbar * omega2 * (n[5] + 0.5) \
                 + potential_expectation

        elif self.model == '31':

            potential = q1 * q2 * (
                1 / self.distance \
                - 1 / tf.abs(self.distance + a1 * np.repeat(x[:,tf.newaxis], L, 1)) \
                - 1 / tf.abs(self.distance + a2 * np.repeat(x[tf.newaxis,:], L, 0)) \
                + 1 / tf.abs(self.distance + a1 * np.repeat(x[:,tf.newaxis], L, 1) - a2 * np.repeat(x[tf.newaxis,:], L, 0))
            )
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '32':

            potential = q1 * q2 * (
                1 / self.distance \
                - 1 / tf.sqrt(self.distance**2 + a1**2 * np.repeat((x**2)[:,tf.newaxis], L, 1)) \
                - 1 / tf.sqrt(self.distance**2 + a2**2 * np.repeat((x**2)[tf.newaxis,:], L, 0)) \
                + 1 / tf.sqrt(self.distance**2 + a1**2 * np.repeat((x**2)[:,tf.newaxis], L, 1) + a2**2 * np.repeat((x**2)[tf.newaxis,:], L, 0) - 2 * a1 * a2 * tf.einsum('a,b->ab', x, x))
            )
            potential_expectation = tf.einsum('ab,ab->', dx**self.modes * density, potential)
            cost = sf.hbar * omega1 * (n[0] + 0.5) + sf.hbar * omega2 * (n[1] + 0.5) + potential_expectation

        elif self.model == '33':

            potential0 = 1 / self.distance

            term11 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 3), L, 4), L, 5)
            term12 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 2), L, 3), L, 4), L, 5)
            term13 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 3), L, 4), L, 5)
            term14 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(x[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 3), L, 4), L, 5)

            potential1 = - 1 / tf.sqrt(self.distance**2 + a1**2 * (term11 + term12 + term13) + 2 * a1 * self.distance * term14)

            term21 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 2), L, 4), L, 5)
            term22 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 1), L, 2), L, 3), L, 5)
            term23 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 2), L, 3), L, 4)
            term24 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(x[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 2), L, 3), L, 4)

            potential2 = - 1 / tf.sqrt(self.distance**2 + a2**2 * (term21 + term22 + term23) - 2 * a2 * self.distance * term24)

            term301 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 2), L, 4), L, 5)
            term302 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 3), L, 4), L, 5)
            term303 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[:,tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis], L, 1), L, 2), L, 4), L, 5)
            term304 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 1), L, 2), L, 3), L, 5)
            term305 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 2), L, 3), L, 4), L, 5)
            term306 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[tf.newaxis,:,tf.newaxis,tf.newaxis,:,tf.newaxis], L, 0), L, 2), L, 3), L, 5)
            term307 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 2), L, 3), L, 4)
            term308 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,a->a', x, x)[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 3), L, 4), L, 5)
            term309 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.einsum('a,b->ab', x, x)[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 3), L, 4)
            term310 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(x[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:], L, 0), L, 1), L, 2), L, 3), L, 4)
            term311 = tf.repeat(tf.repeat(tf.repeat(tf.repeat(tf.repeat(x[tf.newaxis,tf.newaxis,:,tf.newaxis,tf.newaxis,tf.newaxis], L, 0), L, 1), L, 3), L, 4), L, 5)

            potential3 = 1 / tf.sqrt(
                self.distance**2 \
                + a2**2 * term301 \
                + a1**2 * term302 \
                -2 * a1 * a2 * term303 \
                + a2**2 * term304 \
                + a1**2 * term305 \
                -2 * a1 * a2 * term306 \
                + a2**2 * term307 \
                + a1**2 * term308 \
                -2 * a1 * a2 * term309 \
                -2 * a2 * self.distance * term310 \
                +2 * a1 * self.distance * term311
            )

            potential = potential0 + potential1 + potential2 + potential3

            potential_expectation = tf.einsum('abcdef,abcdef->', dx**self.modes * density, potential)

            cost = sf.hbar * omega1 * (n[0] + 0.5) \
                 + sf.hbar * omega1 * (n[1] + 0.5) \
                 + sf.hbar * omega1 * (n[2] + 0.5) \
                 + sf.hbar * omega2 * (n[3] + 0.5) \
                 + sf.hbar * omega2 * (n[4] + 0.5) \
                 + sf.hbar * omega2 * (n[5] + 0.5) \
                 + potential_expectation

        return cost

    def train(self, epsilon=1e-3, alpha=0.99):

        prev_loss = float('inf')
        avg_loss = 0
        cpt = 0

        self.loss_history = []
        self.loss_history_average = []

        while True:

            if self.eng.run_progs:
                self.eng.reset()

            with tf.GradientTape() as tape:
                mapping = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1]))}
                state = self.eng.run(self.qnn, args=mapping).state
                loss = self.cost(state)

            avg_loss = alpha * avg_loss + (1 - alpha) * loss

            if np.abs(prev_loss - avg_loss) < epsilon:
                break

            prev_loss = avg_loss

            gradients = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip([gradients], [self.weights]))
            self.loss_history.append(float(loss))
            self.loss_history_average.append(float(avg_loss))

            cpt += 1

            print("Epoch {} | Loss {:.10f} | Running average loss {:.10f}".format(cpt, loss, avg_loss))

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

