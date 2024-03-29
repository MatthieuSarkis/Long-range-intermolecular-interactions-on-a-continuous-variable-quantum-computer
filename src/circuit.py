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
from strawberryfields import ops
from strawberryfields.program import Program
from strawberryfields.program_utils import RegRef
from typing import List

class Circuit():

    def __init__(
        self,
        qnn: Program,
        sf_params: np.ndarray,
        layers: int
    ) -> None:
        """Constructor method of the Circuit class.

        This method initializes the object with the given quantum neural network `qnn`,
        the array of `sf_params` parameters, and the number of `layers` in the neural network.

        Args:
            qnn (Program): The quantum neural network.
            sf_params (np.ndarray): An array of parameters for the scaling and squeezing transformations.
            layers (int): The number of layers in the quantum neural network.

        Returns:
            None
        """

        self.qnn = qnn
        self.sf_params = sf_params

        with self.qnn.context as q:
            for k in range(layers):
                self.layer(sf_params[k], q)

    def interferometer(
        self,
        params: List[float],
        q: List[RegRef]
    ) -> None:
        """Parameterised interferometer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``max(1, N-1) + (N-1)*N`` parameters.

                * The first ``N(N-1)/2`` parameters correspond to the beamsplitter angles
                * The second ``N(N-1)/2`` parameters correspond to the beamsplitter phases
                * The final ``N-1`` parameters correspond to local rotation on the first N-1 modes

            q (list[RegRef]): list of Strawberry Fields quantum registers the interferometer
                is to be applied to
        """

        N = len(q)
        theta = params[:N*(N-1)//2]
        phi = params[N*(N-1)//2:N*(N-1)]
        rphi = params[-N+1:]

        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return

        n = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]

    def layer(
        self,
        params: List[float],
        q: List[RegRef]
    ) -> None:
        """CV quantum neural network layer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
                the number of parameters for the layer
            q (list[RegRef]): list of Strawberry Fields quantum registers the layer
                is to be applied to
        """
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        int1 = params[:M]
        s = params[M:M+N]
        int2 = params[M+N:2*M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]

        self.interferometer(int1, q)

        for i in range(N):
            ops.Sgate(s[i]) | q[i]

        self.interferometer(int2, q)

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]