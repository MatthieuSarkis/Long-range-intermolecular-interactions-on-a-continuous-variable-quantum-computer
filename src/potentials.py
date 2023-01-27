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

import numpy as np
import tensorflow as tf

from src.utils import Atom

def V11(
    xvec: np.ndarray,
    x_density: tf.Tensor,
    atom1: Atom,
    atom2: Atom,
    distance: float
) -> tf.tensor:

    modes = x_density.shape[0]
    assert modes == 2

    dx = xvec[1] - xvec[0]

    m1 = atom1.m
    m2 = atom2.m
    q1 = atom1.q
    q2 = atom2.q
    omega1 = atom1.omega
    omega2 = atom2.omega

    potential = q1 * q2 * (-2 / distance**3) * x[0] * x[1]



