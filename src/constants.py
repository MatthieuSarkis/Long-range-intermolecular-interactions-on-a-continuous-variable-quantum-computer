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

# The engine runs with atomic units, in which \hbar = 1.
HBAR = 1

# Obtained from "Optimized Quantum Drude Oscillators for Atomic and Molecular Response Properties"
# Those quantities are in atomic units (q_e = 1, m_e = 1, hbar = 1, Energy in Hartree)
ATOMIC_PARAMETERS = {
    'H':  {'omega': 0.4280, 'm': 0.8348, 'q': 0.8295},
    'Ne': {'omega': 1.1933, 'm': 0.3675, 'q': 1.1820},
    'Ar': {'omega': 0.6958, 'm': 0.3562, 'q': 1.3835},
    'Kr': {'omega': 0.6122, 'm': 0.3401, 'q': 1.4635},
    'Xe': {'omega': 0.5115, 'm': 0.3298, 'q': 1.5348},
}

ENERGY_UNIT_CONVERSION_FACTOR = {
    'hartree': 1.0,
    'eV': 27.2107,
    'cm-1': 219474.63,
    'kcalPerMol': 627.503,
    'kJPerMol': 2625.5,
    'kelvin': 315777,
    'J': 43.60e-19,
    'Hz': 6.57966e15,
}