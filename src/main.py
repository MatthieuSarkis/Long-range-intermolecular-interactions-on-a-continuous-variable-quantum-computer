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

from argparse import ArgumentParser
from datetime import datetime
import json
import numpy as np
import os
import strawberryfields as sf
import tensorflow as tf

from src.energy_surface import EnergySurface
from src.utils import Atom
from src.constants import *

def main(args):

    # In StrawberryFields the default value of \hbar is 2. We change this
    # value to fit atomic units, in which \hbar=1, more natural in molecular physics.
    # Energies are then measured in Hartrees.
    # One can convert at will to other unit systems, cf. src.constants file.
    sf.hbar = HBAR

    # Initialize the random seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Define the log directory
    save_dir = os.path.join(
        args.save_dir,
        datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    )

    os.makedirs(save_dir, exist_ok=True)

    # Save the parameters of the run to the log directory
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Read out the atomic data and store them in the `Atom` structure
    atoms = []
    for atom in args.atom_list:
        atoms.append(Atom(
            m=ATOMIC_PARAMETERS[atom]['m'],
            omega=ATOMIC_PARAMETERS[atom]['omega'],
            q=ATOMIC_PARAMETERS[atom]['q']
        ))

    # Define the position quadrature grid.
    x_quadrature_grid = np.linspace(-6.0, 6.0, 500)

    # Instanciate an `EnergySurface` object
    energy_surface = EnergySurface(
        layers=args.layers,
        distance_list=distances,
        theta_list=thetas,
        x_quadrature_grid=x_quadrature_grid,
        atoms=atoms,
        active_sd=args.active_sd,
        passive_sd=args.passive_sd,
        cutoff_dim=args.cutoff_dim,
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        verbose=False
    )

    energy_surface.construct_energy_surface_parallelized(
        epsilon=args.epsilon,
        patience=args.patience
    )

if __name__ == '__main__':

    parser = ArgumentParser()

    distances = list(np.linspace(0.1, 3.5, 200))
    thetas = list(np.linspace(0.0, 0.5 * np.pi, 20))

    parser.add_argument("--layers",                   type=int,   default=8)
    parser.add_argument("--cutoff_dim",               type=int,   default=5)
    parser.add_argument("--distance_list", nargs='+', type=float, default=distances)
    parser.add_argument("--theta_list",    nargs='+', type=float, default=thetas)
    parser.add_argument('--atom_list',     nargs='+', type=str,   default=['Un', 'Un'], choices= ['Un', 'H', 'Ne', 'Ar', 'Kr', 'Xe'])
    parser.add_argument("--active_sd",                type=float, default=0.0001)
    parser.add_argument("--passive_sd",               type=float, default=0.1)
    parser.add_argument("--epochs",                   type=int,   default=100)
    parser.add_argument("--learning_rate",            type=float, default=0.01)
    parser.add_argument("--epsilon",                  type=float, default=1e-3)
    parser.add_argument("--patience",                 type=int,   default=10)
    parser.add_argument("--seed",                     type=int,   default=42)
    parser.add_argument("--save_dir",                 type=str,   default='./logs/')

    args = parser.parse_args()

    main(args=args)