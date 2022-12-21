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
    # value to fit atomic units, in which \hbar=1. The reason is that this set of units is more
    # natural in molecular physics. Energies are then measured in Hartrees.
    # One can convert at will to other unit systems, cf. src.constants file.
    sf.hbar = HBAR

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = os.path.join(
        args.save_dir,
        'dimension={}'.format(args.dimension),
        'potential={}'.format(args.order),
        'direction={}'.format(args.direction),
        datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    ) if args.dimension==1 else os.path.join(
        args.save_dir,
        'dimension={}'.format(args.dimension),
        'potential={}'.format(args.order),
        datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    )

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    atoms = []
    for atom in args.atom_list:
        atoms.append(Atom(
            m=ATOMIC_PARAMETERS[atom]['m'],
            omega=ATOMIC_PARAMETERS[atom]['omega'],
            q=ATOMIC_PARAMETERS[atom]['q']
        ))

    energy_surface = EnergySurface(
        layers=args.layers,
        distance_list=args.distance_list,
        order=args.order,
        direction=args.direction,
        dimension=args.dimension,
        atoms=atoms,
        active_sd=args.active_sd,
        passive_sd=args.passive_sd,
        cutoff_dim=args.cutoff_dim,
        epochs=args.epochs,
        save_dir=save_dir
    )

    energy_surface.construct_energy_surface()
    energy_surface.save_logs()

if __name__ == '__main__':

    parser = ArgumentParser()

    distances = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
        3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
        5.1, 5.2, 5.3, 5.5, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0
    ]

    parser.add_argument("--layers",                   type=int,   default=8)
    parser.add_argument("--cutoff_dim",               type=int,   default=6)
    parser.add_argument("--distance_list", nargs='+', type=float, default=distances)
    parser.add_argument("--order",                    type=str,   default='full',       choices=['quadratic', 'quartic', 'full'])
    parser.add_argument("--direction",                type=str,   default='parallel',   choices=['parallel', 'perpendicular'])
    parser.add_argument("--dimension",                type=int,   default=3,            choices=[1, 3])
    parser.add_argument('--atom_list',     nargs='+', type=str,   default=['Ar', 'Ar'], choices=['H', 'Ne', 'Ar', 'Kr', 'Xe'])
    parser.add_argument("--active_sd",                type=float, default=0.0001)
    parser.add_argument("--passive_sd",               type=float, default=0.1)
    parser.add_argument("--epochs",                   type=int,   default=100)
    parser.add_argument("--seed",                     type=int,   default=42)
    parser.add_argument("--save_dir",                 type=str,   default='./logs/')

    args = parser.parse_args()

    main(args=args)