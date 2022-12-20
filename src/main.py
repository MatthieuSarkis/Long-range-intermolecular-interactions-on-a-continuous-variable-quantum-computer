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
import tensorflow as tf

from src.energy_surface import EnergySurface
from src.utils import Atom

def main(args):

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = os.path.join(
        args.save_dir,
        'potential={}'.format(args.order),
        'direction={}'.format(args.direction),
        datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    )

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    atoms = []
    for triplet in zip(args.mass_list, args.frequency_list, args.charge_list):
        atoms.append(Atom(m=triplet[0], omega=triplet[1], charge=triplet[2]))

    energy_surface = EnergySurface(
        modes=args.modes,
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

    parser.add_argument("--modes",                     type=int,   default=2)
    parser.add_argument("--layers",                    type=int,   default=8)
    parser.add_argument("--cutoff_dim",                type=int,   default=6)
    parser.add_argument("--distance_list",  nargs='+', type=float, default=1.0)
    parser.add_argument("--order",                     type=str,   default='quadratic', choices=['quadratic', 'quartic', 'full'])
    parser.add_argument("--direction",                 type=str,   default='parallel',  choices=['parallel', 'perpendicular'])
    parser.add_argument("--dimension",                 type=int,   default=1,           choices=[1, 3])
    parser.add_argument("--mass_list",      nargs='+', type=float, default=1.0)
    parser.add_argument("--frequency_list", nargs='+', type=float, default=1.0)
    parser.add_argument("--charge_list",    nargs='+', type=float, default=1.0)
    parser.add_argument("--order",                     type=str,   default='quadratic', choices=['quadratic', 'quartic', 'full'])
    parser.add_argument("--active_sd",                 type=float, default=0.0001)
    parser.add_argument("--passive_sd",                type=float, default=0.1)
    parser.add_argument("--epochs",                    type=int,   default=100)
    parser.add_argument("--seed",                      type=int,   default=42)
    parser.add_argument("--save_dir",                  type=str,   default='./logs/')

    args = parser.parse_args()

    main(args=args)