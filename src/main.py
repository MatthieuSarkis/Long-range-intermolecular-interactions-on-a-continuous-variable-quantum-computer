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
        'model={}'.format(args.model),
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
        model=args.model,
        atoms=atoms,
        active_sd=args.active_sd,
        passive_sd=args.passive_sd,
        cutoff_dim=args.cutoff_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=save_dir
    )

    energy_surface.construct_energy_surface()
    energy_surface.save_logs()

if __name__ == '__main__':

    parser = ArgumentParser()

    distances = list(np.linspace(1.0, 3.5, 30))
    #distances = [1.5]

    parser.add_argument("--layers",                   type=int,   default=8)
    parser.add_argument("--cutoff_dim",               type=int,   default=6)
    parser.add_argument("--distance_list", nargs='+', type=float, default=distances)
    parser.add_argument("--order",                    type=str,   default='full',       choices=['quadratic', 'quartic', 'full'])
    parser.add_argument("--model",                    type=str,   default='11',         choices=['debug', '11', '12', '13', '21', '22', '23', '31', '32', '33'])
    parser.add_argument('--atom_list',     nargs='+', type=str,   default=['Ar', 'Ar'], choices=['debug', 'H', 'Ne', 'Ar', 'Kr', 'Xe'])
    parser.add_argument("--active_sd",                type=float, default=0.0001)
    parser.add_argument("--passive_sd",               type=float, default=0.1)
    parser.add_argument("--epochs",                   type=int,   default=100)
    parser.add_argument("--learning_rate",            type=float, default=0.01)
    parser.add_argument("--seed",                     type=int,   default=42)
    parser.add_argument("--save_dir",                 type=str,   default='./logs/')

    args = parser.parse_args()

    main(args=args)