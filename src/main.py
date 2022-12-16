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
import numpy as np
import os
import tensorflow as tf

from src.vqe import VQE

def main(args):

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    vqe = VQE(
        modes=args.modes,
        layers=args.layers,
        distance=args.distance,
        order=args.order,
        direction=args.direction,
        active_sd=args.active_sd,
        passive_sd=args.passive_sd,
        cutoff_dim=args.cutoff_dim
    )

    vqe.train(epochs=args.epochs)
    vqe.save_logs(save_dir=args.save_dir)

    #for i in range(args.modes):
    #    plot_partial_wigner_function(state=vqe.state, mode=i)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--modes",      type=int,   default=2)
    parser.add_argument("--layers",     type=int,   default=8)
    parser.add_argument("--cutoff_dim", type=int,   default=6)
    parser.add_argument("--distance",   type=float, default=1.0)
    parser.add_argument("--order",      type=int,   default=2,          choices=[2, 4])
    parser.add_argument("--direction",  type=str,   default='parallel', choices=['parallel', 'perpendicular'])
    parser.add_argument("--active_sd",  type=float, default=0.0001)
    parser.add_argument("--passive_sd", type=float, default=0.1)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save_dir",   type=str,   default='./logs/')

    args = parser.parse_args()

    main(args=args)