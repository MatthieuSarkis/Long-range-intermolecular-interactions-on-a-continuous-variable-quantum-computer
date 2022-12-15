from argparse import ArgumentParser
import numpy as np
import tensorflow as tf

from src.vqe import VQE
from src.utils import plot_partial_wigner_function

def main(args):

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    vqe = VQE(
        modes=args.modes,
        layers=args.layers,
        active_sd=args.active_sd,
        passive_sd=args.passive_sd,
        cutoff_dim=args.cutoff_dim
    )

    vqe.train(epochs=args.epochs)
    print(vqe.loss_history)

    for i in range(args.modes):
        plot_partial_wigner_function(state=vqe.state, mode=i)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--modes",      type=int,   default=2)
    parser.add_argument("--layers",     type=int,   default=8)
    parser.add_argument("--cutoff_dim", type=int,   default=6)
    parser.add_argument("--active_sd",  type=float, default=0.0001)
    parser.add_argument("--passive_sd", type=float, default=0.1)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)

    args = parser.parse_args()

    main(args=args)