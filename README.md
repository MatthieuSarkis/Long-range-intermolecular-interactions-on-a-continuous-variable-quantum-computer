# Modeling Non-Covalent Interatomic Interactions on a Photonic Quantum Computer

## Requirements

- python 3.8.10
- numpy
- tensorflow
- strawberryfields
- ipykernel

```shell
pip install -e .
```

## Description of the project

This repositories contains the code accompanying the paper "Modeling Non-Covalent Interatomic Interactions on a Photonic Quantum Computer".

## Training

The quantum neural network can be trained with the following command:
```shell
python src/main.py --atom_list Ar Ar --save_dir ./logs/
```

or to have a better control on the parameters of the simulation:
```shell
python src/main.py \
--layers 8 \
--cutoff_dim 5 \
--atom_list Ar Ar \
--active_sd 0.0001 \
--passive_sd 0.1 \
--learning_rate 0.005 \
--epsilon 1e-4 \
--patience 20 \
--seed 42 \
--save_dir ./logs/
```

## Citation

> Matthieu Sarkis, Alessio Fallani and Alexandre Tkatchenko.
> "Modeling Non-Covalent Interatomic Interactions on a Photonic Quantum Computer",
> [Journal](url) (2023).

## License

[Apache License 2.0](https://github.com/MatthieuSarkis/qdo/blob/master/LICENSE)
