# QDO

## Requirements

- python 3.8.10
- numpy
- tensorflow
- strawberryfields
- ipykernel

```shell
pip install -e .
```

or

```
conda env create -f tensorflow-apple-metal-conda.yml -n tensorflow
```

```
conda activate tensorflow
```

## Description of the project

We simulate the linear response theory of molecules on a continuous variable-based photonic quantum computer.

[tuto](https://strawberryfields.ai/photonics/demos/run_state_learner.html)

## Training

```shell
python src/main.py \
--modes 2 \
--layers 8 \
--cutoff_dim 6 \
--active_sd 0.0001 \
--passive_sd 0.1 \
--epochs 10 \
--seed 42
```

## License

[Apache License 2.0](https://github.com/MatthieuSarkis/qdo/blob/master/LICENSE)
