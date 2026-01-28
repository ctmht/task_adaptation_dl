import json
from copy import deepcopy


DEFAULT_CONFIG = {
    "datatset": {
        "dataset_name": "CASP",
        "test_size": 0.2,
        "validation_size": 0.2,
    },
    "model": {
        "model_name": "Trunk",
        "hidden_dims": [64, 64, 64],
        "activation": "relu",
    },
    "loss_type": "gaussian_kernel",
    "loss_gk_gamma": 0.5,  # only does something for loss == 'gaussian_kernel'
    "lr": 1e-3,
    "batch_size": 64,
    "num_epochs": 5,
    "l2reg_strength": 0.0,
}


def load_config(path: str):
    return json.load(open(path, "r"))


def set_defaults(config: dict):
    default_config = deepcopy(config)
    default_config.update(config)
    return default_config
