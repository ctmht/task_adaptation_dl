import json
from copy import deepcopy


DEFAULT_CONFIG = {
    "dataset_name": "CASP",
    "test_size": 0.2,
    "validation_size": 0.2,
    "model_name": "Trunk",
    "hidden_dims": [64, 64, 64],
    "activation": "relu",
    "loss_type": "gaussian_kernel",
    "loss_gk_gamma": 0.5,  # only does something for loss == 'gaussian_kernel'
    "lr": 1e-3,
    "batch_size": 64,
    "num_epochs": 5,
    "l2reg_strength": 0.0,
    "early_stopping": True,
}


def load_configs(path: str) -> list[dict]:
    meta_config = json.load(open(path, "r"))
    configs = []
    for i in meta_config["experiments"]:
        configs += vary_lists_configs(i, meta_config["do_not_vary"])

    return configs


def vary_lists_configs(config, leave_out: list[str]) -> list[dict]:
    configs = []
    for k, v in config.items():
        if isinstance(v, list) and k not in leave_out:
            for i in v:
                config_copy = deepcopy(config)
                config_copy[k] = i
                configs += vary_lists_configs(config_copy, leave_out)
            break

    return [config]


def set_defaults(config: dict):
    default_config = deepcopy(config)
    default_config.update(config)
    return default_config
