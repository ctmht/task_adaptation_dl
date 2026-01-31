import json
from copy import deepcopy


DEFAULT_CONFIG = {
    "dataset_name": "CASP",
    "test_size": 0.2,
    "validation_size": 0.2,
    "model_name": "Trunk",  # will this matter?
    "hidden_dims": [64, 64, 64],
    "activation": "relu",
    "dropout": 0.3,
    "loss_type": "gaussian_kernel",
    "loss_gk_gamma": 2.0,  # only does something for loss == 'gaussian_kernel'
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
        if "base_name" not in i:
            raise ValueError("Each individual config must contain a 'base_name'")
        configs += vary_lists_configs(i, meta_config["do_not_vary"])

    print(configs)
    return configs


def vary_lists_configs(
    config, leave_out: list[str], variations: dict | None = None
) -> list[dict]:
    configs = []
    variations = variations or {}
    for k, v in config.items():
        if isinstance(v, list) and k not in leave_out:
            new_leave_out = deepcopy(leave_out)
            new_leave_out.append(k)
            for i in v:
                print(k, i)
                config_copy = deepcopy(config)
                config_copy[k] = i
                variations[str(k)] = str(i)
                configs += vary_lists_configs(config_copy, new_leave_out, variations)
            return configs

    if not variations:
        specific_name = config["base_name"]
    else:
        specific_name = "-".join(f"{k}={v}" for k, v in variations.items())
        specific_name = specific_name.replace(" ", "")  # just in case
    config["_specific_name"] = specific_name
    return [set_defaults(config)]


def set_defaults(config: dict):
    default_config = deepcopy(DEFAULT_CONFIG)
    default_config.update(config)
    return default_config
