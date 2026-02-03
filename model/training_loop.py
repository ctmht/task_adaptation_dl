from copy import deepcopy
from typing import Any
import os

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm as tqdm_bar

import matplotlib.pyplot as plt

from model.config_management import load_configs
from model.datasets import get_dataloader, get_dataset
from model.metrics_management import Metrics, mean
from model.trunk_module import TrunkModule
from model.score_losses import *

# from rich.traceback import install

# install()


# torch.multiprocessing.set_start_method("fork", force=True)


NAME_TO_SCORE = {
    "gaussian_kernel": GaussianKernelScore,
    "gaussian_nll": NLL,
    "gaussian_se": SquaredError,
    "gaussian_crps": NormalCRPS,
}
RANDOM_SEED = 42
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.last_improvement = 0
        self.best = 1e100

    def __call__(self, value: float) -> bool:
        "returns True if the training should be stopped"
        print("\n\n", value, "\n\n")
        if value < self.best:
            self.best = value
            self.last_improvement = 0
        else:
            self.last_improvement += 1

        return self.last_improvement >= self.patience

    def improves(self, new_value: float) -> bool:
        return new_value < self.best


def ensure_environment(path, overwrite: bool = True):
    os.makedirs(path, exist_ok=overwrite)


def _forwardpass_over_data(
    model,
    input_data,
    output_data,
    training: bool = False,
    validation: bool = False,
    **hyperparameters: dict[str, Any],
):
    experiment_path = os.path.join(
        "data", "logs", hyperparameters["base_name"], hyperparameters["_specific_name"]
    )
    ensure_environment(experiment_path)

    # Generally used hyperparameters, and definition of the scores (here as metrics)
    batch_size = hyperparameters.get("batch_size", 64)
    num_epochs = hyperparameters.get("num_epochs", 10)

    l2reg_strength = hyperparameters.get("l2reg_strength", 0.1)  # TODO: use

    reduction = hyperparameters.get("reduction", "mean")
    ensemble = hyperparameters.get("ensemble", False)
    gaussker_gamma = hyperparameters.get("gaussker_gamma", 1.0)
    score_metric_objs = {
        name: val(reduction=reduction, ensemble=ensemble, loss_gk_gamma=gaussker_gamma)
        for name, val in NAME_TO_SCORE.items()
    }

    metrics = {name: [] for name in NAME_TO_SCORE.keys()}
    metrics_manager = Metrics()

    if not training:
        # Test hyperparameters
        use_mcdropout = hyperparameters.get("use_mcdropout", False)
        num_epochs = 1  # Force one epoch

        if use_mcdropout:
            # print("Not training this epoch. Using model.eval_mcdropout()")
            model.eval_mcdropout()
        else:
            # print("Not training this epoch. Using model.eval()")
            model.eval()

    else:
        # Training hyperparameters
        lr = hyperparameters.get("lr", 1e-4)

        # Score used as loss
        loss = hyperparameters["loss_type"]
        score_loss = NAME_TO_SCORE[loss](
            reduction=reduction, ensemble=ensemble, loss_gk_gamma=gaussker_gamma
        )

        early_stopping = hyperparameters.get("early_stopping", False)
        if early_stopping:
            val_input_data = hyperparameters["val_input_data"]
            val_output_data = hyperparameters["val_output_data"]
            early_stopping_tracker = EarlyStopping(patience=10)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        validation_metrics_manager = Metrics()

    # Setup model device
    device_name = GLOBAL_DEVICE
    device = torch.device(device_name)
    model = model.to(device)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(input_data.copy().astype(np.float32))
    y_tensor = torch.from_numpy(output_data.copy().astype(np.float32))

    num_samples = len(X_tensor)
    indices = np.arange(num_samples)

    # Iterate through epochs
    epoch_losses = []
    for epoch in range(num_epochs):
        metrics_manager.new_epoch()
        if training:
            np.random.shuffle(indices)
            print()

        scores = []
        metrics_perbatch = {name: [] for name in NAME_TO_SCORE.keys()}

        # Iterate through batches
        for batch_X, batch_y in tqdm_bar(
            get_dataloader(X_tensor, y_tensor, batch_size),
            desc=f"Epoch {epoch + 1}" if training else "Testing",
        ):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            if training:
                optimizer.zero_grad()

                # Forward pass
                pred_y = model(batch_X)

                # Backward pass
                score = score_loss(pred_y, batch_y)

                score.backward()

                # Clip the norm of all gradients acquired through backprop (rescale down to 3.0 if higher)
                max_norm = 3.0
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()

                scores.append(score.item())
            else:
                pred_y = model(batch_X)
                # print(batch_X.shape, batch_y.shape, pred_y.shape)

            # Metrics
            # TODO: also add UQ metrics
            for name, score_metric in score_metric_objs.items():
                metric = score_metric(pred_y, batch_y)
                metrics_manager(name, metric.item())
                metrics_perbatch[name].append(metric.item())

        # Average metrics over all batches in the epoch and save in metrics
        for name in metrics.keys():
            metrics[name].append(np.mean(metrics_perbatch[name]))

        if training:
            epoch_score = np.mean(scores)
            epoch_losses.append(epoch_score)
            print(f"\tLoss {loss}: {epoch_score:.6f}")

        case = "Training" if training else "Val/Test"
        print(
            f"\t{case} Metrics:",
            *[f"{key} {metrics[key][-1]:.6f}" for key in metrics],
            sep=", ",
        )

        if training:
            val_metrics = validate_model(
                model, val_input_data, val_output_data, **hyperparameters
            )
            validation_metrics_manager.append(val_metrics)
            print(hyperparameters["loss_type"])
            loss = mean(val_metrics.get_epoch_level(hyperparameters["loss_type"])[0])
            if early_stopping_tracker.improves(loss):
                os.makedirs(
                    "data/models/" + hyperparameters["base_name"], exist_ok=True
                )
                model.save(
                    hyperparameters["base_name"]
                    + "/"
                    + hyperparameters["_specific_name"]
                    + "__best"
                )
            if early_stopping and early_stopping_tracker(loss):
                break

    if not validation:
        metrics_manager.save(
            experiment_path,
            "train" if training else "test",
        )
    if training:
        os.makedirs("data/models/" + hyperparameters["base_name"], exist_ok=True)
        model.save(
            hyperparameters["base_name"] + "/" + hyperparameters["_specific_name"]
        )
        validation_metrics_manager.save(experiment_path, "validation")
    return metrics_manager


def train_model(model, input_data, output_data, **hyperparameters: dict[str, Any]):
    """ """
    return _forwardpass_over_data(
        model, input_data, output_data, training=True, **hyperparameters
    )


def validate_model(model, input_data, output_data, **hyperparameters: dict[str, Any]):
    """ """
    return _forwardpass_over_data(
        model,
        input_data,
        output_data,
        training=False,
        validation=True,
        **hyperparameters,
    )


def test_model(model, input_data, output_data, **hyperparameters: dict[str, Any]):
    """ """
    return _forwardpass_over_data(
        model, input_data, output_data, training=False, **hyperparameters
    )


def experiment_from_config(config: dict):
    print("Running experiment from config:", config)
    dataset = get_dataset(config["dataset_name"])

    model = TrunkModule(
        n_features=dataset.n_features,
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        dropout=config["dropout"],
    )
    model.train()

    print(
        "Model parameters count:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print(len(dataset.train_labels))

    hyperparameters = dict(
        # loss=config["loss_type"],
        # loss_gk_gamma=config["loss_gk_gamma"],
        # lr=config["lr"],
        # batch_size=config["batch_size"],
        # num_epochs=config["num_epochs"],
        # l2reg_strength=config["l2reg_strength"],  # not implemented,
        # early_stopping=config["early_stopping"],
        val_input_data=dataset.val_features,
        val_output_data=dataset.val_labels,
        **config,
    )

    train_model(
        model, dataset.train_features, dataset.train_labels, **deepcopy(hyperparameters)
    )

    model.eval_mcdropout()

    test_model(
        model, dataset.test_features, dataset.test_labels, **deepcopy(hyperparameters)
    )


def main():
    configs = load_configs(path="model/config2.json")
    for i in tqdm_bar(configs):
        experiment_from_config(i)


if __name__ == "__main__":
    main()
