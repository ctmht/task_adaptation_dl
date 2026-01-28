from copy import deepcopy
from typing import Literal, Any

import torch.nn.functional as F
import torch.nn as nn
import torch
from model.config_management import load_configs
from model.datasets import get_dataset
import pandas as pd
import numpy as np
import tqdm

import matplotlib.pyplot as plt

from model.trunk_module import TrunkModule
from model.score_losses import *


NAME_TO_SCORE = {
    "gaussian_kernel": GaussianKernelScore,
    "gaussian_nll": NLL,
    "gaussian_se": SquaredError,
    "gaussian_crps": NormalCRPS,
}
RANDOM_SEED = 42


def _forwardpass_over_data(
    model,
    input_data,
    output_data,
    training: bool = False,
    **hyperparameters: dict[str, Any],
):
    """ """
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

    if not training:
        # Test hyperparameters
        use_mcdropout = hyperparameters.get("use_mcdropout", False)
        num_epochs = 1  # Force one epoch

        if use_mcdropout:
            model.eval_mcdropout()
        else:
            model.eval()
    else:
        # Training hyperparameters
        lr = hyperparameters.get("lr", 1e-4)

        # Score used as loss
        loss = hyperparameters["loss"]
        score_loss = NAME_TO_SCORE[loss](
            reduction=reduction, ensemble=ensemble, loss_gk_gamma=gaussker_gamma
        )

        early_stopping = hyperparameters.get("early_stopping", False)
        if early_stopping:
            val_input_data = hyperparameters["val_input_data"]
            val_output_data = hyperparameters["val_output_data"]

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # Setup model device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
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
        if training:
            np.random.shuffle(indices)

        scores = []
        metrics_perbatch = {name: [] for name in NAME_TO_SCORE.keys()}

        if training:
            print()

        # Iterate through batches
        for batch_start in tqdm.tqdm(
            range(0, num_samples, batch_size),
            desc=f"Epoch {epoch + 1}" if training else "Testing",
        ):
            # Prepare batch
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            batch_X = X_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)

            if training:
                optimizer.zero_grad()

            # Forward pass
            pred_y = model(batch_X)

            if training:
                # Backward pass
                score = score_loss(pred_y, batch_y)

                score.backward()
                optimizer.step()

                scores.append(score.item())

            # Metrics
            # TODO: also add UQ metrics
            for name, score_metric in score_metric_objs.items():
                metric = score_metric(pred_y, batch_y)
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
            if early_stopping:
                test_model(model, val_input_data, val_output_data, **hyperparameters)


def train_model(model, input_data, output_data, **hyperparameters: dict[str, Any]):
    """ """
    return _forwardpass_over_data(
        model, input_data, output_data, training=True, **hyperparameters
    )


def test_model(model, input_data, output_data, **hyperparameters: dict[str, Any]):
    """ """
    return _forwardpass_over_data(
        model, input_data, output_data, training=False, **hyperparameters
    )


def experiment_from_config(config: dict):
    dataset = get_dataset(config["datasets"])

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

    hyperparameters = dict(
        loss=config["loss_type"],
        loss_gk_gamma=config["loss_gk_gamma"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        l2reg_strength=config["l2reg_strength"],  # not implemented,
        early_stopping=config["early_stopping"],
        val_input_data=dataset.val_features,
        val_output_data=dataset.test_features,
    )

    train_model(
        model, dataset.train_features, dataset.train_labels, **deepcopy(hyperparameters)
    )

    model.eval_mcdropout()

    test_model(
        model, dataset.test_features, dataset.test_labels, **deepcopy(hyperparameters)
    )


def main():
    configs = load_configs(path="src/config.json")
    for i in configs:
        experiment_from_config(i)


if __name__ == "__main__":
    main()
