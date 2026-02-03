"""
Implements the different uncertainty measures for a Gaussian ensemble.

Source:
https://github.com/cbuelt/kernel_entropy_uq/blob/main/uq/uq_measures.py
Some very slight semantic and stylistic adaptations made
"""

from typing import Literal


import numpy as np
import torch
from torch.utils import data
from model.datasets import get_dataset
from model.plotting import running_average, variable_value_lines
from model.training_loop import GLOBAL_DEVICE
from model.trunk_module import TrunkModule
from scipy.special import hyp1f1
from tqdm import tqdm
from model.score_losses import *
from model.datasets import get_dataloader


class GaussianUQMeasure:
    """Uncertainty measure for first-order Gaussian"""

    def __init__(
        self,
        prediction: torch.Tensor,
        variant: Literal["bma", "pairwise"] = "pairwise",
        second_order: Literal["ensemble", "mc_dropout"] = "ensemble",
        **kwargs,
    ):
        """
        Initialize class

        Args:
                prediction (torch.Tensor): Prediction tensor.
                variant (str, optional): Pairwise or BMA variant. Defaults to "pairwise".
                second_order (str, optional): _description_. Defaults to "ensemble".

        Raises:
                ValueError: second_order method must be "ensemble" or "mc_dropout"
                ValueError: Parediction must have shape [B, d1, ... dN, 2, M].
                NotImplementedError: BMA not implemented
        """
        self.variant = variant
        self.second_order = second_order
        self.gamma = kwargs.get("gamma", 0.1)

        if self.variant == "bma":
            raise NotImplementedError(
                "Bayesian model average not implemented, try 'pairwise' variant"
            )

        # Check sizes
        if self.second_order in ["ensemble", "mc_dropout"]:
            # Prediction should have shape [B, d1, ... dN, 2, M]
            if len(prediction.shape) < 3:
                prediction = prediction.unsqueeze(1)
            if prediction.shape[-2] != 2:
                raise ValueError(
                    f"Prediction must have shape [B, d1, ... dN, 2, M] for {self.second_order} second order method"
                )

            self.dimensions = prediction.shape[:-2]
            mu, sigma = torch.split(prediction.detach(), 1, -2)
            self.mu = mu.flatten(start_dim=1, end_dim=-2)
            self.sigma = sigma.flatten(start_dim=1, end_dim=-2)
            self.ens_size = prediction.shape[-1]
            self.corr = self.ens_size * (self.ens_size - 1)
        else:
            raise ValueError("Invalid second order method")

    def get_uncertainties(
        self, measure: Literal["crps", "kernel", "log", "var"] = "crps"
    ) -> tuple[float]:
        """
        Returns TU, AU, EU for a specific uncertainty measure. Supported measures are:
        - "crps": the Continuous Ranked Probability Score, corresponding to the energy
                score S_{ES} using \beta = 1
        - "kernel": the kernel score derived from a Gaussian kernel with bandwidth \gamma,
                where \gamma must be given as a constructor parameter
        - "log": default log-score corresponding to the Gaussian Negative Log-Likelihood
                which recovers the KL Divergence
        - "var": the variance-based measure resulting from (not strictly proper) score
                given by the squared error kernel

        Args:
                measure (str, optional): Defaults to "crps".

        Raises:
                ValueError: Measure must be one of ["crps", "kernel", "log", "var"]

        Returns:
                tuple: (AU, EU, TU)
        """
        if measure == "crps":
            au, eu = self._get_crps_uncertainty()
        elif measure == "kernel":
            au, eu = self._get_kernel_uncertainty()
        elif measure == "log":
            au, eu = self._get_log_uncertainty()
        elif measure == "var":
            au, eu = self._get_var_uncertainty()
        else:
            raise ValueError("Invalid measure")

        # Reshape to original dimensions
        au = au.reshape(*self.dimensions)
        eu = eu.reshape(*self.dimensions)

        tu = au + eu
        return au, eu, tu

    def _get_crps_uncertainty(self) -> tuple[float]:
        # Aleatoric uncertainty
        au = (self.sigma / np.sqrt(np.pi)).mean(dim=-1)

        # Epistemic uncertainty needs to be on CPU for hyp1f1 in scipy
        device = self.mu.device
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        mu_diff = mu_diff.cpu()
        sigma = sigma.cpu()
        tau = tau.cpu()
        f1 = hyp1f1(
            -0.5,
            0.5,
            -0.5 * torch.pow(mu_diff, 2) / (torch.pow(sigma, 2) + torch.pow(tau, 2)),
        )
        eu = torch.sqrt(torch.pow(sigma, 2) + torch.pow(tau, 2)) * np.sqrt(
            2 / np.pi
        ) * f1 - (sigma + tau) / np.sqrt(np.pi)
        diag_mask = 1 - torch.eye(self.ens_size, dtype=eu.dtype, device=eu.device)
        eu = eu * diag_mask
        eu = (eu.sum(dim=(-1, -2)) / self.corr).to(device)

        return au, eu

    def _get_kernel_uncertainty(self) -> tuple[float]:
        # Aleatoric uncertainty
        gamma = self.gamma
        au = 0.5 * (
            1 - gamma / (torch.sqrt(gamma**2 + 4 * torch.pow(self.sigma, 2)))
        ).mean(dim=-1)

        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        eu = (
            0.5 * gamma / torch.sqrt(gamma**2 + 4 * torch.pow(sigma, 2))
            + 0.5 * gamma / torch.sqrt(gamma**2 + 4 * torch.pow(tau, 2))
            - gamma
            / torch.sqrt(gamma**2 + 2 * (torch.pow(sigma, 2) + torch.pow(tau, 2)))
            * torch.exp(
                -torch.pow(mu_diff, 2)
                / (gamma**2 + 2 * (torch.pow(sigma, 2) + torch.pow(tau, 2)))
            )
        )
        eu = eu.sum(dim=(-1, -2)) / self.corr

        return au, eu

    def _get_log_uncertainty(self) -> tuple[float]:
        # Aleatoric uncertainty
        au = 0.5 * torch.log(2 * np.pi * np.e * torch.pow(self.sigma, 2)).mean(dim=-1)

        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        eu = (
            torch.log(sigma / tau)
            + (torch.pow(sigma, 2) + torch.pow(mu_diff, 2)) / (2 * torch.pow(tau, 2))
            - 0.5
        )
        eu = eu.sum(dim=(-1, -2)) / self.corr

        return au, eu

    def _get_var_uncertainty(self) -> tuple[float]:
        # Aleatoric uncertainty
        au = torch.pow(self.sigma, 2).mean(dim=-1)

        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        eu = torch.pow(mu_diff, 2).sum(dim=(-1, -2)) / self.corr

        return au, eu


## From here on it's our own code


def main():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    BATCH_SIZE = 128
    NUM_PREDICTORS = 50
    GAMMA = 2.0
    NAME_TO_SCORE = {
        "kernel": GaussianKernelScore,
        "var": SquaredError,  # TODO: check this
        "log": NLL,
        "crps": NormalCRPS,
    }

    pred_large = torch.rand(BATCH_SIZE, 5, 6, 2, NUM_PREDICTORS) * 100

    measures = ["crps", "kernel", "log", "var"]
    prediction_uqs = {k: [] for k in measures}

    dataset = get_dataset("casp")
    model = TrunkModule.load("kernel_casp_doga/dropout=0.3-loss_gk_gamma=2")
    model.eval_mcdropout()
    features = torch.from_numpy(dataset.test_features.copy().astype(np.float32))
    labels = torch.from_numpy(dataset.train_labels.copy().astype(np.float32))
    device = torch.device(GLOBAL_DEVICE)
    model = model.to(device)

    for batch_X, batch_y in tqdm(
        get_dataloader(features, labels, BATCH_SIZE),
        desc="UQ predictions",
    ):
        predictions = torch.stack(
            [model(batch_X.to(device)) for _ in range(NUM_PREDICTORS)], dim=-1
        )
        # pred2 = torch.rand(BATCH_SIZE, 2, NUM_PREDICTORS)  # Example mu2 tensor
        uq = GaussianUQMeasure(
            predictions,
            variant="pairwise",
            second_order="mc_dropout",
            gamma=GAMMA,
        )

        for measure in measures:
            # print(predictions.shape, batch_y.shape)
            au, eu, tu = uq.get_uncertainties(measure=measure)
            loss = NAME_TO_SCORE[measure](reduction=None, loss_gk_gamma=GAMMA)(
                predictions.mean(dim=-1), batch_y.to(device)
            )
            # print(loss.shape)
            for i in range(batch_X.shape[0]):
                prediction_uqs[measure].append(
                    (
                        loss[i].item(),
                        au[i].item(),
                        eu[i].item(),
                        tu[i].item(),
                    )
                )
    for k in prediction_uqs.keys():
        prediction_uqs[k].sort(reverse=True)

    for i, v in enumerate(["loss", "AU", "EU", "TU"]):
        variable_value_lines(
            {
                k: running_average([j[i] for j in v], 30)
                for k, v in prediction_uqs.items()
            },
            linewidth=3,
            title=v,
        )


if __name__ == "__main__":
    main()

