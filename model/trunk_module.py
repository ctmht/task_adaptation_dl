import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrunkModule(nn.Module):
    _activation_class = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int] | None = None,
        activation: Literal["relu", "sigmoid", "tanh"] = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize a trunk module capable of performing Monte Carlo Dropout

        Args:
                n_features (int): number of input neurons, required
                hidden_dims (list[int]): hidden layer sizes. If None or empty,
                        this will correspond to a linear regression with activation
                activation (Literal["relu", "sigmoid", "tanh"]): the nonlinearity
                        which will be used on all layers in the network besides the
                        output heads
                dropout (float): probability of randomly dropping a neuron's
                        activations in the forward pass
        """
        super(TrunkModule, self).__init__()

        self._mcdropout: bool = False

        self.dropout = dropout

        self.activation_class = self._activation_class[activation]

        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims.insert(0, n_features)
        print(hidden_dims)

        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]),
                    nn.BatchNorm1d(hidden_dims[idx + 1]),
                    self.activation_class(),
                )
                for idx in range(len(hidden_dims) - 1)
            ]
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Identity(),  # TODO: change as necessary
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Softplus(),  # TODO: change as necessary
        )

        self.init_weights()

    def init_weights(self) -> None: ...

    def forward(self, data: torch.Tensor):
        """
        Forward pass of the model

        Args:
                data (torch.Tensor): the input data to the model, which is expected
                        to have tensor shape (... , n_features)
        """
        out = data

        for idx, layseq in enumerate(self.linears):
            out = layseq(out)

            out = F.dropout(
                out,
                p=self.dropout,
                training=self.training or self._mcdropout,
                inplace=False,
            )

        mus = self.mu_head(out)
        sigmas = self.sigma_head(out)

        out = torch.cat([mus, sigmas], dim=-1)

        return out

    def train(self, mode: bool = True) -> None:
        """
        Docstring for train

        :param mode: Description
        :type mode: bool
        """
        super(TrunkModule, self).train(mode=mode)
        self._mcdropout = False

    def eval_mcdropout(self) -> None:
        """
        Docstring for eval_mcdropout
        """
        super(TrunkModule, self).eval()
        self._mcdropout = True

    def save(self, sub_path: str):
        path = "data/models"
        path = os.path.join(path, sub_path)
        torch.save(self, path)

    @classmethod
    def load(cls, sub_path: str):
        try:
            path = os.path.join("data", "models", sub_path)
        except FileNotFoundError:
            path = path
        return torch.load(path, weights_only=False)


if __name__ == "__main__":
    # BATCH_SIZE = 1
    # NUM_PREDICTORS = 10
    # N_FEATURES = 9
    #
    # model = TrunkModule(
    #     n_features=N_FEATURES, hidden_dims=[32, 16], activation="tanh", dropout=0.3
    # )
    #
    # model.eval_mcdropout()
    #
    # # Check input-output shapes and that dropout masks actually apply uniquely
    # # to each predictor if we do this
    # dummy_in = torch.rand(BATCH_SIZE, N_FEATURES)
    # dummy_in = dummy_in.unsqueeze(1).repeat(1, NUM_PREDICTORS, 1)
    # dummy_out = model(dummy_in)
    #
    # print(dummy_in, dummy_out, dummy_in.shape, dummy_out.shape, sep="\n")
    model = TrunkModule.load("se_casp_lr/lr=0.001")
    print(model(torch.rand((1, 9))))
