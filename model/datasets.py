import os

import numpy as np
import pandas as pd

from collections import namedtuple
from torch.utils.data import DataLoader, Dataset


DatasetBundle = namedtuple(
    "DatasetBundle",
    [
        "train_features",
        "train_labels",
        "test_features",
        "test_labels",
        "val_features",
        "val_labels",
        "n_features",
    ],
)


class ArrayDataset(Dataset):
    def __init__(self, data_points, labels) -> None:
        self.data_points = data_points
        self.labels = labels

    def __len__(self) -> int:
        return self.data_points.shape[0]

    def __getitem__(self, index: int):
        return self.data_points[index], self.labels[index]


def get_random():
    return DatasetBundle(
        np.random.rand(10, 10),
        np.random.rand(10, 1),
        np.random.rand(10, 10),
        np.random.rand(10, 1),
        np.random.rand(10, 10),
        np.random.rand(10, 1),
        10,
    )


def get_airlines():
    TRAIN_PATH = os.path.abspath("./airlines_train.h5")
    VAL_PATH = os.path.abspath("./airlines_val.h5")
    TEST_PATH = os.path.abspath("./airlines_test.h5")

    X_train = pd.read_hdf(TRAIN_PATH, mode="r", key="X")
    y_train = pd.read_hdf(TRAIN_PATH, mode="r", key="y")
    X_val = pd.read_hdf(VAL_PATH, mode="r", key="X")
    y_val = pd.read_hdf(VAL_PATH, mode="r", key="y")
    X_test = pd.read_hdf(TEST_PATH, mode="r", key="X")
    y_test = pd.read_hdf(TEST_PATH, mode="r", key="y")

    # # TODO: UniqueCarrier, Origin, Dest need one-hot, or maybe they just don't?
    X_train = X_train.drop(
        ["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns"
    ).values
    X_val = X_val.drop(
        ["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns"
    ).values
    X_test = X_test.drop(
        ["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns"
    ).values

    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    print(X_val)
    print(X_train.shape, X_val.shape)
    print(y_val)
    print(y_train.shape, y_val.shape)
    # print(y_test)
    # print(y_train.shape, y_test.shape)

    return DatasetBundle(X_train, y_train, X_test, y_test, X_val, y_val, 11)


def get_dataset(dataset_name: str):
    match dataset_name:
        case "airlines":
            return get_airlines()
        case "random":
            return get_random()
        case _:
            raise ValueError(f"No dataset named: {dataset_name}")


def get_dataloader(
    data_points: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
):
    return DataLoader(
        ArrayDataset(data_points, labels), batch_size, shuffle
    )  # , num_workers=16)
