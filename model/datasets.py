import os

import numpy as np
import pandas as pd

from collections import namedtuple
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


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
        return min(100000, self.data_points.shape[0])

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


def get_casp(
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    log_features=("F7", "F5"),  # the problematic ones
):
    """
    Loads dataset, applies preprocessing, and returns train/test splits

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # Load data
    df = pd.read_csv("data/CASP.csv")

    # Target anf features
    X = df.drop(columns="RMSD")
    y = df["RMSD"]

    # Log-transform selected features
    for feature in log_features:
        if feature in X.columns:
            X[feature] = np.log1p(X[feature])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # Standardize only on features
    # so no data leakage :)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, "scaler.pkl")

    # scale y separately
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    # scaler,
    # y_scaler,

    n_features = 9
    return DatasetBundle(
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        n_features,
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

    """
    # UniqueCarrier, Origin, Dest need encoding
    for var in ["UniqueCarrier", "Origin", "Dest"]:
        
        # X_train_dummies = pd.get_dummies(X_train[var])
        # X_train = pd.concat([X_train, X_train_dummies], axis=1).drop([var], axis=1)
        # X_train[var] = pd.Categorical(getattr(X_train, var))
        # X_train["categorical_" + var] = getattr(X_train, var).codes
        X_train["categorical_" + var] = pd.Categorical(getattr(X_train, var)).codes
        X_train.drop([var], axis="columns")

        # X_test_dummies = pd.get_dummies(X_test[var])
        # X_test = pd.concat([X_test, X_test_dummies], axis=1).drop([var], axis=1)
        # setattr(X_test, var, pd.Categorical(getattr(X_test, var)))
        # X_test["categorical_" + var] = getattr(X_test, var).codes
        X_test["categorical_" + var] = pd.Categorical(getattr(X_test, var)).codes
        X_test.drop([var], axis="columns")

        # X_val_dummies = pd.get_dummies(X_val[var])
        # X_val = pd.concat([X_val, X_val_dummies], axis=1).drop([var], axis=1)
        # setattr(X_val, var, pd.Categorical(getattr(X_val, var)))
        # X_val["categorical_" + var] = getattr(X_val, var).codes
        X_val["categorical_" + var] = pd.Categorical(getattr(X_val, var)).codes
        X_val.drop([var], axis="columns")
    """

    X_train = X_train.drop(
        ["DayofMonth"],  # , "UniqueCarrier", "Origin", "Dest"],
        axis="columns",
    ).values
    X_val = X_val.drop(
        ["DayofMonth"],  # , "UniqueCarrier", "Origin", "Dest"],
        axis="columns",
    ).values
    X_test = X_test.drop(
        ["DayofMonth"],  # , "UniqueCarrier", "Origin", "Dest"],
        axis="columns",
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

    n_features = 14
    return DatasetBundle(X_train, y_train, X_test, y_test, X_val, y_val, n_features)


def get_dataset(dataset_name: str):
    match dataset_name:
        case "airlines":
            return get_airlines()
        case "casp":
            return get_casp()
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
