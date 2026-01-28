import os

from torch import Value
import pd

from collections import namedtuple


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


def get_airlines():
    TRAIN_PATH = os.path.abspath("./airlines_train.h5")
    VAL_PATH = os.path.abspath("./airlines_val.h5")

    X_train = pd.read_hdf(TRAIN_PATH, mode="r", key="X")
    y_train = pd.read_hdf(TRAIN_PATH, mode="r", key="y")
    X_val = pd.read_hdf(VAL_PATH, mode="r", key="X")
    y_val = pd.read_hdf(VAL_PATH, mode="r", key="y")

    # # TODO: UniqueCarrier, Origin, Dest need one-hot, or maybe they just don't?
    X_train = X_train.drop(
        ["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns"
    ).values
    X_val = X_val.drop(
        ["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns"
    ).values

    y_train = y_train.values
    y_val = y_val.values

    print(X_val)
    print(X_train.shape, X_val.shape)
    print(y_val)
    print(y_train.shape, y_val.shape)

    return DatasetBundle(X_train, y_train, None, None, X_val, y_val, 11)


def get_dataset(dataset_name: str):
    match dataset_name:
        case "airlines":
            return get_airlines()

        case _:
            raise ValueError(f"No dataset named: {dataset_name}")
