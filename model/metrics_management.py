import os
from datetime import datetime
import pickle
from copy import copy


def mean(iterator):
    return sum(iterator) / len(iterator)


def mean_and_dev(iterator):
    it_mean = mean(iterator)
    if len(iterator) <= 1:
        return it_mean, 0.0
    dev = sum((it_mean - i) ** 2 for i in iterator) / (len(iterator) - 1)
    return it_mean, dev


def stats(iterator):
    it_mean, dev = mean_and_dev(iterator)
    return it_mean, dev, min(iterator), max(iterator)


class Metrics:
    def __init__(self, per_epoch: bool = True) -> None:
        self.data = {}
        self.per_epoch = per_epoch

    def new_epoch(self):
        for k in self.data.keys():
            self.data[k].append([])

    def __call__(self, data_name: str, value: float):
        if data_name not in self.data:
            self.data[data_name] = [[]]
        epoch_list = self.data[data_name][-1]
        epoch_list.append(value)

    def get_epoch_level(self, data_name: str):
        if not self.per_epoch:
            print(self.per_epoch)
            return [mean(i) for i in self.data[data_name]]
        return [i for i in self.data[data_name]]

    def get_batch_level(self, data_name: str):
        if self.per_epoch:
            raise ValueError("Can't get per batch level for per epoch metrics manager")
        total = []
        for i in self.data[data_name]:
            total += copy(i)

    def get_all(self, epoch_level: bool = True):
        return {
            k: self.get_epoch_level(k) if epoch_level else self.get_batch_level(k)
            for k in self.data.keys()
        }

    def save(self, path: str, sub_name: str) -> None:
        print(f"Saving metrics {list(self.data.keys())} to:", path)
        # print("data:", self.data)
        compressed_data = {k: [stats(i) for i in v] for k, v in self.data.items()}
        pickle.dump(
            compressed_data,
            open(
                os.path.join(
                    path,
                    sub_name
                    + "_metrics_at_"
                    + datetime.now().strftime("%Y-%m-%d%H%M%S"),
                ),
                "wb",
            ),
        )

    def load(self, path: str) -> None:
        self.data = pickle.load(open(path, "rb"))
