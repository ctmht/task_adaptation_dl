import os
from datetime import datetime
import pickle
from copy import copy


def mean(iterator):
    return sum(iterator) / len(iterator)


class Metrics:
    def __init__(self) -> None:
        self.data = {}

    def new_epoch(self):
        for k in self.data.keys():
            self.data[k].append([])

    def __call__(self, data_name: str, value: float):
        if data_name not in self.data:
            self.data[data_name] = [[]]
        epoch_list = self.data[data_name][-1]
        epoch_list.append(value)

    def get_epoch_level(self, data_name: str):
        return [mean(i) for i in self.data[data_name]]

    def get_batch_level(self, data_name: str):
        total = []
        for i in self.data[data_name]:
            total += copy(i)

    def get_all(self, epoch_level: bool = True):
        return {
            k: self.get_epoch_level(k) if epoch_level else self.get_batch_level(k)
            for k in self.data.keys()
        }

    def save(self, path: str, sub_name: str) -> None:
        print("Saving metrics to:", path)
        print("data:", self.data)
        pickle.dump(
            self.data,
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
