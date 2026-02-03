from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import tensor, Tensor
import torch
import seaborn as sns
import os
from rich.console import Console
from rich.table import Table
from rich import box
import re

from model.metrics_management import Metrics

sns.set_theme()


def mean(iterable):
    return sum(iterable) / len(iterable)


def attempt_float_conversion(text):
    try:
        return float(text)
    except Exception as e:
        return text


def ordering_key(text: str):
    # https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    # https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
    return [
        attempt_float_conversion(c)
        for c in re.split(
            r"[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)", text
        )
    ]


def ordered_dict(dictionary, build_dict: bool = False):
    keys = sorted(list(dictionary.keys()), key=ordering_key)
    gen = ((k, dictionary[k]) for k in keys)
    if not build_dict:
        return gen
    return {k: v for k, v in gen}


# function gotten from bachelor thesis code
def running_average(
    sequence,
    size_average: int = 5,
    discards_beginning: bool = False,
) -> list[float]:
    if size_average <= 1:
        return sequence
    old_size_average = size_average
    size_average = min(size_average, len(sequence))
    # the running_sum and first output creation code below is not efficient
    running_sum = sum(sequence[:size_average])
    if not discards_beginning:
        output = [sum(sequence[: i + 1]) / (i + 1) for i in range(size_average)]
    else:
        output = []
    for i in range(size_average, len(sequence)):
        running_sum -= sequence[i - size_average]
        running_sum += sequence[i]
        output.append(running_sum / size_average)

    size_average = old_size_average
    return output


def test_performance_line(
    test_losses: list[float] | dict[str, list[float]],
    log: bool = True,
    path: str | None = None,
    show: bool = True,
    title: str = "",
) -> None:
    """
    test performance plot for debugging and quick evaluation
    """
    if not isinstance(test_losses, dict):
        test_losses = {"loss": test_losses}
    fig = plt.figure(figsize=(8, 4.5))
    for name, line in test_losses.items():
        plt.plot(range(1, len(line) + 1), line, label=name)

    plt.title(title)
    plt.legend()
    if log:
        plt.yscale("log")
    if path is not None:
        fig.savefig(f"data/logs/{path}")
    if show:
        plt.show()
    plt.close("all")


def variable_value_lines(
    data: list[float] | dict[str, list[float]],
    x: list[float | int] | None = None,
    log: bool = True,
    path: str | None = None,
    show: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    line_styles: list[str] | None = None,
    linewidth: int | float = 3,
    order: list[str] | None = None,
) -> None:
    """
    Create a prettier line plot for checking the changes in \
    values across different values of a variable. This allows \
    the use of colour gradients. 
    """
    if not isinstance(data, dict):
        data = {"loss": data}
    scaling = 0.6
    fig = plt.figure(figsize=(16 * scaling, 9 * scaling))
    colours = sns.color_palette("crest", n_colors=len(data.keys()))
    names = order if order else data.keys()
    line_styles = line_styles or ["-"] * len(names)
    for name, c, style in zip(names, colours, line_styles):
        line = data[name]
        # print("line", line, len(x if x else range(1, len(line) + 1)), c, style, name)
        plt.plot(
            x if x else range(1, len(line) + 1),
            line,
            label=name,
            color=c,
            linestyle=style,
            linewidth=linewidth,
        )

    plt.title(title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log:
        plt.yscale("log")
    if path is not None:
        print("saved")
        fig.savefig(f"data/images/{path}.png", dpi=300)
    if show:
        plt.show()
    plt.close("all")


def print_table(columns: list[str], data: list[list], title: str = ""):
    table = Table(title=title, box=box.ROUNDED)

    table.add_column(columns[0], justify="right", style="cyan", no_wrap=True)
    for column in columns[1:]:
        table.add_column(column, justify="right", style="green", no_wrap=True)

    for row in data:
        table.add_row(*row)

    console = Console()
    console.print(table, no_wrap=True)


def list_dir_visible(path: str):
    """list all public files in a directory"""
    for i in os.listdir(path):
        if not i.startswith("."):
            yield i


# def get_metric(experiment_name: str, metric_name: str) -> list:
#     """Get a metric from an experiment log"""
#     path = f"data/logs/{experiment_name}/metrics"
#     latest = max(list_dir_visible(path))
#     path = os.path.join(path, latest)
#     metrics = Metrics({})
#     metrics.load(path)
#     return metrics.archived_metrics[metric_name]


def extreme_mean(data, minimum: bool = True):
    data = [i[0] for i in data] if isinstance(data[0], tuple) else data
    value = min(data) if minimum else max(data)
    return f"{value:.5f}"


def get_metric_objs(experiment_name: str) -> tuple[Metrics, str]:
    """Get a metric from an experiment log"""
    path = f"data/logs/{experiment_name}"  # /metrics"

    latest = max([i for i in list_dir_visible(path) if "train_" in i])
    train_path = os.path.join(path, latest)

    latest = max([i for i in list_dir_visible(path) if "validation_" in i])
    val_path = os.path.join(path, latest)

    train_metrics = Metrics()
    train_metrics.load(train_path)
    val_metrics = Metrics()
    val_metrics.load(val_path)

    has_test_data = any("test_" in i for i in list_dir_visible(path))
    if has_test_data:
        latest = max([i for i in list_dir_visible(path) if "test_" in i])
        test_path = os.path.join(path, latest)
        test_metrics = Metrics()
        test_metrics.load(test_path)
        # print("loading test data")
        # print(
        #     len(train_metrics.get_epoch_level("gaussian_se")),
        #     len(val_metrics.get_epoch_level("gaussian_se")),
        #     len(test_metrics.get_epoch_level("gaussian_se")),
        # )

        return train_metrics, val_metrics, test_metrics
    return train_metrics, val_metrics, None


def get_metric(exp_name, metric_name, epoch_level: bool = True):
    train, val, test = get_metric_objs(exp_name)
    if epoch_level:
        return train.get_epoch_level(metric_name), val.get_epoch_level(metric_name)
    return train.get_batch_level(metric_name), val.get_batch_level(metric_name)


def get_multi_experiment_metric(experiment_base_name: str, metric_name: str) -> dict:
    """
    Get the metric data for a given metric across all experiments in a multi-config experiment.
    """
    path = f"data/logs/{experiment_base_name}"
    # print(path, list(list_dir_visible(path)))
    names = [(i, f"{experiment_base_name}/{i}") for i in list_dir_visible(path)]
    return {
        sub_name: get_metric(full_name, metric_name) for sub_name, full_name in names
    }


def get_cleaned_multi_exp_metric(
    base_name: str, metric: str, run_type: str = "validation"
) -> dict:
    index = ["train", "validation", "test"].index(run_type)
    data = get_multi_experiment_metric(base_name, metric)
    # print(list(data.keys()))
    # print(data[list(data.keys())[0]])
    data = {k: v[index] for k, v in data.items()}
    # print(data)
    return {k: [i[0] for i in v] for k, v in data.items()}


def get_best_performance(
    base_name: str, metrics: list[str], metrics_processor=extreme_mean
):
    columns = ["test"] + metrics
    path = f"data/logs/{base_name}"

    # train metrics table
    data = {i: [] for i in list_dir_visible(path)}

    for metric_name in metrics:
        for exp_name, metric_values in get_multi_experiment_metric(
            base_name, metric_name
        ).items():
            processed_metrics = metrics_processor(metric_values[0])
            data[exp_name].append(processed_metrics)

    print_table(
        columns,
        [[exp_name] + values for exp_name, values in ordered_dict(data)],
        title="Training results",
    )

    # validation metrics table
    data = {i: [] for i in list_dir_visible(path)}

    for metric_name in metrics:
        for exp_name, metric_values in get_multi_experiment_metric(
            base_name, metric_name
        ).items():
            processed_metrics = metrics_processor(metric_values[1])
            data[exp_name].append(processed_metrics)

    print_table(
        columns,
        [[exp_name] + values for exp_name, values in ordered_dict(data)],
        title="Validation results",
    )


def multi_experiment_plotting(base_name: str, metrics: list[str]):
    os.makedirs("data/images/" + base_name, exist_ok=True)

    for metric in metrics:
        base_name_path = f"{base_name}"
        data = ordered_dict(
            get_cleaned_multi_exp_metric(base_name, metric), build_dict=True
        )
        variable_value_lines(
            data,
            title=f"{metric} across tests",
            y_label=metric,
            x_label="epochs",
            path=os.path.join(base_name_path, f"{metric}_across_tests"),
            log=False,
        )


if "__main__" in __name__:
    # multi_experiment_plotting("first_test", False)
    dataset = "casp"
    test_type = "doga"
    metrics = ["gaussian_se", "gaussian_kernel", "gaussian_nll", "gaussian_crps"]
    for i in [i.split("_")[1] for i in metrics]:
        print("loss:", i)
        exp_base = f"{i}_{dataset}_{test_type}"
        get_best_performance(
            exp_base,
            metrics,
        )

        multi_experiment_plotting(exp_base, metrics)
