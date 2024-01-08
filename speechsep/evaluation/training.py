import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from speechsep.plotting import format_plot, save_plot

METRIC_NAMES = {
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "val_stoi": "Validation STOI",
    "val_sisdr": "Validation SI-SDR",
}

VERSION_NAMES = {
    "31+32": "$B=64, p=0$",
    "33+34+35+36": "$B=8, p=0$",
    "37+38": "$B=64, p=0.2$",
}

PLOT_FOLDER = "data/evaluation/training/plots"


def plot_single_training(log: pd.DataFrame, metrics: list[str], params: dict):
    fig, (ax_loss, ax_metrics) = plt.subplots(2, 1, figsize=(14, 5), sharex="all")

    for metric in metrics:
        if metric not in METRIC_NAMES.keys():
            continue

        rows = log.dropna(subset=metric)[["epoch_frac", metric]]
        data = rows[metric]

        if "loss" in metric:
            ax = ax_loss
            data = data.clip(upper=0.025)
        else:
            ax = ax_metrics

        ax.plot(rows["epoch_frac"], data, label=METRIC_NAMES[metric])

    ax_metrics.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_metrics.set_ylabel("Metric")

    ax_loss.legend()
    ax_metrics.legend()

    format_plot()
    plt.show()
    save_plot(f"training_{params['versions']}", PLOT_FOLDER, fig)


def plot_multiple_trainings(logs: list[pd.DataFrame], paramss: list[dict]):
    fig, axs = plt.subplots(2, 1, figsize=(14, 5), sharex="all")

    for log, params in zip(logs, paramss):
        for loss, ax in zip(["train_loss", "val_loss"], axs):
            rows = log.dropna(subset=loss)[["epoch_frac", loss]]
            ax.plot(
                rows["epoch_frac"],
                _smoothen(rows[loss], loss).clip(upper=0.025),
                label=VERSION_NAMES[params["versions"]],
            )

    axs[0].legend()
    axs[0].set_ylabel("Training loss")
    axs[1].set_ylabel("Validation loss")

    axs[1].set_xlabel("Epoch")

    format_plot()
    plt.show()
    save_plot("training_losses", PLOT_FOLDER, fig)


def _smoothen(data: pd.Series, metric_name: str):
    with pd.option_context("mode.chained_assignment", None):
        # Smooth with rolling window proportional to logging frequency
        rolling_window = max(1, int(np.log1p(_get_metric_frequency(log, metric_name)) * 200))
        return data.rolling(rolling_window, center=True).mean()


def _get_metric_frequency(log: pd.DataFrame, metric: str) -> int:
    """Get the frequency at which the metric is logged."""
    steps_between_logs = log[log[metric].notna()]["step"].diff().mode().iloc[0]
    return 1 / steps_between_logs


def _steps_to_fractional_epoch(log: pd.DataFrame) -> list:
    """Convert steps to fractional epoch"""
    # Calculate steps in all epochs
    first_of_epoch = log.groupby("epoch").first()["step"]
    steps_per_epoch = first_of_epoch.diff().iloc[1:]
    steps_per_epoch.index -= 1

    # Manually add steps in last epoch
    last_epoch = log[log["epoch"] == log["epoch"].max()]
    steps_per_epoch.at[steps_per_epoch.index.max() + 1] = (
        last_epoch["step"].max() - last_epoch["step"].min()
    )

    # Map epoch length to each row
    steps_per_epoch_map = log["epoch"].apply(
        lambda epoch: steps_per_epoch.loc[epoch] if epoch in steps_per_epoch.index else pd.NA
    )
    first_of_epoch_map = log["epoch"].apply(
        lambda epoch: first_of_epoch.loc[epoch] if epoch in first_of_epoch.index else pd.NA
    )

    epoch_fractional = log["epoch"] + (log["step"] - first_of_epoch_map) / steps_per_epoch_map
    return epoch_fractional


def load_logs(base_folder: Path, versions: list[str]):
    log = []

    for version in versions:
        model_folder = base_folder / f"version_{version}"
        file = str(next(model_folder.glob("events.out.*")))
        print(f"Loading {file}")

        # Load events from log file
        event_acc = EventAccumulator(file).Reload()
        for metric in event_acc.Tags()["scalars"]:
            for event in event_acc.Scalars(metric):
                log.append(
                    {
                        "step": event.step,
                        "metric": metric,
                        "value": event.value,
                    }
                )

    log = pd.DataFrame.from_records(log)
    metrics = list(log["metric"].unique())
    metrics.remove("epoch")

    # Remove non-first rows with multiple values for same metric at a given step
    log = log.groupby(["metric", "step"]).first().reset_index()
    # Make each metric a column
    log = log.pivot(index="step", columns="metric", values="value").reset_index()
    # Drop rows with no values for epoch
    log = log.dropna(subset="epoch").reset_index(drop=True)
    log["epoch"] = log["epoch"].astype(int)

    # Calculate fractional epochs
    log["epoch_frac"] = _steps_to_fractional_epoch(log)
    log = log.dropna(subset="epoch_frac")

    return log, metrics


def print_summary(log: pd.DataFrame, metrics: list[str], params: dict):
    print(f"Versions: {params['versions']}")

    print("Average over last 10 epochs:")

    last_epoch = log["epoch"].max()
    log = log[log["epoch"].between(last_epoch - 9, last_epoch)]

    for metric in metrics:
        if metric not in METRIC_NAMES.keys():
            continue

        metric_mean = log[metric].mean()
        if "loss" in metric:
            metric_mean *= 1e3

        print(f" - {METRIC_NAMES[metric]}: {metric_mean:.3g}")


if __name__ == "__main__":
    model_folder_base = Path(sys.argv[1])
    sets = sys.argv[2:]

    logs = []
    paramss = []

    for versions in sets:
        params = {
            "versions": versions,
            "dataset": model_folder_base.parts[-1],
        }
        paramss.append(params)

        log, metrics = load_logs(model_folder_base, versions.split("+"))
        logs.append(log)

        print_summary(log, metrics, params)
        plot_single_training(log, metrics, params)

    plot_multiple_trainings(logs, paramss)
