import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
from matplotlib.ticker import AutoMinorLocator
from torchmetrics.functional import scale_invariant_signal_noise_ratio

# Initialize seaborn formatting once module is loaded
sb.set(
    context="paper",
    style="ticks",
    font_scale=1.6,
    font="sans-serif",
    rc={
        "lines.linewidth": 1.4,
        "axes.titleweight": "bold",
    },
)


COLOR_DTU_NAVY = "#030F4F"
COLOR_DTU_RED = "#990000"
COLOR_DTU_ORANGE = "#FC7634"


def plot_waveform(waveform, sample_rate):
    # Assume mono signal
    waveform = waveform.numpy()[0]

    _, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.plot(time_axis, waveform)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate):
    waveform = waveform.numpy()[0]

    plt.specgram(waveform, Fs=sample_rate)
    plt.show(block=False)


def plot_separated_with_truth(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, ts: np.ndarray
):
    """
    Plot the separated signal on top of the ground truth.

    Args:
        x: mixed signal, shape (n_channels, n_samples)
        y: ground truth, shape (n_channels, n_samples)
        y_pred: separated signal, shape (n_channels, n_samples)
        ts: time steps
    """
    assert len(x.shape) == 2 and x.shape[0] == 1
    assert len(y.shape) == 2 and y.shape[0] == 2
    assert len(y_pred.shape) == 2 and y_pred.shape[0] == 2

    fig, axs = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True, sharey=True, tight_layout=True, dpi=300
    )

    ax = axs[0]
    ax.fill_between(ts, x[0], color="black")
    ax.set_title("Mixed")

    ax = axs[1]
    sisdr = scale_invariant_signal_noise_ratio(y_pred[0], y[0])
    ax.fill_between(ts, y[0], label="Ground truth", color="black")
    ax.fill_between(ts, y_pred[0], label="Prediction", alpha=0.6, color=COLOR_DTU_ORANGE)
    ax.set_title(f"Speaker 1 (SI-SDR: {sisdr:.2f} dB)")
    ax.legend()

    ax = axs[2]
    sisdr = scale_invariant_signal_noise_ratio(y_pred[1], y[1])
    ax.fill_between(ts, y[1], label="Ground truth", color="black")
    ax.fill_between(ts, y_pred[1], label="Prediction", alpha=0.6, color=COLOR_DTU_RED)
    ax.set_title(f"Speaker 2 (SI-SDR: {sisdr:.2f} dB)")
    ax.set_xlabel("Time [s]")
    ax.legend()

    for ax in axs:
        ax.set_ylabel("Amplitude")

    return fig


def format_plot():
    fig = plt.gcf()
    for ax in fig.axes:
        ax.get_xaxis().set_minor_locator(AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(AutoMinorLocator())
        ax.grid(b=True, which="major", linewidth=1.0)
        ax.grid(b=True, which="minor", linewidth=0.5, linestyle="-.")

    # fig.tight_layout(pad=0.1, h_pad=0.4, w_pad=0.4)]
    fig.tight_layout()


def save_plot(name: str, plots_folder="data/plots", fig=None, type="pdf", transparent=False):
    os.makedirs(
        plots_folder,
        exist_ok=True,
    )

    if fig is None:
        fig = plt.gcf()
    fig.savefig(
        os.path.join(plots_folder, f"{name}.{type}"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=transparent,
    )

    plt.close()
