from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(
    signal: np.ndarray,
    output_path: str | Path,
    *,
    sampling_rate_hz: int,
    title: str,
    ylabel: str = "Amplitude",
) -> None:
    """Save a time-domain signal plot."""

    samples = np.asarray(signal, dtype=np.float64)

    time = (
        np.arange(len(samples))
        / sampling_rate_hz
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(time, samples)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_two_signals(
    first: np.ndarray,
    second: np.ndarray,
    output_path: str | Path,
    *,
    sampling_rate_hz: int,
    first_label: str,
    second_label: str,
    title: str,
) -> None:
    """Save a comparison plot of two time-domain signals."""

    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    if len(first) != len(second):
        raise ValueError(
            "Signals must have equal lengths."
        )

    time = (
        np.arange(len(first))
        / sampling_rate_hz
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(
        time,
        first,
        label=first_label,
    )
    plt.plot(
        time,
        second,
        label=second_label,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sequence(
    values: np.ndarray,
    output_path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Save a generic discrete sequence plot."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    x_axis = np.arange(len(values))

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(
        x_axis,
        values,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_two_sequences(
    first: np.ndarray,
    second: np.ndarray,
    output_path: str | Path,
    *,
    xlabel: str,
    first_label: str,
    second_label: str,
    title: str,
) -> None:
    """Save a comparison plot of two equal-length sequences."""

    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    if len(first) != len(second):
        raise ValueError(
            "Sequences must have equal lengths."
        )

    x_axis = np.arange(len(first))

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(
        x_axis,
        first,
        label=first_label,
    )
    plt.plot(
        x_axis,
        second,
        label=second_label,
    )
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()