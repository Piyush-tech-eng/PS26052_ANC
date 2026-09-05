"""Dependency-light PNG diagnostics for reproducible experiment artifacts.

The project deliberately avoids requiring a desktop plotting backend.  These
helpers use Pillow, which is part of the bundled workspace runtime, to create
clear line plots suitable for automated experiments and review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_COLOURS = (
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
    (23, 190, 207),
    (140, 86, 75),
    (227, 119, 194),
)


def save_line_plot(
    output_path: str | Path,
    series: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str = "Sample",
    ylabel: str = "Value",
    x_values: np.ndarray | None = None,
    logarithmic_y: bool = False,
    width: int = 1_400,
    height: int = 560,
) -> None:
    """Write an annotated multi-series line plot as a portable PNG."""

    if not series:
        raise ValueError("series must not be empty.")
    if width < 400 or height < 250:
        raise ValueError("Plot dimensions are too small.")

    prepared: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for label, values in series.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) == 0:
            raise ValueError("Each plot series must be a non-empty vector.")
        if not np.isfinite(array).all():
            raise ValueError("Plot series contains NaN or Inf.")
        if expected_length is None:
            expected_length = len(array)
        elif len(array) != expected_length:
            raise ValueError("All plot series must have the same length.")
        prepared[label] = np.log10(np.maximum(np.abs(array), 1e-18)) if logarithmic_y else array

    if x_values is None:
        x = np.arange(expected_length, dtype=np.float64)
    else:
        x = np.asarray(x_values, dtype=np.float64)
        if x.ndim != 1 or len(x) != expected_length or not np.isfinite(x).all():
            raise ValueError("x_values must be finite and match the series length.")

    y_all = np.concatenate(tuple(prepared.values()))
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    if np.isclose(y_min, y_max):
        margin = max(1.0, abs(y_min) * 0.1)
        y_min -= margin
        y_max += margin
    else:
        margin = (y_max - y_min) * 0.05
        y_min -= margin
        y_max += margin

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 100, 55, width - 35, height - 90
    plot_width = right - left
    plot_height = bottom - top

    for fraction in np.linspace(0.0, 1.0, 6):
        y_pixel = bottom - fraction * plot_height
        value = y_min + fraction * (y_max - y_min)
        draw.line((left, y_pixel, right, y_pixel), fill=(224, 224, 224), width=1)
        draw.text((8, y_pixel - 6), f"{value:.3g}", fill=(45, 45, 45), font=font)

    draw.line((left, top, left, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)
    draw.text((left, 14), title, fill=(20, 20, 20), font=font)
    draw.text((left, height - 35), xlabel, fill=(20, 20, 20), font=font)
    rendered_ylabel = f"log10(|{ylabel}|)" if logarithmic_y else ylabel
    draw.text((8, top - 18), rendered_ylabel, fill=(20, 20, 20), font=font)
    draw.text((left, bottom + 12), f"{x_min:.3g}", fill=(45, 45, 45), font=font)
    draw.text((right - 35, bottom + 12), f"{x_max:.3g}", fill=(45, 45, 45), font=font)

    max_points = max(2, plot_width)
    source_indices = np.linspace(0, expected_length - 1, min(expected_length, max_points)).astype(int)
    for series_index, (label, values) in enumerate(prepared.items()):
        colour = _COLOURS[series_index % len(_COLOURS)]
        points = []
        for index in source_indices:
            x_pixel = left + (x[index] - x_min) / (x_max - x_min) * plot_width
            y_pixel = bottom - (values[index] - y_min) / (y_max - y_min) * plot_height
            points.append((float(x_pixel), float(y_pixel)))
        draw.line(points, fill=colour, width=2)
        legend_x = right - 205
        legend_y = top + 5 + series_index * 16
        draw.line((legend_x, legend_y + 6, legend_x + 16, legend_y + 6), fill=colour, width=2)
        draw.text((legend_x + 21, legend_y), label, fill=(20, 20, 20), font=font)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
