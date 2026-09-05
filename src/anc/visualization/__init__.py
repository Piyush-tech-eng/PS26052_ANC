"""Visualization helpers with a dependency-light default plotting path."""

from anc.visualization.raster import save_line_plot

__all__ = ["plot_waveform", "plot_spectrum", "save_line_plot"]


def plot_waveform(*args, **kwargs):
    """Lazily import the legacy Matplotlib waveform helper when requested."""

    from anc.visualization.plots import plot_waveform as implementation

    return implementation(*args, **kwargs)


def plot_spectrum(*args, **kwargs):
    """Lazily import the legacy Matplotlib spectrum helper when requested."""

    from anc.visualization.plots import plot_spectrum as implementation

    return implementation(*args, **kwargs)
