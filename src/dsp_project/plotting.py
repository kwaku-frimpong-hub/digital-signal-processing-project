"""Diagnostic plotting helpers.

NumPy's FFT is used ONLY to visualize spectra in this milestone. It is not a
substitute for the project's own DFT/FFT algorithms in Milestone 3.
"""

import numpy as np
from matplotlib.axes import Axes


def plot_time(
    ax: Axes,
    samples: np.ndarray,
    sample_rate: float,
    label: str,
    max_samples: int | None = None,
) -> None:
    """Plot samples on their original time axis, without hiding causal delay."""
    values = np.asarray(samples)
    if max_samples is not None:
        values = values[:max_samples]
    ax.plot(np.arange(len(values)) / sample_rate, values, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_spectrum(
    ax: Axes, samples: np.ndarray, sample_rate: float, label: str
) -> None:
    """Plot a one-sided amplitude spectrum of real samples.

    No taper is applied: finite-record leakage remains visible. DC and Nyquist
    bins are not doubled, unlike positive frequencies with negative partners.
    """
    values = np.asarray(samples, dtype=float)
    count = len(values)
    amplitude = np.abs(np.fft.rfft(values)) / count
    if count % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2
    frequencies = np.fft.rfftfreq(count, d=1 / sample_rate)
    ax.plot(frequencies, amplitude, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_filter_response(
    ax: Axes, coefficients: np.ndarray, sample_rate: float, label: str
) -> None:
    """Plot FIR gain in dB without rescaling the supplied coefficients."""
    fft_length = max(4096, 2 ** (len(coefficients) - 1).bit_length())
    response = np.abs(np.fft.rfft(coefficients, n=fft_length))
    frequencies = np.fft.rfftfreq(fft_length, d=1 / sample_rate)
    ax.plot(frequencies, 20 * np.log10(np.maximum(response, 1e-12)), label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_ylim(-100, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()
