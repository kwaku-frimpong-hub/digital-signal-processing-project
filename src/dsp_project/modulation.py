"""Coherent cosine modulation and demodulation on a shared sample clock."""

import numpy as np


def real_samples(samples: np.ndarray) -> np.ndarray:
    """Validate a finite, nonempty, real, one-dimensional signal."""
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0 or np.iscomplexobj(values):
        raise ValueError("samples must be a nonempty one-dimensional real signal")
    values = values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    return values


def cosine_carrier(
    length: int,
    carrier_hz: float,
    sample_rate: float,
    start_index: int = 0,
) -> np.ndarray:
    """Generate cos(2*pi*f*n/fs), starting at the absolute sample start_index.

    Frequency above Nyquist is allowed here for deliberate aliasing experiments.
    The pipeline checks whether a complete band configuration is safe.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not np.isfinite(carrier_hz) or carrier_hz < 0:
        raise ValueError("carrier_hz must be finite and nonnegative")
    n = np.arange(length, dtype=float) + start_index
    return np.cos(2 * np.pi * carrier_hz * n / sample_rate)


def modulate(
    samples: np.ndarray,
    carrier_hz: float,
    sample_rate: float,
    start_index: int = 0,
) -> np.ndarray:
    """Shift a real signal to both sidebands by multiplication with cosine."""
    values = real_samples(samples)
    return values * cosine_carrier(len(values), carrier_hz, sample_rate, start_index)


def demodulate(
    samples: np.ndarray,
    carrier_hz: float,
    sample_rate: float,
    start_index: int = 0,
) -> np.ndarray:
    """Mix down using 2*cosine; apply a low-pass separately to remove images.

    The factor 2 compensates for cos(a)*cos(a) = (1 + cos(2*a))/2.
    """
    return 2 * modulate(samples, carrier_hz, sample_rate, start_index)
