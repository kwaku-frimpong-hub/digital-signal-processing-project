"""Generate finite, discrete-time test signals.

Frequencies are in Hz, durations in seconds, and sample rates in samples/second.
Sample n represents time n / sample_rate; the endpoint is not included.
"""

import numpy as np


def time_axis(sample_rate: float, duration: float) -> np.ndarray:
    """Return sample times for round(sample_rate * duration) samples."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    count = round(sample_rate * duration)
    if count < 1:
        raise ValueError("duration must produce at least one sample")
    return np.arange(count, dtype=float) / sample_rate


def sine_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Return amplitude * sin(2*pi*frequency*t + phase).

    Phase is in radians. Frequencies above Nyquist are deliberately allowed:
    this will be useful for the later aliasing experiments.
    """
    if not all(np.isfinite(value) for value in (frequency, amplitude, phase)):
        raise ValueError("frequency, amplitude, and phase must be finite")
    if frequency < 0:
        raise ValueError("frequency must be nonnegative")
    times = time_axis(sample_rate, duration)
    return amplitude * np.sin(2 * np.pi * frequency * times + phase)


def sine_mixture(
    frequencies: list[float],
    sample_rate: float,
    duration: float,
    amplitudes: list[float] | None = None,
) -> np.ndarray:
    """Sum sine waves with zero initial phase and optional amplitudes."""
    if len(frequencies) == 0:
        raise ValueError("provide at least one frequency")
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    if len(amplitudes) != len(frequencies):
        raise ValueError("one amplitude is required per frequency")
    result = np.zeros_like(time_axis(sample_rate, duration))
    for frequency, amplitude in zip(frequencies, amplitudes):
        result += sine_wave(frequency, sample_rate, duration, amplitude)
    return result


def impulse(length: int, position: int = 0) -> np.ndarray:
    """Return a unit impulse: one at position, zero everywhere else."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or not 0 <= position < length
    ):
        raise ValueError("position must be an integer within the signal")
    result = np.zeros(length, dtype=float)
    result[position] = 1.0
    return result
