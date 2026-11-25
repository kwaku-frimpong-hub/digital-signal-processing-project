"""Windowed-sinc, causal, linear-phase FIR low-pass filters."""

import numpy as np

from dsp_project.convolution import convolve_direct


def lowpass_fir(
    cutoff_hz: float,
    sample_rate: float,
    num_taps: int,
    window: str = "hamming",
) -> np.ndarray:
    """Compute coefficients once, then reuse them for filtering.

    For D = (num_taps - 1)/2, the ideal impulse response is
        h[n] = (2*cutoff_hz/sample_rate)
               * sinc((2*cutoff_hz/sample_rate) * (n-D)).
    Here sinc(u) = sin(pi*u)/(pi*u), including sinc(0) = 1.

    Windowing truncates the infinite ideal response. Division by the sum of
    coefficients gives unity DC gain. Odd tap counts make the causal delay D
    an integer. The cutoff is a transition midpoint, not a brick-wall boundary.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("cutoff_hz must lie strictly between zero and Nyquist")
    if (
        not isinstance(num_taps, int)
        or isinstance(num_taps, bool)
        or num_taps < 3
        or num_taps % 2 == 0
    ):
        raise ValueError("num_taps must be an odd integer of at least 3")
    if window not in ("rectangular", "hamming"):
        raise ValueError("window must be 'rectangular' or 'hamming'")

    n = np.arange(num_taps, dtype=float)
    delay = (num_taps - 1) // 2
    bandwidth = 2 * cutoff_hz / sample_rate
    coefficients = bandwidth * np.sinc(bandwidth * (n - delay))
    if window == "hamming":
        # The symmetric window is centered on the same delay as the sinc.
        coefficients *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_taps - 1))
    coefficients /= np.sum(coefficients)
    return coefficients


def apply_fir(samples: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Filter by full direct convolution; retain both delay and trailing tail."""
    return convolve_direct(samples, coefficients)
