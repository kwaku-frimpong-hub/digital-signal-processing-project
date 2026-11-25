"""Our direct and frequency-domain implementations of full linear convolution."""

import numpy as np

from dsp_project.transforms import dft, get_fft_plan, idft, next_power_of_two


def _inputs(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared validation and promotion for all convolution implementations."""
    x = np.asarray(x)
    h = np.asarray(h)
    for values in (x, h):
        if values.ndim != 1 or values.size == 0:
            raise ValueError("inputs must be nonempty one-dimensional sequences")
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must contain only finite samples")

    # Promote integers to floating point to avoid integer arithmetic/overflow.
    dtype = np.result_type(x.dtype, h.dtype, np.float64)
    return x.astype(dtype, copy=False), h.astype(dtype, copy=False)


def convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return y[n] = sum_k x[k] * h[n-k], treating outside samples as zero.

    Inputs must be nonempty, finite, one-dimensional sequences. The output
    contains len(x) + len(h) - 1 samples, including the filter's trailing tail.
    Real and complex inputs are supported. No cropping or delay compensation
    is performed. This intentionally simple algorithm costs O(N*M) operations.
    """
    x, h = _inputs(x, h)
    dtype = np.result_type(x.dtype, h.dtype)
    output = np.zeros(x.size + h.size - 1, dtype=dtype)

    # Input x[k] contributes a scaled copy of h beginning at output index k.
    # Writing the sum explicitly makes the indexing easy to inspect.
    for k in range(x.size):
        for j in range(h.size):
            output[k + j] += x[k] * h[j]
    return output


def _convolve_frequency(x: np.ndarray, h: np.ndarray, method: str) -> np.ndarray:
    x, h = _inputs(x, h)
    output_length = len(x) + len(h) - 1
    if method == "fft":
        length = next_power_of_two(output_length)
        plan = get_fft_plan(length)
        forward, inverse = plan.fft, plan.ifft
    else:
        # Exact linear-convolution length is enough for a general-length DFT.
        length = output_length
        forward, inverse = dft, idft
    padded_x = np.pad(x, (0, length - len(x)))
    padded_h = np.pad(h, (0, length - len(h)))
    output = inverse(forward(padded_x) * forward(padded_h))[:output_length]
    # Only discard roundoff imaginary parts when both original inputs are real.
    if not np.iscomplexobj(x) and not np.iscomplexobj(h):
        return output.real.copy()
    return output


def convolve_dft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full linear convolution via our DFT/IDFT; intended for small test cases."""
    return _convolve_frequency(x, h, "dft")


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full linear convolution via our FFT/IFFT, padded to prevent wraparound.

    A cached FFT plan is prepared on first use. Call get_fft_plan() beforehand
    if preparation must be excluded from processing timings.
    """
    return _convolve_frequency(x, h, "fft")
