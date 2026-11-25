"""Our DFT/IDFT and iterative radix-2 FFT/IFFT.

Forward: X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).
Inverse: x[n] = sum_k X[k] * exp(+2*pi*i*k*n/N) / N.
No NumPy FFT routines are used here.
"""

from functools import lru_cache

import numpy as np


def _samples(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("transform input must be nonempty and one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("transform input must contain only finite samples")
    return values


def next_power_of_two(length: int) -> int:
    """Smallest radix-2 transform length at least as large as length."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    return 1 << (length - 1).bit_length()


def dft(samples: np.ndarray) -> np.ndarray:
    """Direct O(N²) transform, supporting arbitrary positive lengths.

    Compute one row at a time rather than allocating an N-by-N matrix.
    Unlike the FFT, this educational reference evaluates trigonometry per row.
    """
    values = _samples(samples)
    length = len(values)
    indices = np.arange(length)
    output = np.empty(length, dtype=complex)
    for k in range(length):
        angle = -2 * np.pi * k * indices / length
        factors = np.cos(angle) + 1j * np.sin(angle)
        output[k] = np.sum(values * factors)
    return output


def idft(spectrum: np.ndarray) -> np.ndarray:
    """Positive-exponent inverse with 1/N scaling, via conjugation."""
    values = _samples(spectrum)
    return np.conjugate(dft(np.conjugate(values))) / len(values)


class FFTPlan:
    """Precompute bit reversal and twiddles for a fixed power-of-two length.

    Construct a plan during setup, then reuse fft()/ifft() for signal processing.
    Execution performs only indexing, addition, multiplication, and scaling:
    no sine, cosine, or complex exponential evaluation.
    """

    def __init__(self, length: int) -> None:
        if next_power_of_two(length) != length:
            raise ValueError("FFT length must be a power of two")
        self.length = length

        # Reverse exactly log2(N) bits of each input index.
        indices = np.arange(length)
        reversed_indices = np.zeros(length, dtype=int)
        remaining = indices.copy()
        for _ in range(length.bit_length() - 1):
            reversed_indices = (reversed_indices << 1) | (remaining & 1)
            remaining >>= 1
        self._bit_reversal = reversed_indices

        # Every stage uses a stride through this single table of N-th roots.
        angles = -2 * np.pi * np.arange(length // 2) / length
        self._twiddles = np.cos(angles) + 1j * np.sin(angles)
        self._bit_reversal.setflags(write=False)
        self._twiddles.setflags(write=False)

    def fft(self, samples: np.ndarray) -> np.ndarray:
        """Iterative decimation-in-time FFT; input is never modified."""
        values = _samples(samples)
        if len(values) != self.length:
            raise ValueError("input length must match the FFT plan")
        output = values[self._bit_reversal].copy()
        width = 2
        while width <= self.length:
            half = width // 2
            twiddles = self._twiddles[:: self.length // width]
            # Each row is one butterfly group. NumPy handles independent groups
            # together; the radix-2 algorithm and twiddle indexing are ours.
            groups = output.reshape(-1, width)
            even = groups[:, :half].copy()
            odd = groups[:, half:] * twiddles
            groups[:, :half] = even + odd
            groups[:, half:] = even - odd
            width *= 2
        return output

    def ifft(self, spectrum: np.ndarray) -> np.ndarray:
        """Reuse the same forward twiddles with conjugation and 1/N scaling."""
        values = _samples(spectrum)
        return np.conjugate(self.fft(np.conjugate(values))) / self.length


@lru_cache(maxsize=16)
def get_fft_plan(length: int) -> FFTPlan:
    """Prepare/cache a plan. Call during setup to exclude setup from processing."""
    return FFTPlan(length)


def fft(samples: np.ndarray) -> np.ndarray:
    """Convenience FFT; prepares a cached plan on the first call per length.

    No implicit padding: callers must supply a power-of-two number of samples.
    Use an explicitly prepared FFTPlan when separating setup and execution.
    """
    values = _samples(samples)
    return get_fft_plan(len(values)).fft(values)


def ifft(spectrum: np.ndarray) -> np.ndarray:
    """Convenience inverse FFT, with the same length rules and plan cache."""
    values = _samples(spectrum)
    return get_fft_plan(len(values)).ifft(values)
