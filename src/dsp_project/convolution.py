"""Our direct implementation of full, linear discrete convolution."""

import numpy as np


def convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return y[n] = sum_k x[k] * h[n-k], treating outside samples as zero.

    Inputs must be nonempty, finite, one-dimensional sequences. The output
    contains len(x) + len(h) - 1 samples, including the filter's trailing tail.
    Real and complex inputs are supported. No cropping or delay compensation
    is performed. This intentionally simple algorithm costs O(N*M) operations.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    for values in (x, h):
        if values.ndim != 1 or values.size == 0:
            raise ValueError("inputs must be nonempty one-dimensional sequences")
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must contain only finite samples")

    # Promote integers to floating point to avoid integer arithmetic/overflow.
    dtype = np.result_type(x.dtype, h.dtype, np.float64)
    x = x.astype(dtype, copy=False)
    h = h.astype(dtype, copy=False)
    output = np.zeros(x.size + h.size - 1, dtype=dtype)

    # Input x[k] contributes a scaled copy of h beginning at output index k.
    # Writing the sum explicitly makes the indexing easy to inspect.
    for k in range(x.size):
        for j in range(h.size):
            output[k + j] += x[k] * h[j]
    return output
