import numpy as np
import pytest

from dsp_project.convolution import convolve_dft, convolve_direct, convolve_fft
from dsp_project.filters import apply_fir


@pytest.mark.parametrize("backend", [convolve_dft, convolve_fft])
@pytest.mark.parametrize("lengths", [(1, 1), (1, 9), (8, 1), (3, 4), (8, 9), (17, 6), (32, 31)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_full_convolution_matches_direct_and_numpy(backend, lengths, complex_input):
    rng = np.random.default_rng(sum(lengths))
    x, h = (rng.normal(size=n) for n in lengths)
    if complex_input:
        x = x + 1j * rng.normal(size=lengths[0])
        h = h + 1j * rng.normal(size=lengths[1])
    original_x, original_h = x.copy(), h.copy()
    result = backend(x, h)
    assert len(result) == sum(lengths) - 1
    assert np.iscomplexobj(result) == complex_input
    np.testing.assert_allclose(result, convolve_direct(x, h), atol=3e-12)
    np.testing.assert_allclose(result, np.convolve(x, h), atol=3e-12)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(h, original_h)


@pytest.mark.parametrize("backend", [convolve_dft, convolve_fft])
def test_tail_does_not_wrap_to_start(backend):
    np.testing.assert_allclose(
        backend([0, 0, 1], [0, 0, 1]), [0, 0, 0, 0, 1], atol=1e-14
    )
    np.testing.assert_allclose(backend([2, -1, 4], [0, 0, 1]), [0, 0, 2, -1, 4], atol=1e-14)
    np.testing.assert_allclose(backend([1, 2, 3], [4, 5]), [4, 13, 22, 15], atol=1e-13)


@pytest.mark.parametrize("backend", [convolve_dft, convolve_fft])
@pytest.mark.parametrize("bad", [[], [[1]], [np.nan], [np.inf]])
def test_invalid_inputs(backend, bad):
    with pytest.raises(ValueError):
        backend(bad, [1])
    with pytest.raises(ValueError):
        backend([1], bad)


@pytest.mark.parametrize("method", ["direct", "dft", "fft"])
def test_fir_backend_selection(method):
    np.testing.assert_allclose(apply_fir([1, 2, 3], [0.5, 0.5], method), [0.5, 1.5, 2.5, 1.5])


def test_bad_fir_backend():
    with pytest.raises(ValueError, match="method"):
        apply_fir([1], [1], method="unknown")
