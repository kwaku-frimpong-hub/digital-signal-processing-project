import numpy as np
import pytest

from dsp_project.convolution import convolve_direct
from dsp_project.signals import impulse


def test_hand_calculated_convolution():
    np.testing.assert_array_equal(convolve_direct([1, 2, 3], [4, 5]), [4, 13, 22, 15])


def test_impulse_identity_and_shift():
    x = np.array([2, -1, 4])
    np.testing.assert_array_equal(convolve_direct(x, [1]), x)
    np.testing.assert_array_equal(convolve_direct(x, impulse(3, 2)), [0, 0, 2, -1, 4])


def test_commutativity_and_linearity():
    x = np.array([1.5, -2, 0.25])
    y = np.array([2, 0, 1])
    h = np.array([0.1, 0.8, 0.1, -0.2])
    np.testing.assert_allclose(convolve_direct(x, h), convolve_direct(h, x))
    np.testing.assert_allclose(
        convolve_direct(2 * x - 3 * y, h),
        2 * convolve_direct(x, h) - 3 * convolve_direct(y, h),
        atol=1e-14,
    )


@pytest.mark.parametrize("complex_input", [False, True])
def test_matches_numpy_reference(complex_input):
    rng = np.random.default_rng(2026)
    x, h = rng.normal(size=17), rng.normal(size=6)
    if complex_input:
        x = x + 1j * rng.normal(size=17)
        h = h + 1j * rng.normal(size=6)
    result = convolve_direct(x, h)
    assert len(result) == 22
    np.testing.assert_allclose(result, np.convolve(x, h), atol=1e-13)


def test_integer_inputs_do_not_overflow():
    x = np.array([100, 100], dtype=np.int8)
    np.testing.assert_array_equal(convolve_direct(x, x), [10000, 20000, 10000])


@pytest.mark.parametrize("bad", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        convolve_direct(bad, [1])
    with pytest.raises(ValueError):
        convolve_direct([1], bad)
