import numpy as np
import pytest

from dsp_project.modulation import cosine_carrier, demodulate, modulate


def test_carrier_known_samples():
    np.testing.assert_allclose(cosine_carrier(4, 1, 4), [1, 0, -1, 0], atol=1e-15)


def test_modulation_and_demodulation_scaling():
    x = np.array([1, 2, 3, 4], dtype=float)
    carrier = cosine_carrier(4, 1, 4)
    np.testing.assert_allclose(modulate(x, 1, 4), x * carrier)
    np.testing.assert_allclose(demodulate(x, 1, 4), 2 * x * carrier)


def test_absolute_clock_continuity():
    whole = cosine_carrier(100, 1234, 16000)
    split = np.concatenate([
        cosine_carrier(37, 1234, 16000),
        cosine_carrier(63, 1234, 16000, start_index=37),
    ])
    np.testing.assert_array_equal(whole, split)


@pytest.mark.parametrize("samples", [[], [[1]], [1j], [np.nan], [np.inf]])
def test_invalid_samples(samples):
    for operation in (modulate, demodulate):
        with pytest.raises(ValueError):
            operation(samples, 2000, 16000)


@pytest.mark.parametrize(
    "length,frequency,fs,start", [(0, 1, 4, 0), (4, -1, 4, 0),
                                 (4, 1, 0, 0), (4, np.inf, 4, 0), (4, 1, 4, -1)]
)
def test_invalid_carrier(length, frequency, fs, start):
    with pytest.raises(ValueError):
        cosine_carrier(length, frequency, fs, start)
