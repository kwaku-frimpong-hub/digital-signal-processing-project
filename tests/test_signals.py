import numpy as np
import pytest

from dsp_project.signals import impulse, sine_mixture, sine_wave, time_axis


def test_sample_times_exclude_endpoint():
    np.testing.assert_allclose(time_axis(8, 0.5), [0, 0.125, 0.25, 0.375])


def test_sine_known_samples_and_phase():
    np.testing.assert_allclose(sine_wave(1, 4, 1), [0, 1, 0, -1], atol=1e-15)
    np.testing.assert_allclose(
        sine_wave(1, 4, 1, amplitude=2, phase=np.pi / 2),
        [2, 0, -2, 0],
        atol=1e-15,
    )


def test_mixture_is_sum():
    expected = sine_wave(200, 8000, 0.1) + sine_wave(700, 8000, 0.1, 0.5)
    np.testing.assert_allclose(sine_mixture([200, 700], 8000, 0.1, [1, 0.5]), expected)


def test_impulse_position():
    np.testing.assert_array_equal(impulse(5, 2), [0, 0, 1, 0, 0])


def test_above_nyquist_is_allowed_for_later_experiments():
    np.testing.assert_allclose(
        sine_wave(6000, 8000, 0.01), -sine_wave(2000, 8000, 0.01), atol=1e-12
    )


@pytest.mark.parametrize("sample_rate,duration", [(0, 1), (8000, 0), (8, 0.01), (np.inf, 1)])
def test_invalid_time_axis(sample_rate, duration):
    with pytest.raises(ValueError):
        time_axis(sample_rate, duration)


def test_invalid_signal_parameters():
    with pytest.raises(ValueError):
        sine_wave(-1, 8000, 1)
    with pytest.raises(ValueError):
        sine_wave(200, 8000, 1, amplitude=np.nan)
    with pytest.raises(ValueError):
        sine_mixture([], 8000, 1)
    with pytest.raises(ValueError):
        sine_mixture([200, 400], 8000, 1, [1])
    with pytest.raises(ValueError):
        impulse(5, 5)
