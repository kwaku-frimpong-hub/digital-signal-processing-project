import numpy as np
import pytest

from dsp_project.filters import apply_fir, lowpass_fir
from dsp_project.signals import impulse, sine_wave


@pytest.mark.parametrize("window", ["hamming", "rectangular"])
def test_coefficients_symmetric_and_unity_dc(window):
    h = lowpass_fir(1000, 8000, 101, window)
    assert len(h) == 101
    np.testing.assert_allclose(h, h[::-1], atol=1e-15)
    assert np.sum(h) == pytest.approx(1.0)


def test_impulse_response_and_delay():
    h = lowpass_fir(1000, 8000, 101)
    response = apply_fir(impulse(1), h)
    np.testing.assert_array_equal(response, h)
    assert np.argmax(np.abs(response)) == 50
    # Full output preserves both the causal startup and trailing filter tail.
    assert len(apply_fir(np.ones(200), h)) == 300


def test_sinc_center_and_rectangular_coefficients():
    # Independent form sin(wc*m)/(pi*m), with wc = 2*pi*fc/fs.
    m = np.arange(-2, 3)
    expected = np.empty(5)
    expected[m == 0] = 0.25
    expected[m != 0] = np.sin(np.pi / 4 * m[m != 0]) / (np.pi * m[m != 0])
    expected /= expected.sum()
    np.testing.assert_allclose(lowpass_fir(1000, 8000, 5, "rectangular"), expected)


@pytest.mark.parametrize("frequency,expected_gain", [(200, 1.0), (2000, 0.0)])
def test_steady_state_pass_and_stop(frequency, expected_gain):
    x = sine_wave(frequency, 8000, 0.2)
    h = lowpass_fir(1000, 8000, 101)
    delay = 50
    y = apply_fir(x, h)[delay : delay + len(x)]
    # Remove edge regions requiring samples beyond the finite input.
    np.testing.assert_allclose(y[delay:-delay], expected_gain * x[delay:-delay], atol=0.003)


def test_two_tone_filtering():
    low = sine_wave(200, 8000, 0.2)
    high = sine_wave(2000, 8000, 0.2)
    h = lowpass_fir(1000, 8000, 101)
    aligned = apply_fir(low + high, h)[50 : 50 + len(low)]
    rmse = np.sqrt(np.mean((aligned[50:-50] - low[50:-50]) ** 2))
    assert rmse < 0.003


@pytest.mark.parametrize(
    "cutoff,fs,taps,window",
    [
        (0, 8000, 101, "hamming"),
        (4000, 8000, 101, "hamming"),
        (1000, 0, 101, "hamming"),
        (np.nan, 8000, 101, "hamming"),
        (1000, 8000, 10, "hamming"),
        (1000, 8000, 1, "hamming"),
        (1000, 8000, 11.0, "hamming"),
        (1000, 8000, 101, "unknown"),
    ],
)
def test_invalid_filter_parameters(cutoff, fs, taps, window):
    with pytest.raises(ValueError):
        lowpass_fir(cutoff, fs, taps, window)
