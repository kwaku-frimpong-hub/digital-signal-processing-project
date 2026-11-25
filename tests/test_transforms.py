import numpy as np
import pytest

from dsp_project.transforms import FFTPlan, dft, fft, get_fft_plan, idft, ifft, next_power_of_two


@pytest.mark.parametrize("forward", [dft, fft])
def test_hand_calculated_sine_and_impulse(forward):
    np.testing.assert_allclose(forward([0, 1, 0, -1]), [0, -2j, 0, 2j], atol=1e-14)
    np.testing.assert_allclose(forward([1, 0, 0, 0]), [1, 1, 1, 1], atol=1e-14)
    np.testing.assert_allclose(forward([2, 2, 2, 2]), [8, 0, 0, 0], atol=1e-14)


@pytest.mark.parametrize("forward", [dft, fft])
def test_two_tone_spectrum(forward):
    n = np.arange(16)
    x = np.sin(2 * np.pi * 2 * n / 16) + 0.5 * np.cos(2 * np.pi * 5 * n / 16)
    expected = np.zeros(16, dtype=complex)
    expected[2], expected[14] = -8j, 8j
    expected[5], expected[11] = 4, 4
    np.testing.assert_allclose(forward(x), expected, atol=1e-13)


@pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 17, 32, 64])
def test_dft_and_idft_against_numpy(length):
    rng = np.random.default_rng(3000 + length)
    for _ in range(5):
        x = rng.normal(size=length) + 1j * rng.normal(size=length)
        np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=2e-12)
        np.testing.assert_allclose(idft(x), np.fft.ifft(x), atol=2e-12)
        np.testing.assert_allclose(idft(dft(x)), x, atol=2e-12)


@pytest.mark.parametrize("length", [1, 2, 4, 8, 16, 32, 64, 128, 1024])
@pytest.mark.parametrize("complex_input", [False, True])
def test_fft_many_random_signals(length, complex_input):
    rng = np.random.default_rng(length + 2026)
    plan = FFTPlan(length)
    for _ in range(10):
        x = rng.normal(size=length)
        if complex_input:
            x = x + 1j * rng.normal(size=length)
        original = x.copy()
        np.testing.assert_allclose(plan.fft(x), np.fft.fft(x), atol=2e-12)
        np.testing.assert_allclose(plan.ifft(x), np.fft.ifft(x), atol=2e-12)
        np.testing.assert_allclose(plan.ifft(plan.fft(x)), x, atol=2e-12)
        if length <= 64:
            np.testing.assert_allclose(plan.fft(x), dft(x), atol=2e-12)
            np.testing.assert_allclose(plan.ifft(x), idft(x), atol=2e-12)
        np.testing.assert_array_equal(x, original)


def test_prepared_fft_has_no_trigonometry_or_library_fft(monkeypatch):
    plan = FFTPlan(16)
    x = np.arange(16)
    expected = np.fft.fft(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("FFT execution evaluated trig or called library FFT")

    for name in ("sin", "cos", "exp"):
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(np.fft, "fft", forbidden)
    monkeypatch.setattr(np.fft, "ifft", forbidden)
    np.testing.assert_allclose(plan.fft(x), expected, atol=1e-12)
    np.testing.assert_allclose(plan.ifft(expected), x, atol=1e-12)


def test_plan_cache_and_convenience_wrappers():
    assert get_fft_plan(8) is get_fft_plan(8)
    x = np.arange(8)
    np.testing.assert_allclose(ifft(fft(x)), x, atol=1e-14)
    assert [next_power_of_two(n) for n in (1, 2, 3, 7, 8, 9)] == [1, 2, 4, 8, 8, 16]


@pytest.mark.parametrize("operation", [dft, idft, fft, ifft])
@pytest.mark.parametrize("bad", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_transform_inputs(operation, bad):
    with pytest.raises(ValueError):
        operation(bad)


@pytest.mark.parametrize("bad_length", [0, -1, 3, 6, 8.0, True])
def test_invalid_fft_length(bad_length):
    with pytest.raises(ValueError):
        FFTPlan(bad_length)


def test_fft_does_not_silently_pad():
    with pytest.raises(ValueError, match="power of two"):
        fft([1, 2, 3])
    with pytest.raises(ValueError, match="match"):
        FFTPlan(8).fft([1, 2, 3, 4])
