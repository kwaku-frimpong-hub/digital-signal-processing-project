import numpy as np
import pytest
from dataclasses import replace

from dsp_project.pipeline import TransmissionConfig, reconstruction_metrics, transmit
from dsp_project.signals import sine_mixture, sine_wave


@pytest.fixture(scope="module")
def inputs():
    return (
        sine_mixture([200, 3500], 16000, 0.08, [0.7, 0.2]),
        sine_mixture([600, 3500], 16000, 0.08, [0.7, 0.2]),
    )


@pytest.fixture(scope="module")
def result(inputs):
    return transmit(*inputs)


def test_stage_lengths_delays_and_medium(result):
    n = len(result.inputs[0])
    assert result.receiver_delay == 50
    assert result.total_delay == 100
    for stage in (result.sender_filtered, result.modulated, result.demodulated):
        assert all(len(x) == n + 100 for x in stage)
    assert all(len(x) == n + 200 for x in result.recovered)
    assert all(len(x) == n + 100 for x in result.aligned_recovered)
    np.testing.assert_array_equal(result.transmitted, sum(result.modulated))


def test_stages_against_independent_numpy_reference(inputs, result):
    h = result.coefficients
    n = np.arange(len(result.transmitted))
    reference_medium = sum(
        np.convolve(x, h) * np.cos(2 * np.pi * f * n / 16000)
        for x, f in zip(inputs, (2000, 5000))
    )
    np.testing.assert_allclose(result.transmitted, reference_medium, atol=1e-13)
    for recovered, f in zip(result.recovered, (2000, 5000)):
        reference = np.convolve(reference_medium * 2 * np.cos(2 * np.pi * f * n / 16000), h)
        np.testing.assert_allclose(recovered, reference, atol=1e-13)


def test_sender_rejects_out_of_band_tone(result):
    # Compare sender output with the intended low tone after sender alignment.
    for branch, frequency in enumerate((200, 600)):
        n = len(result.inputs[branch])
        aligned = result.sender_filtered[branch][50:50 + n]
        target = sine_wave(frequency, 16000, n / 16000, amplitude=0.7)
        assert np.sqrt(np.mean((aligned[50:-50] - target[50:-50])**2)) < 0.003


def test_two_channel_reconstruction(result):
    for metric in reconstruction_metrics(result):
        assert metric["steady_rmse"] < 0.003
        assert metric["steady_relative_rmse"] < 0.006
        assert metric["steady_start"] == 150
        assert metric["steady_stop"] == len(result.inputs[0]) - 50


@pytest.mark.parametrize("active", [0, 1])
def test_single_active_channel_and_leakage(inputs, active):
    signals = [np.zeros_like(inputs[0]), np.zeros_like(inputs[1])]
    signals[active] = inputs[active]
    result = transmit(*signals)
    metrics = reconstruction_metrics(result)
    assert metrics[active]["steady_relative_rmse"] < 0.006
    inactive = 1 - active
    assert metrics[inactive]["steady_rmse"] < 0.001
    assert metrics[inactive]["steady_relative_rmse"] is None
    np.testing.assert_array_equal(result.sender_filtered[inactive], 0)


def test_zero_inputs_and_no_mutation(inputs):
    before = [x.copy() for x in inputs]
    transmit(*inputs)
    for original, copy in zip(inputs, before):
        np.testing.assert_array_equal(original, copy)
    zeros = np.zeros(220)
    result = transmit(zeros, zeros)
    np.testing.assert_array_equal(result.transmitted, 0)
    for recovered in result.recovered:
        np.testing.assert_array_equal(recovered, 0)


def test_unequal_inputs_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        transmit(np.ones(20), np.ones(21))


def test_short_signals_allowed_but_metrics_require_interior():
    result = transmit(np.ones(20), np.zeros(20))
    with pytest.raises(ValueError, match="too short"):
        reconstruction_metrics(result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"carrier_hz": (2000,)},
        {"carrier_hz": (-1, 5000)},
        {"carrier_hz": (np.nan, 5000)},
        {"carrier_hz": (2000, 3500)},  # overlap
        {"carrier_hz": (2000, 7500)},  # above Nyquist after modulation
        {"carrier_hz": (500, 5000)},   # sidebands cross zero
        {"carrier_hz": (1100, 5000)},  # self-image lacks guard
        {"guard_hz": -1},
        {"num_taps": 10},
        {"sample_rate": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TransmissionConfig(**kwargs)


def test_unsafe_override_warns():
    with pytest.warns(UserWarning, match="Unsafe nominal"):
        config = TransmissionConfig(carrier_hz=(2000, 3000), allow_unsafe_bands=True)
    assert config.allow_unsafe_bands


def test_override_does_not_allow_invalid_filter():
    with pytest.raises(ValueError):
        TransmissionConfig(cutoff_hz=9000, allow_unsafe_bands=True)


@pytest.mark.parametrize("active", [0, 1, "both"])
def test_fft_pipeline_matches_all_direct_stages(inputs, active):
    signals = [x.copy() for x in inputs]
    if active != "both":
        signals[1 - active] *= 0
    direct = transmit(*signals)
    fast = transmit(*signals, replace(direct.config, convolution_method="fft"))
    for name in ("sender_filtered", "modulated", "demodulated", "recovered", "aligned_recovered"):
        for expected, actual in zip(getattr(direct, name), getattr(fast, name)):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fast.transmitted, direct.transmitted, atol=1e-12)
    assert fast.total_delay == direct.total_delay
    for expected, actual in zip(reconstruction_metrics(direct), reconstruction_metrics(fast)):
        assert actual["steady_rmse"] == pytest.approx(expected["steady_rmse"], abs=1e-12)


def test_invalid_pipeline_backend():
    with pytest.raises(ValueError, match="convolution_method"):
        TransmissionConfig(convolution_method="unknown")


@pytest.mark.parametrize("length", [1, 900])
def test_fft_pipeline_padding_boundaries(length):
    # At N=900, sender convolution fits 1024 points, but receiver needs 2048.
    # N=1 also verifies preservation of tails for a very short input.
    rng = np.random.default_rng(length)
    first, second = rng.normal(size=length), rng.normal(size=length)
    direct = transmit(first, second)
    fast = transmit(first, second, TransmissionConfig(convolution_method="fft"))
    for actual, expected in zip(fast.recovered, direct.recovered):
        assert len(actual) == length + 200
        np.testing.assert_allclose(actual, expected, atol=1e-12)
