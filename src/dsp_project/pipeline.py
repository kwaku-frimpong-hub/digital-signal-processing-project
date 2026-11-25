"""Two-channel transmission using the direct-convolution baseline.

All arrays keep their causal sample positions and full convolution tails.
No gain normalization or automatic delay removal is applied to the pipeline.
"""

from dataclasses import dataclass
import warnings

import numpy as np

from dsp_project.filters import apply_fir, lowpass_fir
from dsp_project.modulation import demodulate, modulate, real_samples


@dataclass(frozen=True)
class TransmissionConfig:
    """Shared settings for both branches; frequencies are in Hz.

    guard_hz is a nominal spacing allowance, not a guarantee about FIR stopband
    attenuation. The actual window/tap-count response still needs inspection.
    """

    sample_rate: float = 16000
    cutoff_hz: float = 1000
    carrier_hz: tuple[float, float] = (2000, 5000)
    num_taps: int = 101
    window: str = "hamming"
    guard_hz: float = 250
    allow_unsafe_bands: bool = False

    def __post_init__(self) -> None:
        # Reuse the FIR's parameter validation rather than inventing different
        # rules here. Actual filtering coefficients are prepared in transmit().
        lowpass_fir(self.cutoff_hz, self.sample_rate, self.num_taps, self.window)
        if len(self.carrier_hz) != 2:
            raise ValueError("exactly two carrier frequencies are required")
        if not all(np.isfinite(f) and f >= 0 for f in self.carrier_hz):
            raise ValueError("carriers must be finite and nonnegative")
        if not np.isfinite(self.guard_hz) or self.guard_hz < 0:
            raise ValueError("guard_hz must be finite and nonnegative")

        issues = []
        bandwidth = self.cutoff_hz
        nyquist = self.sample_rate / 2
        for frequency in self.carrier_hz:
            if frequency - bandwidth <= 0:
                issues.append("a transmitted band reaches/crosses zero frequency")
            if frequency + bandwidth >= nyquist:
                issues.append("a transmitted band reaches/crosses Nyquist")
        if abs(self.carrier_hz[1] - self.carrier_hz[0]) <= 2 * bandwidth + self.guard_hz:
            issues.append("carrier bands overlap or lack the requested guard")

        # Demodulation creates images at 2*f1, 2*f2, f1+f2 and |f1-f2|.
        # Fold centers modulo fs: images near multiples of fs alias to baseband.
        f1, f2 = self.carrier_hz
        for center in (2 * f1, 2 * f2, f1 + f2, abs(f1 - f2)):
            folded = abs((center + nyquist) % self.sample_rate - nyquist)
            if folded <= 2 * bandwidth + self.guard_hz:
                issues.append("a demodulation image reaches the baseband guard")
                break
        if issues:
            message = "Unsafe nominal frequency plan: " + "; ".join(issues)
            if not self.allow_unsafe_bands:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)


@dataclass
class TransmissionResult:
    """Intermediate signals for inspection, in branch order 1 then 2."""

    config: TransmissionConfig
    inputs: tuple[np.ndarray, np.ndarray]
    coefficients: np.ndarray
    sender_filtered: tuple[np.ndarray, np.ndarray]
    modulated: tuple[np.ndarray, np.ndarray]
    transmitted: np.ndarray
    demodulated: tuple[np.ndarray, np.ndarray]
    recovered: tuple[np.ndarray, np.ndarray]

    @property
    def receiver_delay(self) -> int:
        """Additional delay relative to the already sender-filtered reference."""
        return (self.config.num_taps - 1) // 2

    @property
    def total_delay(self) -> int:
        """Combined sender and receiver FIR delay relative to the raw input."""
        return 2 * self.receiver_delay

    @property
    def aligned_recovered(self) -> tuple[np.ndarray, np.ndarray]:
        """Comparison views aligned to sender_filtered, not to raw input."""
        delay = self.receiver_delay
        length = len(self.sender_filtered[0])
        return tuple(y[delay : delay + length] for y in self.recovered)


def transmit(
    signal1: np.ndarray,
    signal2: np.ndarray,
    config: TransmissionConfig | None = None,
) -> TransmissionResult:
    """Run sender filtering -> modulation -> sum -> demodulation -> filtering.

    Both real inputs must have equal lengths and a common sample rate. Unequal
    lengths are rejected rather than silently padded or trimmed in this milestone.
    Input index zero, sender output index zero, and receiver mixing index zero
    all refer to the same clock; filter delays do NOT restart carrier phases.
    """
    if config is None:
        config = TransmissionConfig()
    inputs = (real_samples(signal1).copy(), real_samples(signal2).copy())
    if len(inputs[0]) != len(inputs[1]):
        raise ValueError("both signals must have equal lengths")
    h = lowpass_fir(config.cutoff_hz, config.sample_rate, config.num_taps, config.window)
    sender = tuple(apply_fir(x, h) for x in inputs)
    modulated = tuple(
        modulate(x, carrier, config.sample_rate)
        for x, carrier in zip(sender, config.carrier_hz)
    )
    medium = modulated[0] + modulated[1]
    mixed = tuple(
        demodulate(medium, carrier, config.sample_rate) for carrier in config.carrier_hz
    )
    recovered = tuple(apply_fir(x, h) for x in mixed)
    return TransmissionResult(config, inputs, h, sender, modulated, medium, mixed, recovered)


def reconstruction_metrics(result: TransmissionResult) -> list[dict]:
    """Compare recovered channels to sender-filtered signals without rescaling.

    Include all aligned samples, plus a steady-state interval excluding both
    FIR boundary regions. For raw input length N and per-filter delay D, sender
    filtering is settled for indices [2D, N). A centered receiver comparison
    needs D more samples on either side, leaving [3D, N-D).

    Relative RMSE is undefined for a zero reference; return None, not a
    misleading ratio. Its absolute RMSE then measures leakage into that branch.
    """
    delay = result.receiver_delay
    start = 3 * delay
    stop = len(result.inputs[0]) - delay
    if start >= stop:
        raise ValueError("input is too short for steady-state metrics; need N > 4*delay")
    metrics = []
    for reference, recovered in zip(result.sender_filtered, result.aligned_recovered):
        error = recovered - reference
        steady_reference = reference[start:stop]
        steady_error = error[start:stop]
        rmse = float(np.sqrt(np.mean(steady_error**2)))
        reference_rms = float(np.sqrt(np.mean(steady_reference**2)))
        metrics.append({
            "aligned_rmse_all": float(np.sqrt(np.mean(error**2))),
            "steady_rmse": rmse,
            "steady_reference_rms": reference_rms,
            "steady_relative_rmse": rmse / reference_rms if reference_rms > 0 else None,
            "steady_start": start,
            "steady_stop": stop,
        })
    return metrics
