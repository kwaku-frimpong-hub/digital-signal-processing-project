"""Reproduce transform, convolution, and complete pipeline equivalence checks.

This is a correctness demonstration, not the later timing benchmark suite.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp_project.convolution import convolve_dft, convolve_direct, convolve_fft
from dsp_project.pipeline import TransmissionConfig, reconstruction_metrics, transmit
from dsp_project.signals import sine_mixture
from dsp_project.transforms import dft, fft, idft, ifft


def max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/milestone3"))
    args = parser.parse_args()
    seed = 2026
    rng = np.random.default_rng(seed)

    n = np.arange(32)
    signal = np.sin(2 * np.pi * 3 * n / 32) + 0.5 * np.cos(2 * np.pi * 7 * n / 32)
    slow_spectrum, fast_spectrum = dft(signal), fft(signal)
    transform_errors = {
        "fft_vs_dft": max_error(fast_spectrum, slow_spectrum),
        "idft_round_trip": max_error(idft(slow_spectrum), signal),
        "ifft_round_trip": max_error(ifft(fast_spectrum), signal),
    }

    convolution_checks = []
    for x_length, h_length in ((3, 4), (17, 6), (32, 15)):
        x, h = rng.normal(size=x_length), rng.normal(size=h_length)
        reference = convolve_direct(x, h)
        convolution_checks.append({
            "input_length": x_length,
            "filter_length": h_length,
            "output_length": len(reference),
            "dft_max_abs_error": max_error(convolve_dft(x, h), reference),
            "fft_max_abs_error": max_error(convolve_fft(x, h), reference),
        })

    config = TransmissionConfig()
    signals = (
        sine_mixture([200, 3500], config.sample_rate, 0.25, [0.7, 0.2]),
        sine_mixture([600, 3500], config.sample_rate, 0.25, [0.7, 0.2]),
    )
    direct = transmit(*signals, config)
    fast = transmit(*signals, replace(config, convolution_method="fft"))
    pipeline_errors = {}
    for name in ("sender_filtered", "modulated", "demodulated", "recovered"):
        pipeline_errors[name] = [
            max_error(actual, expected)
            for actual, expected in zip(getattr(fast, name), getattr(direct, name))
        ]
    pipeline_errors["transmitted"] = max_error(fast.transmitted, direct.transmitted)

    # Fail visibly if a future change breaks the demonstration; do not merely
    # produce impressive-looking plots with an incorrect numerical result.
    np.testing.assert_allclose(fast_spectrum, slow_spectrum, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(idft(slow_spectrum), signal, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(ifft(fast_spectrum), signal, atol=1e-11, rtol=1e-11)
    for actual, expected in zip(fast.recovered, direct.recovered):
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    for check in convolution_checks:
        if max(check["dft_max_abs_error"], check["fft_max_abs_error"]) > 1e-10:
            raise AssertionError("frequency-domain convolution differs from direct")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "direct_config": asdict(config),
        "fft_config": asdict(fast.config),
        "transform_errors": transform_errors,
        "convolution_checks": convolution_checks,
        "pipeline_max_abs_errors": pipeline_errors,
        "direct_reconstruction_metrics": reconstruction_metrics(direct),
        "fft_reconstruction_metrics": reconstruction_metrics(fast),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].stem(np.arange(32), np.abs(slow_spectrum), linefmt="C0-", markerfmt="C0o", label="DFT")
    axes[0].plot(np.arange(32), np.abs(fast_spectrum), "C1x", label="FFT")
    axes[0].set_xlabel("Frequency bin k")
    axes[0].set_ylabel("|X[k]| (unnormalized)")
    axes[0].legend()
    axes[1].plot(n, np.abs(fast_spectrum - slow_spectrum))
    axes[1].set_xlabel("Frequency bin k")
    axes[1].set_ylabel("Absolute FFT–DFT difference")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "transforms.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for branch, ax in enumerate(axes):
        time = np.arange(len(fast.recovered[branch])) / config.sample_rate
        ax.plot(time, fast.recovered[branch] - direct.recovered[branch])
        ax.set_title(f"Channel {branch + 1}: full FFT output minus full direct output")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Sample difference")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pipeline_difference.png")
    plt.close(fig)

    print("Transform maximum absolute errors:", transform_errors)
    print("Convolution checks:", convolution_checks)
    print("Recovered pipeline maximum absolute errors:", pipeline_errors["recovered"])
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
