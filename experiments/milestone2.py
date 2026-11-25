"""Synthetic single-channel or two-channel direct-convolution transmission."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp_project.pipeline import TransmissionConfig, reconstruction_metrics, transmit
from dsp_project.plotting import plot_spectrum, plot_time
from dsp_project.signals import sine_mixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-channels", choices=("1", "2", "both"), default="both")
    parser.add_argument("--sample-rate", type=float, default=16000)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--cutoff-hz", type=float, default=1000)
    parser.add_argument("--carriers", type=float, nargs=2, default=(2000, 5000))
    parser.add_argument("--num-taps", type=int, default=101)
    parser.add_argument("--window", choices=("hamming", "rectangular"), default="hamming")
    parser.add_argument("--guard-hz", type=float, default=250)
    parser.add_argument("--allow-unsafe-bands", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/milestone2"))
    args = parser.parse_args()
    try:
        config = TransmissionConfig(
            sample_rate=args.sample_rate,
            cutoff_hz=args.cutoff_hz,
            carrier_hz=tuple(args.carriers),
            num_taps=args.num_taps,
            window=args.window,
            guard_hz=args.guard_hz,
            allow_unsafe_bands=args.allow_unsafe_bands,
        )
        # Distinct baseband tones, with an out-of-band component in each input
        # to demonstrate that sender low-pass filtering is actually active.
        first = sine_mixture([200, 3500], config.sample_rate, args.duration, [0.7, 0.2])
        second = sine_mixture([600, 3500], config.sample_rate, args.duration, [0.7, 0.2])
        if args.active_channels == "1":
            second = np.zeros_like(second)
        elif args.active_channels == "2":
            first = np.zeros_like(first)
        result = transmit(first, second, config)
        metrics = reconstruction_metrics(result)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": asdict(config),
        "duration_seconds": args.duration,
        "active_channels": args.active_channels,
        "input_tones_hz": [[200, 3500], [600, 3500]],
        "input_amplitudes": [0.7, 0.2],
        "receiver_delay_samples": result.receiver_delay,
        "total_delay_samples": result.total_delay,
        "channels": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = {"coefficients": result.coefficients, "transmitted": result.transmitted}
    for name in ("inputs", "sender_filtered", "modulated", "demodulated", "recovered"):
        for branch, values in enumerate(getattr(result, name), start=1):
            arrays[f"{name}_{branch}"] = values
    np.savez(args.output_dir / "signals.npz", **arrays)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for branch in range(2):
        plot_time(axes[0, branch], result.inputs[branch], config.sample_rate, "Raw input", 640)
        plot_time(axes[1, branch], result.recovered[branch], config.sample_rate, "Causal recovered", 640)
        plot_time(axes[2, branch], result.sender_filtered[branch], config.sample_rate, "Sender-filtered reference", 640)
        plot_time(axes[2, branch], result.aligned_recovered[branch], config.sample_rate, "Receiver-delay aligned", 640)
        axes[0, branch].set_title(f"Channel {branch + 1}")
    fig.suptitle("Two-channel transmission: full causal outputs, comparison alignment only")
    fig.tight_layout()
    fig.savefig(args.output_dir / "reconstruction.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for branch in range(2):
        plot_spectrum(axes[0, branch], result.sender_filtered[branch], config.sample_rate, "Sender-filtered")
        plot_spectrum(axes[1, branch], result.modulated[branch], config.sample_rate, "Modulated")
        plot_spectrum(axes[2, branch], result.recovered[branch], config.sample_rate, "Recovered")
        axes[0, branch].set_title(f"Channel {branch + 1}")
    fig.tight_layout()
    fig.savefig(args.output_dir / "stage_spectra.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_spectrum(ax, result.transmitted, config.sample_rate, "Shared medium")
    for carrier in config.carrier_hz:
        ax.axvspan(carrier - config.cutoff_hz, carrier + config.cutoff_hz, alpha=0.12)
    ax.set_title("Nominal positive-frequency bands (shaded); FIR skirts are not zero")
    fig.tight_layout()
    fig.savefig(args.output_dir / "transmitted_spectrum.png")
    plt.close(fig)

    print(f"Active channels: {args.active_channels}; backend: direct convolution")
    print(f"Receiver delay: {result.receiver_delay}; total delay: {result.total_delay} samples")
    for branch, metric in enumerate(metrics, start=1):
        print(f"Channel {branch}: steady RMSE={metric['steady_rmse']:.6g}, "
              f"relative RMSE={metric['steady_relative_rmse']}")
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
