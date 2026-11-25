"""Small reproducible filtering demonstration; no transmission system yet."""

import argparse
from pathlib import Path

import matplotlib

# Save figures without requiring a desktop/display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp_project.filters import apply_fir, lowpass_fir
from dsp_project.plotting import plot_filter_response, plot_spectrum, plot_time
from dsp_project.signals import sine_mixture, sine_wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-taps", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, default=Path("results/milestone1"))
    args = parser.parse_args()

    sample_rate = 8000
    duration = 0.25
    cutoff = 1000
    samples = sine_mixture([200, 2000], sample_rate, duration)
    low_tone = sine_wave(200, sample_rate, duration)
    hamming = lowpass_fir(cutoff, sample_rate, args.num_taps, "hamming")
    rectangular = lowpass_fir(cutoff, sample_rate, args.num_taps, "rectangular")
    filtered = apply_fir(samples, hamming)
    delay = (args.num_taps - 1) // 2

    # A view for comparison only. The returned filter output itself is untouched.
    aligned = filtered[delay : delay + len(samples)]
    # Exclude boundaries: a centered FIR needs delay samples on either side.
    if len(samples) <= 2 * delay:
        parser.error("num-taps is too large to leave a steady-state comparison")
    interior = slice(delay, len(samples) - delay)
    rmse = np.sqrt(np.mean((aligned[interior] - low_tone[interior]) ** 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    plot_time(axes[0], samples, sample_rate, "Input: 200 Hz + 2000 Hz", 240)
    plot_time(axes[1], filtered, sample_rate, "Causal filtered output", 240)
    plot_time(axes[2], aligned, sample_rate, "Delay-aligned output", 160)
    plot_time(axes[2], low_tone, sample_rate, "Ideal 200 Hz reference", 160)
    fig.suptitle(f"Hamming low-pass: {args.num_taps} taps, delay {delay} samples")
    fig.tight_layout()
    fig.savefig(args.output_dir / "time_domain.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_spectrum(axes[0], samples, sample_rate, "Input")
    plot_spectrum(axes[1], aligned, sample_rate, "Filtered (including boundaries)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "spectra.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(hamming, label="Hamming")
    axes[0].plot(rectangular, label="Rectangular", alpha=0.7)
    axes[0].set_xlabel("Coefficient index n")
    axes[0].set_ylabel("h[n]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plot_filter_response(axes[1], hamming, sample_rate, "Hamming")
    plot_filter_response(axes[1], rectangular, sample_rate, "Rectangular")
    axes[1].axvline(cutoff, color="black", linestyle="--", label="Cutoff")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "filters.png")
    plt.close(fig)

    print(f"Filter: {args.num_taps} taps, cutoff {cutoff} Hz, fs {sample_rate} Hz")
    print(f"Delay: {delay} samples ({delay / sample_rate:.6f} s)")
    print(f"Lengths: input={len(samples)}, full output={len(filtered)}")
    print(f"Steady-state RMSE against ideal 200 Hz tone: {rmse:.6g}")
    print(f"Figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
