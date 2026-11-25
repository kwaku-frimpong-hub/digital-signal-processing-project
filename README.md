# DSP Project — Milestone 1

Only the foundations are implemented here. Transmission, audio I/O, and our
DFT/FFT algorithms belong to later milestones.

## Setup and run

From the `project/` directory, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run python experiments/milestone1.py
```

Alternatively, install with `python -m pip install -e . pytest` in a virtual
environment, then run `pytest` and `python experiments/milestone1.py`.

The demo saves three figures in `results/milestone1/`:

- `time_domain.png`: input, causal output, and delay-aligned comparison.
- `spectra.png`: input and filtered amplitude spectra.
- `filters.png`: coefficients and frequency responses of both windows.

Try a shorter filter and compare the figures:

```bash
uv run python experiments/milestone1.py --num-taps 11 --output-dir results/milestone1_short
```

Odd tap counts of at least 3 are supported. There is no automatic playback,
transmission, resampling, or hidden signal normalization.

## Suggested reading order

1. `src/dsp_project/signals.py` — time indices, sinusoids, mixtures, impulses.
2. `src/dsp_project/convolution.py` — explicit nested-loop convolution.
3. `src/dsp_project/filters.py` — windowed sinc, normalization, and filtering.
4. `tests/` — small examples of the properties each function must satisfy.
5. `experiments/milestone1.py` — puts the pieces together.
6. `src/dsp_project/plotting.py` — diagnostic figures.

The existing `main.py` scaffold is not the demo entry point; use the command above.

## What the demo does

At 8000 samples/second, generate 0.25 seconds of a 200 Hz sine plus a 2000 Hz
sine. Apply a 101-tap Hamming low-pass filter with cutoff 1000 Hz.
The low tone should remain while the high tone is strongly attenuated.

The rectangular and Hamming responses illustrate the tradeoff between
transition width and sidelobe suppression. Hamming is not uniformly better at
every frequency, and short filters do not behave like ideal brick-wall filters.

### Convolution and delay

For an input of N samples and a filter of M coefficients, full convolution
returns N + M - 1 samples. Samples outside each input are treated as zero.
The trailing tail is retained rather than silently discarded.

Our symmetric odd-length FIR has delay D = (M - 1)/2 samples. For 101 taps at
8000 Hz this is 50 samples, or 6.25 ms. The filtering routine retains this delay.
Only the comparison plot extracts `output[D:D+N]` to align signals.
The error calculation then excludes D samples at each boundary to avoid
comparing startup/shutdown transients with an infinite-duration reference.

### Filter formula

For `b = 2 * cutoff_hz / sample_rate`:

```text
D = (M - 1) / 2
h[n] = b * sinc(b * (n - D)),       n = 0, ..., M-1
sinc(u) = sin(pi*u) / (pi*u),       sinc(0) = 1
```

Multiply by either a rectangular window (all ones) or the symmetric Hamming
window `0.54 - 0.46*cos(2*pi*n/(M-1))`. Divide the coefficients by their sum
to preserve constant signals in steady state (unity DC gain).

Cutoff is in Hz, strictly below Nyquist (`sample_rate/2`). It marks the filter's
transition region; frequencies immediately below it are not necessarily
unchanged. Sine generation deliberately permits above-Nyquist frequencies for
future aliasing experiments.

## Validation and library boundaries

Tests cover known samples, impulse shifts, linearity, commutativity, real and
complex convolution, integer promotion, filter symmetry, DC gain, coefficient
formula, delay, pass/stop behavior, and a two-tone filtering example.

Production convolution is our own direct O(N*M) implementation. NumPy's
convolution is used only in tests as an independent reference.
NumPy's FFT is used only in diagnostic plots; no project DFT/FFT algorithm has
been implemented yet. Plotting spectra are one-sided, without a taper, so
finite-record leakage can be visible.
