# DSP Project — Milestones 1 and 2

Foundations and the direct-convolution transmission system are implemented.
Audio I/O and our DFT/FFT algorithms belong to later milestones.

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
resampling, or hidden signal normalization.

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

## Milestone 2 — two-channel transmission

```bash
uv run python experiments/milestone2.py
uv run python experiments/milestone2.py --active-channels 1 --output-dir results/milestone2_single
uv run python experiments/milestone2.py --active-channels 2 --output-dir results/milestone2_second
```

Read these files in order:

1. `src/dsp_project/modulation.py` — cosine carriers and the factor 2 at reception.
2. `src/dsp_project/pipeline.py` — configuration, full chain, intermediate arrays,
   explicit delays, and reconstruction metrics.
3. `tests/test_pipeline.py` — single-channel leakage, two-channel recovery,
   independent NumPy reference, and configuration checks.
4. `experiments/milestone2.py` — plots and saved numerical results.

### Pipeline

```text
input 1 -> low-pass -> multiply by cos(2*pi*f1*n/fs) --+
                                                     +-> transmitted sum
input 2 -> low-pass -> multiply by cos(2*pi*f2*n/fs) --+

sum -> multiply by 2*cos(2*pi*f1*n/fs) -> low-pass -> recovered 1
sum -> multiply by 2*cos(2*pi*f2*n/fs) -> low-pass -> recovered 2
```

The shared medium is addition only: no noise, phase offset, or propagation
delay. Both inputs must be real, finite, nonempty, and equally long.

Default configuration: fs = 16 kHz, cutoff = 1 kHz, carriers = 2 and 5 kHz,
101 Hamming taps, and nominal guard = 250 Hz. The demo uses baseband tones of
200 and 600 Hz. Each input also includes a small 3500 Hz tone to show the
sender filter removing an out-of-band component.

Positive-frequency transmission bands are nominally 1–3 and 4–6 kHz.
The configuration checks zero/Nyquist crossings, band spacing, and folded
demodulation images at `2*f1`, `2*f2`, `f1+f2`, and `abs(f1-f2)`.
These checks use the nominal cutoff, not a claimed perfect band limit.
Finite filters have skirts; inspect their response and reconstruction errors.
The plots use a library FFT for diagnostics only, as in Milestone 1.

For later failure experiments, `--allow-unsafe-bands` permits unsafe frequency
plans with a warning. It does not bypass malformed signal/filter parameters.
All major settings can be changed; run `--help` for the list.

### Delays and comparisons

For N raw samples, M taps, and D = (M-1)/2:

| Stage | Length | Delay relative to raw input |
| --- | --- | --- |
| Sender-filtered / modulated / transmitted / mixed | N + M - 1 | D |
| Recovered | N + 2*(M-1) | 2D |

Delay here refers to the baseband envelope, not a shifted carrier clock.
Every carrier starts at the same absolute sample index zero. We do not
subtract the sender delay from the receiver's carrier phase.

The PDF asks for comparison to the **sender-filtered** inputs. Consequently,
`aligned_recovered` removes only the additional receiver delay D. It does not
alter the full recovered arrays. Receiver filtering can still add ripple and
attenuation after alignment, so perfect reconstruction is not expected.

Metrics include:

- RMSE over the complete delay-aligned sender reference (including boundaries).
- Steady-state RMSE and relative RMSE, without gain normalization.
- Explicit steady-state indices `[3D, N-D)` on the sender-output timeline.

The steady interval excludes both FIR boundary regions. It requires N > 4D;
shorter signals can still be transmitted but cannot use these steady metrics.
For an inactive channel, relative error is undefined (`null` in JSON) and
absolute RMSE measures leakage from the active channel.

### Saved results

`results/milestone2/` contains:

- `reconstruction.png`: raw signals, causal outputs, and aligned comparisons.
- `stage_spectra.png`: sender-filtered, modulated, and recovered spectra.
- `transmitted_spectrum.png`: combined spectrum with nominal bands shaded.
- `metrics.json`: complete configuration, active inputs, delays, and errors.
- `signals.npz`: coefficients and all intermediate arrays, without scaling.

For example, inspect arrays interactively with
`np.load("results/milestone2/signals.npz")`; array names are listed in `.files`.
No WAV output or optimized convolution backend is included in this milestone.
