# Decision contract: A5 MLP from audio samples to the decision

Frozen on 18 October 2026, before any board run. It defines every step of the A5 MLP pipeline, from the 16-bit samples of a WAV file to the healthy/fault decision. It also fixes golden outputs for each step and the tolerances a port must meet. A port that meets them makes the PC's decisions. Only then is the board's result an A5 result.

- Reference implementation: [`src/bearing_audio/reference.py`](../src/bearing_audio/reference.py)
- Frozen files: [`reference/a5/`](../reference/a5/), written by [`scripts/freeze_reference.py`](../scripts/freeze_reference.py)
- Checks: [`tests/test_reference.py`](../tests/test_reference.py) on every push

## 1. Why

The earlier ESP32 prototype computed different features from the ones its model was trained on, so its outputs had no meaning ([KNOWN_ISSUES §1](KNOWN_ISSUES.md#1-desktop-and-device-features-have-different-meanings)). This contract prevents that. Each stage is defined exactly and has numbers to compare against, so a mismatch shows up at the stage where it starts.

## 2. What is frozen

| File | Contents |
|---|---|
| `contract.json` | Every constant below (filter coefficients, STFT settings, feature order, spread, threshold), the tolerances, the probe indices and the SHA-256 of all 60 WAV files. A port reads its constants from here. |
| `models/lobo_bearing_NN.json` | The 20 leave-one-bearing-out models: z-score, PCA(4) and MLP(10), each trained on 16 bearings with seed 42, exactly as in [results/ablation_lobo.md](../results/ablation_lobo.md). |
| `models/all_bearings.json` | The same pipeline trained on all 20 bearings: the model to deploy. Its decisions on these 60 recordings are on training data, so they are not an accuracy. |
| `window_features.csv` | The 25 features of all 540 windows, the lag gap (§4), and P(fault) from the window's fold model and from the all-bearings model. |
| `decisions.csv` | The 60 recording decisions of both, with the margin to the threshold. |
| `golden/<recording>.json` | Every stage of every window of five recordings (below). |
| `float32_check.json` | How close a single-precision implementation comes to the reference (§4). |

Two checks run before anything is written:

- The reference features equal the archived A5 script's features **in every bit**, for all 540 windows. The reference uses no librosa, writes the zero-phase notch out and computes only the 201 autocorrelation lags it needs.
- The fold models' decisions and scores equal the published leave-one-bearing-out run: 58 of 60 correct, balanced accuracy 0.9625.

The golden recordings:

| Recording | Truth | Out-of-fold result | Why it is golden |
|---|---|---|---|
| H_10_0 | healthy | healthy (P 0.009) | Correct; one of the two WAVs used on the earlier board |
| B_11_2 | fault | fault (P 0.641) | Correct, but the closest of all 60 to the threshold (margin 0.141); the other earlier-board WAV |
| H_13_0 | healthy | fault (P 0.727) | One of A5's two errors |
| C_19_2 | fault | healthy (P 0.105) | The other error |
| H_2_0 | healthy | healthy (P < 0.001) | Its recording has a DC offset of about 0.20 of full scale, so every window starts with a spike (§3, step 4) |

## 3. The pipeline

All arrays are one window of 32,000 samples unless stated. "float32" means the arithmetic is single precision; that matters in steps 2, 3 and 5.

| Step | Definition | Notes for a port |
|---|---|---|
| 1. Input | WAV, 16-bit signed PCM, little-endian, mono, 16 kHz; 10 s = 160,000 samples | Read the `data` chunk; check `audio_sha256` in `contract.json` |
| 2. Normalise | `y = s / 32768` (float32); `gain = float32(min(1, 0.8912509 / max abs(y)))`; `n = int16(trunc(clip(y * gain, -1, 1) * 32767))` | Truncate toward zero. The gain is 1 for most recordings |
| 3. Windows | `x = n / 32767` (float32); window k = 0…8 is `x[16000·k : 16000·k + 32000]`; `w = int16(trunc(clip(window, -1, 1) * 32767))`; `signal = w / 32768` | Writing ×32767 and reading ÷32767 moves about half the samples one step toward zero. That is what the thesis pipeline did; keep it |
| 4. High-pass | Butterworth, order 5, 100 Hz, causal, **starting from rest in every window** | Second-order sections (`sos` in `contract.json`) in float32 meet the tolerance. Do not carry the filter state across windows: with it, the start of each window changes. H_2_0 and H_19_0 have DC offsets of 0.20 and 0.18, so their windows start with a large step response; the model was trained on that |
| 5. Pre-emphasis | `y[0] = h[0]`, `y[n] = h[n] − 0.97·h[n−1]`, stored as float32 | |
| 6. Notch | 50 Hz, Q 30, zero-phase: extend by 9 samples at each end with odd symmetry (`2·p[0] − p[9…1]` in front, `2·p[−1] − p[−2…−10]` behind); run forward from state `zi·ext[0]`; reverse; run again from `zi·(first sample)`; reverse; drop the 9 extra samples at each end | Equals `scipy.signal.filtfilt` with its defaults. `b`, `a` and `zi` are in `contract.json` |
| 7. Features | 25 values in the order below, from the notch output as float32 | See the list below |
| 8. Standardise | `z = (f − mean) / scale` | From the model file |
| 9. PCA | `p = (z − pca_mean) · componentsᵀ` | 4 scores |
| 10. MLP | `h = ReLU(p · W₁ + b₁)` (10 units); `P(fault) = 1 / (1 + exp(−(h · w₂ + b₂)))` | In float32, `exp` overflows for large negative inputs; for `a < 0` use `exp(a) / (1 + exp(a))` |
| 11. Decision | Recording score = mean of the 9 window probabilities; **fault when score ≥ 0.5**; label 1 = fault | The earlier device runtime used 1 = healthy; this contract uses 1 = fault |

The features, in order:

1. **Time domain (10):** mean; standard deviation; RMS = `sqrt(mean(x²) + 1e-12)`; maximum absolute value; peak-to-peak; skewness and excess kurtosis (both with the biased moment estimates); mean absolute difference of neighbouring samples; mean squared difference; zero-crossing rate. The zero-crossing rate is `(1 + number of sign changes) / 32000`. Values within 1e-10 of zero count as positive, and the 1 counts the first sample, as librosa does.
2. **Band powers (8):** the share of the total power `|X[k]|²` in 0–250, 250–500, 500–1000, 1000–2000, 2000–3000, 3000–4000, 4000–6000 and 6000–8000 Hz. Each band includes its lower edge and excludes its upper edge. `X` is the **exact 32,000-point DFT** (16,001 bins, 0.5 Hz apart). Zero-padding to 32,768 changes the features (§5). One way to get the exact DFT from power-of-two FFTs is Bluestein's algorithm with 65,536-point FFTs. Reduce the chirp index `n² mod 64000` in integers before converting it to an angle, because float32 cannot represent `n²`. Three 65,536-point complex arrays take about 1.5 MB, so they belong in PSRAM.
3. **Spectral shape and autocorrelation (7):** centroid and bandwidth (in Hz) of `|X|`. Roll-off and flatness are means over 63 STFT frames: periodic Hann window, 2048 points, hop 512, 1024 zeros added at each end. Roll-off is the lowest bin frequency at which the cumulative `|X|` reaches 85 % of the frame total. Flatness is the geometric mean of `max(1e-10, |X|²)` divided by its arithmetic mean. The spectral slope is the least-squares slope of `|X|` against the bin index (0…16000). For the autocorrelation, remove the window's mean and compute lags 1–200. The peak feature is the largest value divided by the window's energy, and the lag feature is that lag (with lag 0 set to zero before the search).

## 4. Tolerances

| Check | Tolerance |
|---|---|
| PCM (steps 2–3) | Identical: the SHA-256 of each window's 16-bit samples, plus their sum, minimum, maximum and 15 probe samples |
| High-pass, pre-emphasis, notch | Largest difference at the 15 probe samples and in peak, RMS and mean, as a fraction of the window's peak: 2e-4, 1e-4 and 4e-3 |
| Features | Difference ≤ tolerance × the feature's spread, its standard deviation over the 540 windows (`spread` in `contract.json`). 0.001: std, RMS, mean absolute and mean squared difference, slope. 0.005: peak-to-peak, 3–4 kHz. 0.01: maximum, 0.5–1, 2–3, 4–6 and 6–8 kHz, roll-off. 0.02: 0.25–0.5 and 1–2 kHz, centroid, bandwidth. 0.05: autocorrelation peak. 0.1: skewness, zero-crossing rate, 0–250 Hz. 0.2: kurtosis. 0.5: mean. 1: flatness |
| Autocorrelation lag | Identical. In a near tie, where the best lag beats the second by less than 1 % (`lag_gap` below 0.01; 21 of the 540 windows), a different lag is reported but allowed, and the probability check decides |
| Model stages, from the golden features | Standardised values, PCA scores and hidden activations within 1e-4 × max(1, abs(value)); P(fault) within 1e-5 |
| End to end | P(fault) per window within 0.05; recording score within 0.02; **all 60 decisions identical** |

How they were set: [`src/bearing_audio/port32.py`](../src/bearing_audio/port32.py) computes every stage the way a microcontroller port would. It uses float32 throughout and adds element by element, runs the high-pass as second-order sections and gets the 32,000-point DFT from Bluestein's algorithm. Over all 540 windows it lands at most 0.28 of any tolerance from the reference. The largest difference is 0.012 in a window's P(fault) and 0.006 in a recording score, and all 60 decisions are the same. The feature tolerances are about ten times its largest difference.

The tolerances are in the model's own units: the spread is what its scaler divides by. Skewness, kurtosis and the lowest band powers have a few extreme windows (H_2_0's and H_19_0's spikes give kurtosis values up to about 7,800). That makes their spread large, so small slips in their definitions pass, for example Pearson instead of excess kurtosis. Such slips also move the decision very little. The end-to-end checks cover the decision.

The checks do catch typical porting mistakes. In a test on B_11_2 with the float32 implementation, each of these failed several checks: a high-pass that carries its state across windows (up to 4,300 × the pre-emphasis tolerance), a forward-only notch (P off by 0.09 in one window), and an FFT zero-padded to 32,768 points (band powers up to 95 × their tolerance). The forward-only notch is part of `tests/test_reference.py`.

## 5. Named variants

A port that changes a definition computes a different pipeline. Examples: a zero-padded FFT, streaming filters, another window length, live microphone input, fixed-point arithmetic, fewer windows per decision. Such a variant gets a name (for example `A5-fft32768`), its own features on the PC, its own leave-one-bearing-out evaluation and its own frozen reference. Its accuracy is reported as that variant's, not as A5's.

## 6. Checking a board

1. Put on the board: `contract.json`, the model files it needs and the WAV files. The 60 WAVs take 19.2 MB, more than the 8 MB flash, so copy them in batches.
2. Model code first: feed the golden features of each window to the board's model code. Record the standardised values, PCA scores, hidden activations and P(fault) (`model_from_golden` below). This tests the model arithmetic on its own.
3. Then the golden recordings, stage by stage, then all 60 with their fold models.
4. For each recording the board writes one JSON file:

```json
{"recording": "H_13_0", "model": "models/lobo_bearing_13.json",
 "windows": [{"window": 1,
              "pcm": {"sha256": "…", "sum": -6432200, "min": -5003, "max": 6779, "probes": [-241, …]},
              "highpass": {"peak": …, "rms": …, "mean": …, "probes": […]},
              "preemphasis": {…}, "notch": {…},
              "features": [25 numbers], "probability": 0.596}, …],
 "model_from_golden": [{"window": 1, "standardized": […], "pca": […], "hidden": […], "probability": …}, …],
 "score": 0.727, "decision": 1}
```

   Only `features` and `probability` are required per window. The other stages are checked when present, and only for golden recordings. The probes are the samples at the indices in `contract.json`. `peak` is the largest absolute value.
5. On the PC: `python scripts/check_board.py board/*.json`. It prints PASS or FAIL per recording, each check as a fraction of its tolerance, and exits with status 1 if anything fails.
6. When all 60 decisions match, the board has reproduced the leave-one-bearing-out result: the same 60 predictions, so the same 0.9625. This is not a new estimate. Then time each stage with the model loaded once, and record the memory used.

For debugging, `python scripts/trace_recording.py H_13_0 --float32` shows every window and the float32 emulation's worst check. `--window 3 --save w3.npz` writes all stages of one window as arrays.

## 7. Not covered

A live microphone path, other machines or rigs, real-time behaviour, and the earlier M5Stack Core2 firmware. A5's accuracy itself remains a 20-bearing, one-rig estimate, selected after seeing all 86 results ([README](../README.md#results)).
