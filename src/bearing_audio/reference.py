"""A5 MLP from PCM samples to the recording decision, written out stage by stage (2026 follow-up).

This is the frozen reference for porting A5 to a microcontroller (docs/DECISION_CONTRACT.md). It computes what
the archived A5 script computes (records/runs/A5-MLP_20250813_232809/code.py) on the segments that
prep_segments.py wrote, with every step spelled out so that a port can be compared stage by stage:

* librosa is not used. The zero-crossing rate, spectral roll-off and spectral flatness follow librosa's
  definitions: a 2048-point periodic Hann STFT, hop 512, frames centred with zero padding.
* The zero-phase notch filter is written out (odd extension, steady-state initial conditions, a forward and a
  backward pass) instead of calling scipy.signal.filtfilt.
* The autocorrelation computes only the 201 lags it uses instead of all 63,999.

scripts/freeze_reference.py checks the features against the archived feature code for all 540 windows (they
agree in every bit) and writes the models and golden outputs to reference/a5/. tests/test_reference.py repeats
the check on the golden recordings in CI (pushes to main and pull requests).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.fft
import soundfile as sf
from scipy.signal import butter, get_window, iirnotch, lfilter, lfilter_zi
from scipy.stats import kurtosis, skew

from .features import FEATURE_NAMES

# ---- input and windows (data.py, prep_segments.py) ----
SAMPLE_RATE = 16000
WINDOW = 32000                       # 2 s
HOP = 16000                          # 1 s
PEAK_TARGET = 10.0 ** (-1.0 / 20.0)  # -1 dBFS

# ---- front end (A5) ----
HPF_HZ, HPF_ORDER = 100.0, 5
HPF_B, HPF_A = butter(HPF_ORDER, HPF_HZ / (SAMPLE_RATE / 2), btype="high")
HPF_SOS = butter(HPF_ORDER, HPF_HZ / (SAMPLE_RATE / 2), btype="high", output="sos")   # same filter, for ports
PREEMPHASIS = 0.97
NOTCH_HZ, NOTCH_Q = 50.0, 30.0
NOTCH_B, NOTCH_A = iirnotch(NOTCH_HZ / (SAMPLE_RATE / 2), Q=NOTCH_Q)
NOTCH_ZI = lfilter_zi(NOTCH_B, NOTCH_A)
NOTCH_PAD = 3 * max(len(NOTCH_A), len(NOTCH_B))   # 9 samples: scipy.signal.filtfilt's default

# ---- features ----
BANDS_HZ = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000))
ZC_THRESHOLD = 1e-10                 # librosa.zero_crossings: |x| <= threshold counts as zero (positive)
N_FFT, STFT_HOP = 2048, 512          # librosa.feature.spectral_rolloff / spectral_flatness defaults
HANN = get_window("hann", N_FFT, fftbins=True)
ROLL_PERCENT = 0.85
FLATNESS_AMIN = 1e-10
MAX_LAG = 200

# ---- decision ----
THRESHOLD = 0.5                      # a recording is a fault when the mean window P(fault) is >= 0.5

# ---- checking a port against the golden outputs (docs/DECISION_CONTRACT.md) ----
PROBES = (0, 1, 2, 3, 4, 5, 10, 100, 1000, 8000, 16000, 24000, 31990, 31995, 31999)   # sample indices
# Signal stages: largest difference at the probes and in peak, rms and mean, as a fraction of the window's peak.
SIGNAL_TOLERANCE = {"highpass": 2e-4, "preemphasis": 1e-4, "notch": 4e-3}
# Features: as a fraction of the feature's spread (its standard deviation over all 540 windows). About ten times
# the largest difference of the float32 emulation (port32.py) over all 540 windows, rounded up to 1, 2 or 5 x 10^k
# and at least 0.001. The autocorrelation lag must match exactly.
FEATURE_TOLERANCE = dict(zip(FEATURE_NAMES, (
    0.5, 1e-3, 1e-3, 0.01, 5e-3, 0.1, 0.2, 1e-3, 1e-3, 0.1,       # time domain
    0.1, 0.02, 0.01, 0.02, 0.01, 5e-3, 0.01, 0.01,                 # band powers
    0.02, 0.02, 0.01, 1.0, 1e-3, 0.05, 0.0)))                      # spectral shape, autocorrelation
# Model stages computed from the golden features: standardised values, PCA scores and hidden activations within
# 1e-4 x max(1, |value|), P(fault) within 1e-5. End to end: window P(fault) within 0.05, recording score within 0.02.
MODEL_TOLERANCE = 1e-4
PROBABILITY_TOLERANCE = {"from_golden_features": 1e-5, "window": 0.05, "recording": 0.02}
# Reproducing the frozen files on another PC (not a board). They were computed on an x86-64 CPU on which NumPy uses
# AVX-512; elsewhere float32 sums round differently in the last bits. Features then move by up to about 2e-6 of
# their spread and recording scores by up to about 2.5e-6, with the same decisions. A reproduction must make the same
# decisions, with features within 1e-5 of their spread and probabilities and scores within 1e-4.
HOST_TOLERANCE = {"feature_fraction_of_spread": 1e-5, "probability": 1e-4}


# --------------------------------------------------------------------------------------------- PCM and windows

def read_pcm(path: Path) -> np.ndarray:
    """The recording's 16-bit samples, as stored in the WAV file."""
    s, sr = sf.read(path, dtype="int16", always_2d=False)
    if sr != SAMPLE_RATE or s.ndim != 1:
        raise ValueError(f"{path}: expected mono {SAMPLE_RATE} Hz")
    return s


def normalize(pcm: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Peak-normalise to -1 dBFS and requantise, as prep_segments.standardize() wrote data/wav16k/*.wav.

    Returns the new 16-bit samples and the gain, which is applied in float32. The conversion to int16 truncates
    toward zero.
    """
    y = pcm.astype(np.float32) / np.float32(32768.0)
    peak = float(np.max(np.abs(y)) + 1e-12)        # float32 sum: the 1e-12 is below its resolution
    gain = np.float32(min(1.0, PEAK_TARGET / peak))
    return (np.clip(y * gain, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16), gain


def windows(normalized: np.ndarray) -> list[np.ndarray]:
    """Two-second windows with a one-second hop as 16-bit samples, as prep_segments.segment_file() wrote them.

    The samples are read back with a different scale (/32767) than they are written with (*32767, truncated),
    so about half of them come out one step closer to zero. This is what the thesis pipeline did.
    """
    x = normalized.astype(np.float32) / np.float32(32767.0)
    return [(np.clip(x[s:s + WINDOW], -1.0, 1.0) * np.float32(32767.0)).astype(np.int16)
            for s in range(0, len(x) - WINDOW + 1, HOP)]


def window_signal(window: np.ndarray) -> np.ndarray:
    """A window as the A5 script read it: float32 samples, int16 / 32768."""
    return window.astype(np.float32) / np.float32(32768.0)


# --------------------------------------------------------------------------------------------- front end

def highpass(x: np.ndarray) -> np.ndarray:
    """Fifth-order Butterworth high-pass at 100 Hz, causal, starting from rest in every window (float64)."""
    return lfilter(HPF_B, HPF_A, x)


def preemphasis(h: np.ndarray) -> np.ndarray:
    """y[0] = h[0], y[n] = h[n] - 0.97 h[n-1], computed in float64 and stored as float32 (as in A5)."""
    out = np.empty_like(h, dtype=np.float32)
    out[0] = h[0]
    out[1:] = h[1:] - PREEMPHASIS * h[:-1]
    return out


def notch(p: np.ndarray) -> np.ndarray:
    """Zero-phase 50 Hz notch (Q 30): scipy.signal.filtfilt(NOTCH_B, NOTCH_A, p) with its defaults, written out.

    1. Odd extension by 9 samples at each end: 2*p[0] - p[9..1] in front, 2*p[-1] - p[-2..-10] behind.
    2. Forward pass starting from the steady state of a step of height ext[0]: zi = NOTCH_ZI * ext[0].
    3. Backward pass over the reversed output, starting from NOTCH_ZI times its first sample.
    4. Reverse again and drop the 9 extension samples at each end.
    """
    n = NOTCH_PAD
    ext = np.concatenate((2 * p[:1] - p[n:0:-1], p, 2 * p[-1:] - p[-2:-n - 2:-1]))
    fwd, _ = lfilter(NOTCH_B, NOTCH_A, ext, zi=NOTCH_ZI * ext[0])
    bwd, _ = lfilter(NOTCH_B, NOTCH_A, fwd[::-1], zi=NOTCH_ZI * fwd[-1])
    return bwd[::-1][n:-n]


def front_end(x: np.ndarray) -> dict[str, np.ndarray]:
    h = highpass(x)
    p = preemphasis(h)
    return {"highpass": h, "preemphasis": p, "notch": notch(p)}


# --------------------------------------------------------------------------------------------- features

def zero_crossing_rate(x: np.ndarray) -> float:
    """librosa.zero_crossings(x).sum() / len(x): sign changes, values within 1e-10 of zero counted as positive,
    and the first sample always counted (librosa's pad=True)."""
    neg = x.astype(np.float64) < -ZC_THRESHOLD
    return float(1 + np.count_nonzero(neg[1:] != neg[:-1])) / len(x)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """|STFT| as librosa computes it for float32 input: frames x 1025 bins, 63 frames for a 2 s window."""
    xp = np.pad(x, N_FFT // 2)                              # centred frames, zero padding
    n_frames = 1 + (len(xp) - N_FFT) // STFT_HOP
    frames = xp[np.arange(N_FFT)[None, :] + STFT_HOP * np.arange(n_frames)[:, None]]
    spec = scipy.fft.rfft(HANN * frames, axis=-1)            # float64 window x float32 frames
    return np.abs(spec.astype(np.complex64))                # librosa stores a float32 input's STFT as complex64


def spectral_rolloff(x: np.ndarray) -> float:
    """Mean over frames of the lowest bin frequency at which the cumulative magnitude reaches 85 % of the total."""
    S = stft_magnitude(x)
    cum = np.cumsum(S, axis=-1)
    first = np.argmax(cum >= ROLL_PERCENT * cum[:, -1:], axis=-1)
    return float(np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)[first].mean())


def spectral_flatness(x: np.ndarray) -> float:
    """Mean over frames of geometric mean / arithmetic mean of the power spectrum, floored at 1e-10."""
    P = np.maximum(FLATNESS_AMIN, stft_magnitude(x) ** 2)
    return float((np.exp(np.mean(np.log(P), axis=-1)) / np.mean(P, axis=-1)).mean())


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Autocorrelation of the mean-removed window at lags 0-200 (float32), with lag 0 set to zero."""
    y = x - x.mean()
    ac = np.array([np.dot(y[k:], y[:len(y) - k]) for k in range(MAX_LAG + 1)])
    ac[0] = 0.0
    return ac


def autocorrelation_peak(x: np.ndarray) -> tuple[float, int]:
    """Largest autocorrelation over lags 1-200 of the mean-removed window, divided by its energy, and its lag."""
    ac = autocorrelation(x)
    y = x - x.mean()
    k = int(np.argmax(ac))
    return float(ac[k] / (np.sum(y ** 2) + 1e-12)), k


def lag_gap(x: np.ndarray) -> float:
    """How clearly the chosen lag wins: (best - second best) / best over lags 1-200. Near 0 means a near tie."""
    top = np.sort(autocorrelation(x)[1:].astype(np.float64))[-2:]
    return float((top[1] - top[0]) / abs(top[1])) if top[1] else 0.0


def features25(signal: np.ndarray) -> np.ndarray:
    """The 25 A5 features of one front-end output, in FEATURE_NAMES order (float32, as the archived code)."""
    x = np.asarray(signal).astype(np.float32)
    if len(x) != WINDOW:
        raise ValueError(f"expected {WINDOW} samples, got {len(x)}")
    d = np.diff(x)
    feat = [float(np.mean(x)), float(np.std(x)), float(np.sqrt(np.mean(x ** 2) + 1e-12)),
            float(np.max(np.abs(x))), float(np.ptp(x)), float(skew(x)), float(kurtosis(x, fisher=True)),
            float(np.mean(np.abs(d))), float(np.mean(np.square(d))), zero_crossing_rate(x)]
    mag = np.abs(np.fft.rfft(x))                            # 16001 bins, 0.5 Hz apart (float32)
    power = mag ** 2
    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    feat += [float(power[(freqs >= lo) & (freqs < hi)].sum() / (power.sum() + 1e-12)) for lo, hi in BANDS_HZ]
    centroid = float((freqs * (mag + 1e-12)).sum() / (mag.sum() + 1e-12))
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * (mag + 1e-12)).sum() / (mag.sum() + 1e-12)))
    k = np.arange(len(mag))
    slope = float(((k - k.mean()) * (mag - mag.mean())).sum() / (((k - k.mean()) ** 2).sum() + 1e-12))
    ac_peak, ac_lag = autocorrelation_peak(x)
    feat += [centroid, bandwidth, spectral_rolloff(x), spectral_flatness(x), slope, ac_peak, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)


# --------------------------------------------------------------------------------------------- model

def export_model(fitted, **info) -> dict:
    """An A5 model (scaler, PCA, MLP) as plain numbers, with what a port needs to use it."""
    scaler, pca, mlp = fitted.scaler, fitted.pca, fitted.est
    if mlp.out_activation_ != "logistic" or mlp.activation != "relu" or len(mlp.coefs_) != 2:
        raise ValueError("expected one ReLU hidden layer and a logistic output")
    if list(mlp.classes_) != [0, 1]:
        raise ValueError("expected classes [0, 1]")
    f = lambda a: np.asarray(a, dtype=np.float64).tolist()
    return {
        "pipeline": "A5-MLP",
        **info,
        "features": list(FEATURE_NAMES),
        "standardize": {"mean": f(scaler.mean_), "scale": f(scaler.scale_)},
        "pca": {"mean": f(pca.mean_), "components": f(pca.components_)},
        "mlp": {"hidden_weights": f(mlp.coefs_[0]), "hidden_bias": f(mlp.intercepts_[0]), "hidden_activation": "relu",
                "output_weights": f(mlp.coefs_[1][:, 0]), "output_bias": float(mlp.intercepts_[1][0]),
                "output_activation": "logistic"},
        "decision": {"window_score": "P(fault)", "recording_score": "mean of the window scores",
                     "threshold": THRESHOLD, "fault_when": "recording score >= threshold",
                     "labels": {"0": "healthy", "1": "fault"}},
    }


def model_stages(features: np.ndarray, model: dict) -> dict[str, np.ndarray]:
    """Standardised features, PCA scores, hidden activations and P(fault) for one window, in float64."""
    z = (features.astype(np.float64) - model["standardize"]["mean"]) / np.asarray(model["standardize"]["scale"])
    p = (z - model["pca"]["mean"]) @ np.asarray(model["pca"]["components"]).T
    m = model["mlp"]
    h = np.maximum(0.0, p @ np.asarray(m["hidden_weights"]) + m["hidden_bias"])
    out = float(h @ np.asarray(m["output_weights"]) + m["output_bias"])
    return {"standardized": z, "pca": p, "hidden": h, "probability": np.array([1.0 / (1.0 + np.exp(-out))])}


def recording_decision(probabilities) -> tuple[float, int]:
    score = float(np.mean(probabilities))
    return score, int(score >= THRESHOLD)


# --------------------------------------------------------------------------------------------- trace

def trace(path: Path, model: dict | None = None) -> dict:
    """Every stage of every window of one recording. Without a model, the stages stop at the features."""
    normalized, gain = normalize(read_pcm(path))
    out = {"gain": gain, "normalized": normalized, "windows": []}
    for w in windows(normalized):
        st = {"pcm": w, **front_end(window_signal(w))}
        st["features"] = features25(st["notch"])
        if model is not None:
            st.update(model_stages(st["features"], model))
        out["windows"].append(st)
    if model is not None:
        out["score"], out["decision"] = recording_decision([w["probability"][0] for w in out["windows"]])
    return out


# --------------------------------------------------------------------------------------------- golden outputs

LAG_NEAR_TIE = 0.01      # a different lag is accepted (and reported) only when the golden lag_gap is below this


def pcm_summary(window: np.ndarray) -> dict:
    """A window's 16-bit samples, which a port must reproduce exactly (SHA-256 of their little-endian bytes)."""
    w = np.asarray(window, dtype=np.int16)
    return {"sha256": hashlib.sha256(w.astype("<i2").tobytes()).hexdigest(), "sum": int(w.astype(np.int64).sum()),
            "min": int(w.min()), "max": int(w.max()), "probes": [int(w[i]) for i in PROBES]}


def signal_summary(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"peak": float(np.abs(v).max()), "rms": float(np.sqrt(np.mean(v * v))), "mean": float(v.mean()),
            "probes": [float(v[i]) for i in PROBES]}


def golden_window(stages: dict, index: int) -> dict:
    """The golden record of one window: every stage, summarised where it is a 32,000-sample signal."""
    out = {"window": index + 1, "start_sample": index * HOP, "pcm": pcm_summary(stages["pcm"])}
    for s in ("highpass", "preemphasis", "notch"):
        out[s] = signal_summary(stages[s])
    out["features"] = [float(v) for v in stages["features"]]
    out["lag_gap"] = lag_gap(stages["notch"].astype(np.float32))
    for s in ("standardized", "pca", "hidden"):
        out[s] = [float(v) for v in stages[s]]
    out["probability"] = float(stages["probability"][0])
    return out


def _fraction(diff, tol: float) -> float:
    """A difference as a fraction of its tolerance. Anything that isn't a finite number (NaN, inf) fails: inf."""
    diff = float(diff)
    return diff / tol if np.isfinite(diff) else float("inf")


def _single(value, what: str) -> float:
    """The one number in a scalar or a one-element array (np.ravel(v)[0] would silently drop the rest)."""
    v = np.ravel(np.asarray(value, dtype=np.float64))
    if v.size != 1:
        raise ValueError(f"{what}: expected one number, got {v.size}")
    return float(v[0])


def check_window(stages: dict, golden: dict, spread) -> dict:
    """A port's window against its golden record: each difference as a fraction of its tolerance.

    ``stages`` holds any of the port's pcm, highpass, preemphasis, notch, features and probability for one window
    (missing ones are skipped; scripts/check_board.py requires features and probability); the first four can be
    arrays or their summaries (pcm_summary, signal_summary), as a board writes them. A value of 1.0 is at the limit
    and anything above fails, including NaN and infinite values (reported as inf). PCM must be identical (0 or inf);
    so must the lag (inf), except in a near tie (golden lag_gap below LAG_NEAR_TIE), which is reported as 1.0.
    """
    summary = lambda v, f: v if isinstance(v, dict) else f(v)
    r = {}
    if "pcm" in stages:
        r["pcm"] = 0.0 if summary(stages["pcm"], pcm_summary) == golden["pcm"] else float("inf")
    for s, tol in SIGNAL_TOLERANCE.items():
        if s in stages:
            mine, ref = summary(stages[s], signal_summary), golden[s]
            if len(mine["probes"]) != len(ref["probes"]):
                raise ValueError(f"{s}: expected {len(ref['probes'])} probes, got {len(mine['probes'])}")
            diffs = np.abs(np.concatenate([np.subtract(mine["probes"], ref["probes"], dtype=np.float64),
                                           [mine[k] - ref[k] for k in ("peak", "rms", "mean")]]))
            r[s] = _fraction(np.max(diffs), tol * ref["peak"])       # np.max keeps a NaN; Python's max may not
    if "features" in stages:
        r.update(check_features(stages["features"], golden["features"], spread, golden.get("lag_gap")))
    if "probability" in stages:
        r["probability"] = _fraction(abs(_single(stages["probability"], "probability") - golden["probability"]),
                                     PROBABILITY_TOLERANCE["window"])
    return r


def check_features(features, reference, spread, lag_gap_value=None) -> dict:
    """Each feature's difference as a fraction of its tolerance (FEATURE_TOLERANCE x spread); non-finite fails."""
    f, g = np.ravel(np.asarray(features, dtype=np.float64)), np.asarray(reference, dtype=np.float64)
    if f.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {f.size}")
    r = {}
    for j, name in enumerate(FEATURE_NAMES):
        d, tol = abs(f[j] - g[j]), FEATURE_TOLERANCE[name] * spread[j]
        if tol:
            r[name] = _fraction(d, tol)
        else:
            near_tie = lag_gap_value is not None and lag_gap_value < LAG_NEAR_TIE
            r[name] = 0.0 if d == 0 else (1.0 if near_tie and np.isfinite(d) else float("inf"))
    return r


def check_model(stages: dict, golden: dict) -> dict:
    """A port's model stages, computed from the golden features, against the golden values (fractions of tolerance)."""
    r = {}
    for s in ("standardized", "pca", "hidden"):
        mine, ref = np.ravel(stages[s]).astype(np.float64), np.asarray(golden[s])
        if mine.shape != ref.shape:
            raise ValueError(f"{s}: expected {ref.size} values, got {mine.size}")
        r[s] = _fraction(np.max(np.abs(mine - ref) / np.maximum(1.0, np.abs(ref))), MODEL_TOLERANCE)
    r["probability"] = _fraction(abs(_single(stages["probability"], "probability") - golden["probability"]),
                                 PROBABILITY_TOLERANCE["from_golden_features"])
    return r
