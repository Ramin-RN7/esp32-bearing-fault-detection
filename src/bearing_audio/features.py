"""Feature extraction used by the thesis pipelines (same logic as the archived ablation scripts).

* ``features25``: the 25-feature set from the lab's vibration PDMS, adapted to audio: 10 time-domain statistics,
  8 relative band powers and 7 spectral/autocorrelation descriptors of one two-second window.
* ``a8_front``: A8 front end, a 4th-order Butterworth band-pass 1-5 kHz (zero-phase) at 16 kHz.
* ``a16_front``: A16 front end, resampled to 8 kHz, band-passed 1-3.5 kHz, Hilbert envelope, 20 ms smoothing,
  log1p after 95th-percentile normalisation and RMS matching.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sig
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew

from .data import Recording, load_windows

FEATURE_NAMES = (
    "mean", "std", "rms", "max_abs", "peak_to_peak", "skewness", "kurtosis", "mean_abs_diff", "mean_sq_diff",
    "zero_crossing_rate",
    "bp_0_250", "bp_250_500", "bp_500_1k", "bp_1k_2k", "bp_2k_3k", "bp_3k_4k", "bp_4k_6k", "bp_6k_8k",
    "centroid_hz", "bandwidth_hz", "rolloff_hz", "flatness", "spectral_slope", "autocorr_peak", "autocorr_lag",
)
_BANDS = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000))


def _bandpower(x, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    idx = (freqs >= fmin) & (freqs < fmax)
    return float(S[idx].sum() / (S.sum() + 1e-12))


def _spectral_slope(x):
    S = np.abs(np.fft.rfft(x))
    k = np.arange(len(S))
    km, sm = k.mean(), S.mean()
    return float(((k - km) * (S - sm)).sum() / (((k - km) ** 2).sum() + 1e-12))


def _autocorr(x, max_lag=200):
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + max_lag]
    if len(ac) < 2:
        return 0.0, 0
    ac[0] = 0.0
    k = int(np.argmax(ac))
    return float(ac[k] / (np.sum(x ** 2) + 1e-12)), k


def features25(y: np.ndarray, sr: int, win_s: float = 2.0) -> np.ndarray:
    if len(y) < int(sr * win_s):
        y = np.pad(y, (0, int(sr * win_s) - len(y)))
    x = y.astype(np.float32)
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    feat = [
        float(np.mean(x)), float(np.std(x)), rms, float(np.max(np.abs(x))), float(np.ptp(x)),
        float(skew(x)), float(kurtosis(x, fisher=True)),
        float(np.mean(np.abs(np.diff(x)))), float(np.mean(np.square(np.diff(x)))),
        float(np.count_nonzero(librosa.zero_crossings(x))) / len(x),
    ]
    feat += [_bandpower(x, sr, f1, f2) for f1, f2 in _BANDS]
    S = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    centroid = float((freqs * (S + 1e-12)).sum() / (S.sum() + 1e-12))
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * (S + 1e-12)).sum() / (S.sum() + 1e-12)))
    rolloff = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flatness = float(librosa.feature.spectral_flatness(y=x).mean())
    ac_val, ac_lag = _autocorr(x)
    feat += [centroid, bandwidth, rolloff, flatness, _spectral_slope(x), ac_val, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)


def a8_front(y16: np.ndarray) -> np.ndarray:
    b, a = butter(4, [1000 / 8000, 5000 / 8000], btype="band")
    return filtfilt(b, a, y16)


def a16_front(y16: np.ndarray) -> np.ndarray:
    y = librosa.resample(y16, orig_sr=16000, target_sr=8000).astype(np.float32)
    sos = sig.iirfilter(N=4, Wn=(1000, 3500), btype="bandpass", fs=8000, output="sos")
    ybp = sig.sosfiltfilt(sos, y).astype(np.float32)
    env = np.abs(sig.hilbert(ybp)).astype(np.float32)
    k = max(1, int(round(20 * 8000 / 1000.0)))
    env_s = np.convolve(env, np.ones(k, dtype=np.float32) / k, mode="same")
    env_ln = np.log1p(env_s / (np.percentile(env_s, 95) + 1e-8)).astype(np.float32)
    rms_in, rms_env = np.sqrt(np.mean(y ** 2) + 1e-12), np.sqrt(np.mean(env_ln ** 2) + 1e-12)
    return env_ln * (rms_in / rms_env)


class FeatureTable:
    """Window-level features for all recordings, with the metadata needed to group and split them."""

    def __init__(self, recordings: list[Recording], cache: Path | None = None):
        self.recordings = {r.name: r for r in recordings}
        key = hashlib.sha256("|".join(f"{r.name}:{r.path.stat().st_size}" for r in recordings).encode()).hexdigest()[:16]
        if cache is not None and Path(cache).exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                self._set(z["a8"], z["a16"], z["rec"].tolist())
                return
        a8, a16, names = [], [], []
        for r in recordings:
            for w in load_windows(r):
                a8.append(features25(a8_front(w), 16000))
                a16.append(features25(a16_front(w), 8000))
                names.append(r.name)
        self._set(np.vstack(a8), np.vstack(a16), names)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, key=key, a8=self.a8, a16=self.a16, rec=np.array(self.rec))

    def _set(self, a8, a16, names):
        self.a8, self.a16 = a8, a16
        self.rec = np.array(names)
        self.label = np.array([self.recordings[n].label for n in names])
        self.bearing = np.array([self.recordings[n].bearing for n in names])

    def features(self, kind: str) -> np.ndarray:
        return {"a8": self.a8, "a16": self.a16}[kind]

    def mask(self, names) -> np.ndarray:
        return np.isin(self.rec, list(names))
