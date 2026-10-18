"""The A5 reference in single precision: what a microcontroller port computes (2026 follow-up).

MicroPython on the ESP32 and ulab compute in float32, add up arrays element by element and only have a
power-of-two FFT. This module computes every A5 stage that way, so that the tolerances in
docs/DECISION_CONTRACT.md can be checked on a PC before a board is involved. It is also a template for a
ulab port:

* sums and dot products accumulate in float32, one element at a time;
* the gain for peak normalisation is computed in float32;
* the high-pass runs as second-order sections (scipy.signal.sosfilt and ulab's sosfilt use the same form);
* the notch's forward-backward pass uses the reference's odd extension and initial conditions;
* the exact 32,000-point DFT comes from Bluestein's algorithm with 65,536-point FFTs; the chirp phase is reduced
  modulo 2N in integers first, because float32 cannot hold n**2 for n up to 32,000;
* the STFT uses a float32 Hann window and 2048-point FFTs.

It is not a model of ulab's FFT rounding or of the ESP32's FPU, so a board can differ from it slightly; the
contract's tolerances leave room for that.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from . import reference as R

F32 = np.float32
N = R.WINDOW
HPF_SOS = R.HPF_SOS.astype(F32)
NOTCH_SOS = np.concatenate((R.NOTCH_B, R.NOTCH_A))[None, :].astype(F32)
NOTCH_ZI = sosfilt_zi(NOTCH_SOS).astype(F32)
SLOPE_DENOMINATOR = F32(2 * sum(j * j for j in range(1, N // 4 + 1)))   # sum of (k - 8000)**2 over 16001 bins


def seq_sum(v) -> np.float32:
    """Sum in float32, one element at a time, as a C loop over a float array does."""
    v = np.asarray(v, dtype=F32)
    return np.cumsum(v, dtype=F32)[-1] if len(v) else F32(0)


def seq_mean(v) -> np.float32:
    return F32(seq_sum(v) / F32(len(v)))


# --------------------------------------------------------------------------------------------- PCM

def normalize(pcm: np.ndarray) -> tuple[np.ndarray, np.float32]:
    y = pcm.astype(F32) / F32(32768.0)
    gain = min(F32(1.0), F32(F32(R.PEAK_TARGET) / np.max(np.abs(y))))
    return (np.clip(y * gain, F32(-1.0), F32(1.0)) * F32(32767.0)).astype(np.int16), gain


# --------------------------------------------------------------------------------------------- front end

def highpass(x: np.ndarray) -> np.ndarray:
    return sosfilt(HPF_SOS, x.astype(F32))


def preemphasis(h: np.ndarray) -> np.ndarray:
    out = np.empty_like(h, dtype=F32)
    out[0] = h[0]
    out[1:] = h[1:] - F32(R.PREEMPHASIS) * h[:-1]
    return out


def notch(p: np.ndarray) -> np.ndarray:
    n = R.NOTCH_PAD
    p = p.astype(F32)
    ext = np.concatenate((F32(2) * p[:1] - p[n:0:-1], p, F32(2) * p[-1:] - p[-2:-n - 2:-1]))
    fwd, _ = sosfilt(NOTCH_SOS, ext, zi=NOTCH_ZI * ext[0])
    bwd, _ = sosfilt(NOTCH_SOS, fwd[::-1].copy(), zi=NOTCH_ZI * fwd[-1])
    return bwd[::-1][n:-n]


def front_end(x: np.ndarray) -> dict[str, np.ndarray]:
    h = highpass(x)
    p = preemphasis(h)
    return {"highpass": h, "preemphasis": p, "notch": notch(p)}


# --------------------------------------------------------------------------------------------- spectra

class Bluestein:
    """Exact N-point DFT of a real signal from power-of-two FFTs (here N = 32000, FFT size 65536), in float32."""

    def __init__(self, n: int = N):
        self.n, self.m = n, 1 << (2 * n - 1).bit_length()
        idx = (np.arange(n, dtype=np.int64) ** 2) % (2 * n)                 # exact in integers
        ang = F32(np.pi) * idx.astype(F32) / F32(n)
        self.chirp = (np.cos(ang) - 1j * np.sin(ang)).astype(np.complex64)   # exp(-i pi n^2 / N)
        b = np.zeros(self.m, dtype=np.complex64)
        b[:n] = np.conj(self.chirp)
        b[self.m - n + 1:] = np.conj(self.chirp[1:])[::-1]
        self.B = np.fft.fft(b)

    def rfft(self, x: np.ndarray) -> np.ndarray:
        a = np.zeros(self.m, dtype=np.complex64)
        a[:self.n] = x.astype(F32) * self.chirp
        conv = np.fft.ifft(np.fft.fft(a) * self.B)
        return (self.chirp[:self.n // 2 + 1] * conv[:self.n // 2 + 1]).astype(np.complex64)


_BLUESTEIN = None
HANN32 = (F32(0.5) - F32(0.5) * np.cos(F32(2 * np.pi) * np.arange(R.N_FFT, dtype=F32) / F32(R.N_FFT))).astype(F32)


def dft32000(x: np.ndarray) -> np.ndarray:
    global _BLUESTEIN
    if _BLUESTEIN is None:
        _BLUESTEIN = Bluestein()
    return _BLUESTEIN.rfft(x)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x.astype(F32), R.N_FFT // 2)
    n_frames = 1 + (len(xp) - R.N_FFT) // R.STFT_HOP
    frames = xp[np.arange(R.N_FFT)[None, :] + R.STFT_HOP * np.arange(n_frames)[:, None]]
    return np.abs(np.fft.rfft(HANN32 * frames, axis=-1)).astype(F32)


# --------------------------------------------------------------------------------------------- features

def features25(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal).astype(F32)
    n = F32(len(x))
    m = seq_mean(x)
    d0 = x - m
    m2, m3, m4 = seq_sum(d0 ** 2) / n, seq_sum(d0 ** 3) / n, seq_sum(d0 ** 4) / n
    d = np.diff(x)
    neg = x < F32(-R.ZC_THRESHOLD)
    feat = [m, np.sqrt(m2), np.sqrt(seq_mean(x * x) + F32(1e-12)), np.max(np.abs(x)), np.max(x) - np.min(x),
            m3 / m2 ** F32(1.5), m4 / (m2 * m2) - F32(3), seq_mean(np.abs(d)), seq_mean(d * d),
            F32(1 + np.count_nonzero(neg[1:] != neg[:-1])) / n]
    mag = np.abs(dft32000(x)).astype(F32)
    power = mag * mag
    freqs = (np.arange(len(mag), dtype=F32) * F32(0.5))
    total = seq_sum(power) + F32(1e-12)
    feat += [seq_sum(power[(freqs >= lo) & (freqs < hi)]) / total for lo, hi in R.BANDS_HZ]
    msum = seq_sum(mag) + F32(1e-12)
    centroid = seq_sum(freqs * (mag + F32(1e-12))) / msum
    bandwidth = np.sqrt(seq_sum((freqs - centroid) ** 2 * (mag + F32(1e-12))) / msum)
    k = np.arange(len(mag), dtype=F32)
    slope = seq_sum((k - F32(N // 4)) * (mag - seq_mean(mag))) / SLOPE_DENOMINATOR
    S = stft_magnitude(x)
    cum = np.cumsum(S, axis=-1, dtype=F32)
    first = np.argmax(cum >= F32(R.ROLL_PERCENT) * cum[:, -1:], axis=-1)
    rolloff = seq_mean(first.astype(F32) * F32(R.SAMPLE_RATE / R.N_FFT))
    P = np.maximum(F32(R.FLATNESS_AMIN), S * S)
    flat = [np.exp(seq_mean(np.log(row))) / seq_mean(row) for row in P]
    y = x - m
    ac = np.array([seq_sum(y[j:] * y[:len(y) - j]) for j in range(R.MAX_LAG + 1)], dtype=F32)
    ac[0] = 0
    lag = int(np.argmax(ac))
    feat += [centroid, bandwidth, rolloff, seq_mean(flat), slope, ac[lag] / (seq_sum(y * y) + F32(1e-12)), F32(lag)]
    return np.asarray(feat, dtype=F32)


def autocorrelation(signal: np.ndarray) -> np.ndarray:
    """All 201 lags (lag 0 set to 0), for checking near ties."""
    x = np.asarray(signal).astype(F32)
    y = x - seq_mean(x)
    ac = np.array([seq_sum(y[j:] * y[:len(y) - j]) for j in range(R.MAX_LAG + 1)], dtype=F32)
    ac[0] = 0
    return ac


# --------------------------------------------------------------------------------------------- model

def model_stages(features: np.ndarray, model: dict) -> dict[str, np.ndarray]:
    a = lambda v: np.asarray(v, dtype=F32)
    z = (features.astype(F32) - a(model["standardize"]["mean"])) / a(model["standardize"]["scale"])
    comp = a(model["pca"]["components"])
    p = np.array([seq_sum((z - a(model["pca"]["mean"])) * row) for row in comp], dtype=F32)
    m = model["mlp"]
    W = a(m["hidden_weights"])
    h = np.maximum(F32(0), np.array([seq_sum(p * W[:, j]) for j in range(W.shape[1])], dtype=F32) + a(m["hidden_bias"]))
    out = seq_sum(h * a(m["output_weights"])) + F32(m["output_bias"])
    e = np.exp(-abs(out))                                   # logistic without overflow in float32
    prob = F32(1) / (F32(1) + e) if out >= 0 else e / (F32(1) + e)
    return {"standardized": z, "pca": p, "hidden": h, "probability": np.array([prob], dtype=F32)}
