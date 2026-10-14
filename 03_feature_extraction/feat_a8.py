# /feat_a8.py — A8 features (standard): Hann -> rFFT -> 1–5 kHz PSD -> 25 bands
from ulab import numpy as np
try:
    from ulab import fft
except:
    fft = None  # fallback handled

def _hann(N):
    w = [0.0]*N
    for i in range(N):
        w[i] = 0.5 - 0.5*np.cos(2*np.pi*i/(N-1))
    return w

def _take_frame(x, N):
    buf = [0.0]*N
    L = len(x)
    if L >= N:
        off = L - N
        for i in range(N):
            buf[i] = float(x[off+i])
    else:
        for i in range(L):
            buf[N-L+i] = float(x[i])
    # remove DC
    m = sum(buf)/N
    for i in range(N):
        buf[i] -= m
    # Hann
    w = _hann(N)
    for i in range(N):
        buf[i] *= w[i]
    return buf

def _psd_fft(buf, fs, N, k1, k2):
    X = fft.rfft(buf)  # len = N//2 + 1, complex
    out = [0.0]*((k2-k1)+1)
    for idx, k in enumerate(range(k1, k2+1)):
        z = X[k]            # complex
        re = z.real; im = z.imag
        out[idx] = (re*re + im*im) / N
    return out

def _psd_fallback(buf, fs, N, k1, k2):
    # very slow; only used if fft is missing
    out = [0.0]*((k2-k1)+1)
    for idx, k in enumerate(range(k1, k2+1)):
        re = 0.0; im = 0.0
        for n in range(N):
            ang = 2*np.pi*k*n/N
            c = np.cos(ang); s = np.sin(ang)
            v = buf[n]
            re += v*c
            im -= v*s
        out[idx] = (re*re + im*im)/N
    return out

def feat25(x, fs=16000, N=1024):
    buf = _take_frame(x, N)
    # bin range for 1..5 kHz
    k1 = int(1000*N/fs);
    k2 = int(5000*N/fs)
    if k1 < 1: k1 = 1
    if k2 > N//2: k2 = N//2
    if fft is not None:
        pks = _psd_fft(buf, fs, N, k1, k2)
    else:
        pks = _psd_fallback(buf, fs, N, k1, k2)

    # pool into 25 equal-width bands
    B = len(pks); bands = 25
    out = [0.0]*bands
    size = B//bands; rem = B%bands
    start = 0
    for b in range(bands):
        stop = start + size + (1 if b < rem else 0)
        s = 0.0; cnt = 0
        for i in range(start, stop):
            s += pks[i]; cnt += 1
        out[b] = s/cnt if cnt else 0.0
        start = stop
    return out
