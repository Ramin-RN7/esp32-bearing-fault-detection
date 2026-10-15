# Model Info — A8 → GMM (On-Device)

**Date:** 2025-08-14 16:38

## Summary
- **Model:** A8 → GMM (Gaussian Mixture Model) with on-device preprocessing.
- **Device:** M5Stack Core2 · MicroPython **v1.20.0-53** (2023-09-12).
- **Firmware libs:** `ulab` present; **`ulab.fft` not available** (features use slow DFT fallback).
- **Deployment files on device:**
  - `/models/A8_GMM/preprocess.json`
  - `/models/A8_GMM/gmm.json`
  - `/feat_a8.py` (A8 features)
  - `/model_rt_gmm.py` (preprocess + PCA + GMM)
  - `/time_gmm.py` (timing)

## Inputs (TEST WAVs)
- Healthy: **H_10_0_mic_16000Hz__seg0001.wav** → `/wav/healthy.wav`
- Anomaly: **B_11_2_mic_16000Hz__seg0001.wav** → `/wav/anomaly.wav`
- **Format:** 16 kHz, mono, 16‑bit PCM.

## Feature Pipeline (A8)
- **Sampling:** 16 kHz
- **Frame:** default 256 samples (≈16 ms); timing used **512** and **128** variants
- **Pre-DSP:** DC removal, Hann window
- **Spectrum:** 1–5 kHz band power (PSD); **25 equal-width bands**
  - Note: `ulab.fft` missing → fallback DFT loops (slow)
- **Output:** 25‑D feature vector per frame

## Preprocess (from `preprocess.json`)
- **Scaler:** **Z-score** (mean/scale per feature)
- **PCA:** **4 components** (25‑D → 4‑D)
- **Classes:** {0, 1}

## GMM (from `gmm.json`)
- **Covariance type:** `diag`
- **Parameters:** weights, means, covariances (match PCA dims)
- **Decision threshold (`thr`):** **−11.21834** (observed earlier)
- **Rule:** `y = 1 if loglik ≥ thr else 0`

## Timing (Core2)
- **GMM-only (preprocess+PCA+GMM)** — n=500: **63.3965 ms/call**, free RAM ≈ **3736 kB**
- **End‑to‑end (A8 features N=512 → GMM)** — n=20:
  - **Features:** **11221.24 ms/frame** (Python DFT bottleneck)
  - **GMM:** **63.9968 ms/call**
  - free RAM ≈ **2205 kB**

## Notes & Limitations
- Feature extractor not using FFT → slow and may not match PC features exactly.
- Device log‑likelihood scale is very negative; labels not calibrated yet.
- GMM compute itself is fine; feature step dominates runtime.

## PC Comparison
- **PC file‑F1:** A8‑GMM ≈ **0.909** (baseline B0)
- On device: functional path verified; accuracy pending DSP parity & threshold calibration.

## Reproduce (commands)
- **GMM timing:** `mpremote connect COM7 exec "import time_gmm as t; t.infer_only(500)"`
- **End‑to‑end timing:** `mpremote connect COM7 exec "import time_gmm as t; t.end_to_end(512, 20)"`
- **WAV CSV eval:** `mpremote connect COM7 exec "import a8_eval_wav_gmm as t; t.run_eval(N=128, hop=128)"`