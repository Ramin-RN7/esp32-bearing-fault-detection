# Device Phase Report — Core2 / MicroPython 1.20

**Date:** 2025-08-14

## TL;DR
- **Exports completed:** A44-ET (emlearn header), A45-RF (stub header), A16-MLP (C header), A8-GMM (JSON).  
- **On-device:** Copied preprocess JSONs; implemented **A8→GMM** runtime with ulab; verified live loop runs.  
- **Status:** **A8→GMM end-to-end runs**, but **accuracy not validated** (features don’t match PC; ll ~ -1e7 → always 0).  
- **emlearn + MicroPython 1.20:** modules present; **RF/ET/MLP not callable yet** (no wrapper compiled). If success criterion is “end-to-end RF/ET/MLP inference from MicroPython,” this was **not achieved** in this iteration.

## Artifacts
- `artifacts/exports/A44_ET/model_et.h`, `.../preprocess.json`  
- `artifacts/exports/A45_RF/model_rf.h`, `.../preprocess.json` (from ET stub)  
- `artifacts/exports/A16_MLP/model_mlp.h`, `.../preprocess.json`  
- `artifacts/exports/A8_GMM/gmm.json`, `.../preprocess.json`  
- Device files: `/models/.../*.json`, `/model_rt_gmm.py`, `/feat_a8.py`, `/a8_live_gmm.py`, `/a8_loop_gmm.py`

## What worked
- Exporters produced headers/JSON.  
- JSONs copied to device (verified via mpremote).  
- GMM runtime (preprocess→PCA→log-likelihood) executed on-device.  
- Live loop produced continuous predictions.

## What didn't (and why)
- **GMM predictions not meaningful**: ll around `-10^7` with `thr ≈ -11.21834` → always class `0`.  
  **Cause:** device A8 features are simplified DFT pooling; **don’t match** PC A8 features (window/FFT/PSD/bands).  
- **RF/ET/MLP not callable from MicroPython**: emlearn present, but **model headers + wrapper** were not compiled into firmware.

## Conclusion
- We successfully exported models and built a working **A8→GMM pipeline** on-device.  
- **Parity with PC** not established yet; **RF/ET/MLP** inference from MicroPython **not integrated**.  
- It is accurate to write: _“Adding emlearn-backed model inference to MicroPython 1.20 was **not completed** (no callable RF/ET/MLP from Python).”_

## Suggested next steps
1. **Golden-check A8** with a PC 25-D vector → device preprocess+GMM.  
2. **Match A8 features to PC** (FFT/PSD + exact bands), then calibrate `thr`.  
3. **Rebuild once** with a tiny wrapper exposing `rf_predict/et_predict/mlp_predict` (emlearn).  
4. **Device-sim vs PC**: short TEST set, log `file-F1`, timing, RAM/flash.