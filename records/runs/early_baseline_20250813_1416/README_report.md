# Bearing ML Report Bundle

Created: 20250813_1416

## Parameters
- Sample rate: 16000 Hz
- Window: 2.0 s, overlap 50%
- NFFT: 4096
- Features: rms, zcr, centroid_hz, bandwidth_hz, flatness, kurtosis, crest, bp_0_1k, bp_1_3k, bp_3_8k, hf_ratio

## Test metrics
{
  "GMM_calibrated": {
    "acc": 0.7037037037037037,
    "prec": 0.7272727272727273,
    "rec": 0.8888888888888888,
    "f1": 0.8,
    "cm": [
      [
        9,
        18
      ],
      [
        6,
        48
      ]
    ]
  },
  "MLP_tiny_scaled": {
    "acc": 0.6049382716049383,
    "prec": 1.0,
    "rec": 0.4074074074074074,
    "f1": 0.5789473684210527,
    "cm": [
      [
        27,
        0
      ],
      [
        32,
        22
      ]
    ]
  }
}

## Files
- manifests/: segments, splits, features
- artifacts/: models (.joblib), calibration JSON, confusion matrices
- params.json, metrics.json
