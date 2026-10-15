# Acoustic Bearing-Fault Detection on an ESP32

[![tests](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml)

Can a microphone and a small microcontroller tell a healthy rolling bearing from a faulty one? This repository holds the master's thesis *Audio-Based Bearing Fault Detection on Resource-Constrained Microcontrollers: An Ablation-Driven Study* (Ramin Nassiri, OTH Regensburg, 2025). It contains:

- a 93-run ablation of compact DSP features and lightweight classifiers on the University of Ottawa bearing-sound dataset
- a MicroPython prototype for an ESP32-based M5Stack Core2
- what you need to check the work: the original data split and run records, a package that reproduces the thesis numbers exactly, and stricter re-evaluations

## Results

The table shows balanced accuracy for recording-level decisions (healthy vs. fault). 0.5 is chance level.

| Pipeline | Thesis split (9 test recordings) | Leave-one-bearing-out (60 recordings) | 200 random splits of thesis size |
|---|---:|---:|---:|
| **A16 MLP** (8 kHz, 1–3.5 kHz band-pass, Hilbert envelope + log, MinMax, PCA(8), MLP with 10 units) | 0.833 (F1 0.923) | **0.850** (F1 0.883) | 0.800 ± 0.143 |
| **A8 GMM** (16 kHz, 1–5 kHz band-pass, z-score, PCA(4), 3-component GMM fitted on healthy audio) | 0.917 (F1 0.909) | 0.762 (F1 0.854) | 0.766 ± 0.152 |
| Baseline: logistic regression on RMS and kurtosis | 0.500 | 0.762 | 0.776 ± 0.124 |

- **The thesis results reproduce exactly.** Retraining on the archived split gives the same confusion matrices ([`tests/test_reproduction.py`](tests/test_reproduction.py)).
- **A 9-recording test set is noisy.** Between random splits of that size, balanced accuracy varies by about ±0.15. The thesis split was a favourable draw for the GMM: only about a quarter of random splits score as high. It was an unfavourable draw for the baseline.
- **Bearing overlap does not inflate the scores.** In the thesis split, recordings of one physical bearing appear in both training and test. Matched splits that keep bearings apart score the same, within 0.02 ([details](results/protocol_comparison.md)).
- **A16 MLP is the strongest thesis pipeline.** It reaches 0.79–0.85 balanced accuracy on unseen bearings.
- **A two-feature baseline comes close to the MLP.** It matches it in two cases: on bearing-grouped splits (0.794 vs. 0.790), and when the ball-fault bearings, which were recorded without load, are left out (0.817 vs. 0.800).
- **A8 GMM is no better than the baseline.** This is the model that was ported to the ESP32.

Full tables: [`results/protocol_comparison.md`](results/protocol_comparison.md). Every logged ablation run, with corrected configuration metadata: [`results/ablation_runs.csv`](results/ablation_runs.csv).

## On the ESP32 (M5Stack Core2)

| Part | Status |
|---|---|
| Custom MicroPython v1.20 image with ulab (ESP-IDF 4.4) | Built and flashed. See the [build notes](records/firmware/). |
| Model export (scaler, PCA, GMM → JSON) and the ulab runtime | Works. On identical inputs it matches scikit-learn ([`tests/test_device_runtime.py`](tests/test_device_runtime.py)). |
| On-device feature extraction | **Doesn't match the model yet.** The device computes 25 short-frame band energies, but the model was trained on 25 window statistics. On-device classification accuracy has not been measured ([device notes](records/device/REPORT_DevicePhase.md)). |
| Timing (WAV from flash) | About 63 ms per prediction call, including reloading the model JSON on every call. About 11.2 s per feature frame, because a pure-Python DFT was used even though `ulab.numpy.fft` was available. |

**Next:**
1. Retrain the GMM on features the device computes with FFT.
2. Load the model once instead of on every call.
3. Add host–device parity tests.
4. Re-measure on the board.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pytest -q                                              # ~3 min: thesis split, split integrity, device runtime
python scripts/evaluate_protocols.py                   # ~10 min: writes results/protocol_comparison.md
```

## Repository map

| Path | Contents |
|---|---|
| [`src/bearing_audio/`](src/bearing_audio/) | Feature extraction, the two thesis pipelines and the evaluation protocols |
| [`scripts/`](scripts/) | `evaluate_protocols.py`, `build_ablation_table.py`, archive checks, earlier review diagnostics |
| [`tests/`](tests/) | Reproduction, split-integrity and device-runtime tests, run on every push |
| [`results/`](results/) | Evaluation results and the corrected ablation table |
| [`records/`](records/) | Original research records: split manifests, 101 run folders, logs, device files, firmware notes |
| [`01_uottawa_wavs/`](01_uottawa_wavs/) to [`07_confusion_matrices_full/`](07_confusion_matrices_full/) | The supplementary package as submitted: audio, segmentation, feature and ablation scripts, export, results, confusion matrices |
| [`docs/`](docs/) | [Technical review and known issues](docs/TECHNICAL_REVIEW.md), [validation scope](docs/VALIDATION.md), [running the archived scripts](docs/RUNNING.md) |

## Thesis documents

- [Thesis (PDF)](Master_Thesis_Audio_ESP32.pdf)
- [Defense slides](Presentation%20Audio-Based%20Bearing%20Fault%20Detection.pptx)
- [Supplementary-materials index](Supplementary_Materials_Index_Ramin_Nassiri_MEM_OTH_Regensburg.pdf)

These are the documents as submitted and presented. Two of their on-device statements go beyond the archived evidence:

- Section 4.2 of the thesis says the WAV test exercised the model "identically to the offline pipeline".
- Slide 19 reports about 70–75 ms per frame at offline accuracy.

The [technical review](docs/TECHNICAL_REVIEW.md) explains both.

## Data

The audio comes from the University of Ottawa bearing dataset (UORED-VAFCLS, version 5) by Mert Sehri and Patrick Dumond, licensed CC BY 4.0. It has 60 ten-second microphone recordings, converted here to 16 kHz WAV, from 20 bearings. Each bearing was recorded healthy, with a developing fault and faulty. There are five bearings per fault type: inner race, outer race, ball and cage. See the [attribution](01_uottawa_wavs/ATTRIBUTION.md).

## License and citation

- **Code:** [MIT](LICENSE).
- **Thesis PDF, presentation and supplementary index:** © Ramin Nassiri, all rights reserved.
- **Audio:** CC BY 4.0, as above.

To cite this work, use [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button.
