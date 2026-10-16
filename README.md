# Acoustic Bearing-Fault Detection on an ESP32

[![tests](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml)

Can a microphone and a small microcontroller tell a healthy rolling bearing from a faulty one? This repository holds the master's thesis *Audio-Based Bearing Fault Detection on Resource-Constrained Microcontrollers: An Ablation-Driven Study* (Ramin Nassiri, OTH Regensburg, 2025). It contains:

- an ablation of 93 configurations (104 logged runs) of compact DSP features and lightweight classifiers on the University of Ottawa bearing-sound dataset
- a MicroPython prototype for an ESP32-based M5Stack Core2
- what you need to check the work: the original data split and run records, a package that reproduces the thesis numbers exactly, and stricter follow-up evaluations (2026, not part of the thesis)

## Results

The table shows balanced accuracy for recording-level decisions (healthy vs. fault). 0.5 is chance level. The first column reproduces the thesis. The other two are follow-up evaluations of the two thesis pipelines and a simple baseline.

| Pipeline | Thesis split (9 test recordings) | Leave-one-bearing-out (60 recordings) [95% interval] | 200 random splits of thesis size (mean ± SD) |
|---|---:|---:|---:|
| **A16 MLP** (8 kHz, 1–3.5 kHz band-pass, Hilbert envelope + log, MinMax, PCA(8), MLP with 10 units) | 0.833 (F1 0.923) | **0.850** [0.74–0.94] | 0.800 ± 0.143 |
| **A8 GMM** (16 kHz, 1–5 kHz band-pass, z-score, PCA(4), 3-component GMM fitted on healthy audio) | 0.917 (F1 0.909) | 0.762 [0.65–0.88] | 0.766 ± 0.152 |
| Baseline: logistic regression on RMS and kurtosis | 0.500 | 0.762 [0.66–0.86] | 0.776 ± 0.124 |

Brackets: 95% interval from resampling the 20 bearings, with the predictions fixed. ±: spread between random splits, not a confidence interval.

- **The thesis results reproduce exactly.** Retraining on the archived split gives the same confusion matrices ([`tests/test_reproduction.py`](tests/test_reproduction.py)).
- **A 9-recording test set is noisy.** Between random splits of that size, balanced accuracy varies with a standard deviation of about 0.15. The thesis split was a favourable draw for the GMM: only about a quarter of random splits score as high. It was an unfavourable draw for the baseline.
- **Splitting by recording instead of by bearing made little difference on average.** In the thesis split, recordings of one physical bearing appear in both training and test. Over 200 random splits of the same size, mean balanced accuracy differed by less than 0.02 between splitting by recording and by bearing. That rules out a large average inflation. It doesn't prove the overlap is harmless or measure its effect on the thesis split ([details](results/protocol_comparison.md)).
- **A16 MLP scores highest on unseen bearings, but its lead isn't established.** Leave-one-bearing-out gives 0.850 (fault F1 0.883, against 0.923 on the thesis split). Its paired difference from the two-feature baseline is −0.03 to +0.20. The baseline matches it on bearing-grouped splits (0.794 vs. 0.790) and without the ball-fault bearings, which were recorded without load (0.817 vs. 0.800).
- **The choice of validation bearings matters.** Each leave-one-bearing-out fold sets aside 3 bearings for validation. With ten other choices of those bearings, balanced accuracy ranged 0.762–0.887 for the MLP, 0.675–0.800 for the GMM and 0.787–0.800 for the baseline, against 0.850, 0.762 and 0.762 with the published choice. The MLP scored above the baseline in 7 of the 10.
- **A8 GMM is no better than the baseline.** This is the model that was ported to the ESP32.

Full tables, every fold and held-out score: [`results/protocol_comparison.md`](results/protocol_comparison.md) and `.json`. Every logged ablation run (104 rows, 93 configurations), with configuration fields as recorded and notes where the run's code shows otherwise: [`results/ablation_runs.csv`](results/ablation_runs.csv).

## On the ESP32 (M5Stack Core2)

| Part | Status |
|---|---|
| Custom MicroPython v1.20 image with ulab (ESP-IDF 4.4) | Built and flashed. See the [build notes](records/firmware/). |
| Model export (scaler, PCA, GMM → JSON) and the ulab runtime | Host check passes: run on a PC with NumPy standing in for ulab, it matches scikit-learn on identical inputs ([`tests/test_device_runtime.py`](tests/test_device_runtime.py)). Not yet checked on the board. |
| On-device feature extraction | **Doesn't match the model yet.** The device computes 25 short-frame band energies, but the model was trained on 25 window statistics. On-device classification accuracy has not been measured ([device notes](records/device/REPORT_DevicePhase.md)). |
| Timing (WAV from flash) | About 63 ms per prediction call, including reloading the model JSON on every call. About 11.2 s per feature frame, because a pure-Python DFT was used even though `ulab.numpy.fft` was available. |

**Next:** one validated audio-to-decision path on the board.
1. Define one feature set the device can compute with FFT, and train on exactly those features on the host. Compare against a small linear baseline as well.
2. Add golden test vectors (audio → features → score) that must match on the host and the board.
3. Load the model once instead of on every call, and use ulab's FFT.
4. Evaluate the same recordings on an ESP32 (an ESP32-DevKitC-VE is on order), with timings and memory for each stage.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pytest -q                                              # ~3 min: thesis split, feature cache, fold integrity, device runtime
python scripts/evaluate_protocols.py                   # ~15 min: the follow-up evaluations, see docs/RUNNING.md
```

## Repository map

| Path | Contents |
|---|---|
| [`src/bearing_audio/`](src/bearing_audio/) | Feature extraction, the two thesis pipelines and the evaluation protocols |
| [`scripts/`](scripts/) | `evaluate_protocols.py`, `build_ablation_table.py`, archive checks, earlier review diagnostics |
| [`tests/`](tests/) | Reproduction, feature-cache, fold-integrity, device-runtime and export tests, run on every push |
| [`results/`](results/) | Follow-up evaluation results and the ablation table rebuilt from the run records |
| [`records/`](records/) | Original research records: split manifests, 108 run folders, logs, device files, firmware notes |
| [`01_uottawa_wavs/`](01_uottawa_wavs/) to [`07_confusion_matrices_full/`](07_confusion_matrices_full/) | The supplementary package as submitted: audio, segmentation, feature and ablation scripts, export, results, confusion matrices |
| [`docs/`](docs/) | [Technical review and known issues](docs/TECHNICAL_REVIEW.md), [validation scope](docs/VALIDATION.md), [running the archived scripts](docs/RUNNING.md) |

## Thesis documents

- [Thesis (PDF)](Master_Thesis_Audio_ESP32.pdf)
- [Defense slides](Presentation%20Audio-Based%20Bearing%20Fault%20Detection.pptx)
- [Supplementary-materials index](Supplementary_Materials_Index_Ramin_Nassiri_MEM_OTH_Regensburg.pdf)

These are the documents as submitted and presented. Some of their statements go beyond what the code and records show:

- Section 4.2 of the thesis says the WAV test exercised the model "identically to the offline pipeline".
- Slide 19 reports about 70–75 ms per frame at offline accuracy.
- The feature table (Table 2.1), the data-preparation listing and some conclusions differ from the code. For example, the real split keeps whole recordings together, which is stricter than the listing shows.

The [technical review](docs/TECHNICAL_REVIEW.md#5-thesis-text-versus-code-and-records) lists each one. None of them changes the reproduced scores above.

## Data

The audio comes from the University of Ottawa bearing dataset (UORED-VAFCLS, version 5) by Mert Sehri and Patrick Dumond, licensed CC BY 4.0. It has 60 ten-second microphone recordings, converted here to 16 kHz WAV, from 20 bearings. Each bearing was recorded healthy, with a developing fault and faulty. There are five bearings per fault type: inner race, outer race, ball and cage. See the [attribution](01_uottawa_wavs/ATTRIBUTION.md).

## License and citation

- **Code:** [MIT](LICENSE).
- **Thesis PDF, presentation and supplementary index:** © Ramin Nassiri, all rights reserved.
- **Audio:** CC BY 4.0, as above.

To cite this work, use [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button.
