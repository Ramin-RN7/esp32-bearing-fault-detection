# Acoustic Bearing-Fault Detection and ESP32 Inference Prototype

[![tests](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/actions/workflows/tests.yml)

Can a microphone and a small microcontroller tell a healthy rolling bearing from a faulty one? This repository holds the master's thesis *Audio-Based Bearing Fault Detection on Resource-Constrained Microcontrollers: An Ablation-Driven Study* (Ramin Nassiri, OTH Regensburg, 2025). It contains:

- **The thesis study:** an ablation of 93 configurations (104 logged runs) of compact DSP features and lightweight classifiers on the University of Ottawa bearing-sound dataset, and a MicroPython inference prototype for an ESP32-based M5Stack Core2.
- **What you need to check it:** the original data split and run records, and a package that reproduces the thesis's two headline results (A8 GMM and A16 MLP) exactly and rebuilds 84 of the 86 ablation pipelines to their archived confusion counts.
- **Stricter follow-up evaluations (2026, not part of the thesis):** every ablation pipeline re-tested on bearings unseen in training. Choosing among them without seeing the test bearing (nested selection) reaches 0.950 balanced accuracy, against 0.762 for logistic regression on RMS and kurtosis.

**On the device:** the model arithmetic is verified on a PC, but the board computed different features from the ones the model was trained on, so on-device accuracy has not been measured ([details](#on-the-esp32-m5stack-core2)). For the next board, a [decision contract](docs/DECISION_CONTRACT.md) now fixes every step from audio samples to the decision, with golden outputs and tolerances.

## Quick demo

```bash
pip install -r requirements-lock.txt
python scripts/demo.py        # about 2 minutes the first time (feature extraction), seconds after that
```

It runs the best single pipeline, A5 MLP, through leave-one-bearing-out: 20 rounds, each deciding the three recordings of one held-out bearing with a model trained on 16 other bearings. Expected output:

```text
A5-MLP (100 Hz high-pass, pre-emphasis, 50 Hz notch, 25 features, z-score, PCA(4), MLP(10))
Leave-one-bearing-out: 20 folds, 60 recordings, each decided by a model that never heard its bearing

Bearing  Recording  Truth    Decision  Mean P(fault)
     13  H_13_0     healthy  fault             0.727  <- wrong
     19  C_19_2     fault    healthy           0.105  <- wrong

TP 39, FP 1, TN 19, FN 1: balanced accuracy 0.9625, fault F1 0.9750
```

`--all` lists all 60 decisions. A5 MLP was picked after seeing all 86 results, so 0.9625 is optimistic; nested selection (0.950, below) accounts for that choice.

## Results

Balanced accuracy of recording-level decisions (healthy vs. fault); 0.5 is chance level.

| Pipeline | Unseen bearings: leave-one-bearing-out, 60 recordings [95% interval] | Original split: 9 test recordings |
|---|---:|---:|
| **Nested selection** among all 86 ablation pipelines, chosen in each fold without the held-out bearing | **0.950** [0.89–1.00] | – |
| A5 MLP, the best single pipeline (picked with hindsight): 16 kHz, 100 Hz high-pass, pre-emphasis, 50 Hz notch, 25 time and spectral features, z-score, PCA(4), MLP with 10 units | 0.963 [0.90–1.00] | 0.917 |
| A16 MLP, selected in the thesis: 8 kHz, 1–3.5 kHz band-pass, Hilbert envelope + log, MinMax, PCA(8), MLP with 10 units | 0.850 [0.74–0.94] | 0.833 (F1 0.923) |
| A8 GMM, selected in the thesis and ported to the ESP32: 16 kHz, 1–5 kHz band-pass, z-score, PCA(4), 3-component GMM fitted on healthy audio | 0.762 [0.65–0.88] | 0.917 (F1 0.909) |
| Baseline: logistic regression on RMS and kurtosis | 0.762 [0.66–0.86] | 0.500 |

- **Original split:** the A16 and A8 entries reproduce the thesis's records exactly ([`tests/test_reproduction.py`](tests/test_reproduction.py)); the thesis reported their F1. The A5 and baseline entries on that split are 2026 calculations.
- **Intervals:** from resampling the 20 bearings with the predictions fixed. They condition on the fitted predictions and don't capture the variation from retraining or from the pipeline selection.
- **Against the baseline** (paired 95% intervals): nested selection +0.09 to +0.29; A16 MLP −0.03 to +0.20, so its lead is not established.
- **Nested selection** chose A5 MLP in 12 of 20 folds, A2 MLP in 4, A7 MLP in 3 and A38 MLP in 1.
- **Limit:** 20 bearings from one test rig and one microphone. This is not evidence for other machines, microphones or loads.

Other findings, in short:

- **A 9-recording test set is noisy.** Between random splits of that size, balanced accuracy has a standard deviation of about 0.15.
- **The thesis ranked by F1 on its test split.** By balanced accuracy on that split, A5 MLP was already first (0.917, against 0.833 for A16).
- **A8 GMM, the model ported to the ESP32, did no better than the baseline** on unseen bearings.
- **Splitting by recording instead of by bearing** changed mean balanced accuracy by less than 0.02 over 200 random splits. That doesn't prove the bearing overlap in the thesis split was harmless.

Details: [all 86 pipelines, nested selection and robustness checks](results/ablation_lobo.md); [the thesis pipelines under other protocols](results/protocol_comparison.md) (random splits, other validation bearings, no ball-fault bearings); [every logged ablation run](results/ablation_runs.csv); [what has been validated](docs/VALIDATION.md).

## On the ESP32 (M5Stack Core2)

| Part | Status |
|---|---|
| Firmware | The lab's custom MicroPython v1.20 image with ulab; the device measurements used its original build (v1.20.0-53). Rebuilt with ESP-IDF 4.4 to add model-inference modules: the rebuilt image booted, but those modules were not completed ([notes](records/firmware/)). |
| Model export (scaler, PCA, GMM → JSON) and the ulab runtime | Host check only: run on a PC with NumPy standing in for ulab, it matches scikit-learn on the desktop features ([`tests/test_device_runtime.py`](tests/test_device_runtime.py)). Not checked on the board. |
| On-device feature extraction | **Doesn't match the model.** The device computes 25 short-frame band energies; the model expects 25 window statistics. On-device accuracy has not been measured ([device notes](records/device/REPORT_DevicePhase.md)). |
| Timing (WAV from flash) | About 64 ms per prediction call, including reloading the model JSON every time; about 11.2 s per feature frame, because a pure-Python DFT was used. |

**Next:** one verified audio-to-decision path on an ESP32-DevKitC-VE.
1. **Done (18 Oct 2026):** the [decision contract](docs/DECISION_CONTRACT.md) and its frozen reference in [`reference/a5/`](reference/a5/). The reference code reproduces the archived A5 features in every bit without librosa. The 20 fold models reproduce the published decisions, and an all-bearings model is included for deployment. Golden outputs cover every stage of five recordings. A float32 emulation of a port meets every tolerance on all 540 windows with the same 60 decisions.
2. Port each stage to a pinned MicroPython + ulab build and check it with [`scripts/check_board.py`](scripts/check_board.py). A changed definition, such as a zero-padded FFT, becomes a named variant that is evaluated in its own right.
3. Run the recordings on the board in batches (the WAVs total 19.2 MB, more than the flash), compare every recording's decision with the host, and measure time and memory per stage.

## Reproduce

Tested with Python 3.13, the version CI uses.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python scripts/demo.py                                 # ~2 min: A5 MLP on unseen bearings (above)
python scripts/freeze_reference.py                     # ~1 min: checks that the frozen A5 reference (reference/a5/) reproduces
pytest -q                                              # ~4 min: thesis split, feature cache, fold integrity, ablation rebuild, device runtime, frozen reference
python scripts/evaluate_protocols.py                   # ~15 min: the follow-up evaluations, see docs/RUNNING.md
python scripts/evaluate_ablation.py                    # ~75 min: all 86 ablation pipelines on unseen bearings
```

## Repository map

| Path | Contents |
|---|---|
| [`src/bearing_audio/`](src/bearing_audio/) | Feature extraction, the two thesis pipelines, the evaluation protocols and the rebuild of every ablation pipeline |
| [`scripts/`](scripts/) | `demo.py`, `evaluate_protocols.py`, `evaluate_ablation.py`, `build_ablation_table.py`, `freeze_reference.py`, `trace_recording.py`, `check_board.py`, archive checks and host diagnostics |
| [`tests/`](tests/) | Reproduction, feature-cache, fold-integrity, ablation-rebuild, device-runtime, export, frozen-reference and board-checker tests, run by CI on pushes to `main` and on pull requests |
| [`results/`](results/) | Follow-up evaluation results, the ablation table rebuilt from the run records, and all 86 ablation pipelines on unseen bearings |
| [`reference/a5/`](reference/a5/) | The frozen A5 reference for a board port: models as JSON, golden outputs for every stage, features and decisions of all 60 recordings ([decision contract](docs/DECISION_CONTRACT.md)) |
| [`records/`](records/) | Original research records: split manifests, 108 run folders, logs, device files, firmware notes |
| [`01_uottawa_wavs/`](01_uottawa_wavs/) to [`07_confusion_matrices_full/`](07_confusion_matrices_full/) | The supplementary package as submitted: audio, segmentation, feature and ablation scripts, export, results, confusion matrices |
| [`docs/`](docs/) | [Known issues](docs/KNOWN_ISSUES.md), [what has been validated](docs/VALIDATION.md), [running the scripts](docs/RUNNING.md), [the decision contract for a board port](docs/DECISION_CONTRACT.md) |

## Thesis documents

- [Thesis (PDF)](Master_Thesis_Audio_ESP32.pdf)
- [Defense slides](Presentation%20Audio-Based%20Bearing%20Fault%20Detection.pptx)
- [Supplementary-materials index](Supplementary_Materials_Index_Ramin_Nassiri_MEM_OTH_Regensburg.pdf)

These are the documents as submitted and presented. Some of their statements go beyond what the code and records show:

- Section 4.2 of the thesis says the WAV test exercised the model "identically to the offline pipeline", and contribution 3 calls the device work "On-device validation".
- Slide 19 presents the PC results for A8 (F1 ≈ 0.90–0.91, "same accuracy as offline") as the board's, at about 70–75 ms per frame ("FFT fast, but MicroPython PCA+GMM still ~64 ms"). The 64 ms is measured; no FFT timing is archived. On the board, the features didn't match the model and the measured feature step took about 11.2 s per frame.
- Table 3.1 is headed "file-level F1", but some of its rows are window-level scores.
- The feature table (Table 2.1), the data-preparation listing and some conclusions differ from the code. For example, the real split keeps whole recordings together, which is stricter than the listing shows.

[Known issues](docs/KNOWN_ISSUES.md#5-thesis-text-versus-code-and-records) lists each one. None of them changes the reproduced scores above.

## Data

The audio comes from the University of Ottawa bearing dataset (UORED-VAFCLS, version 5) by Mert Sehri and Patrick Dumond, licensed CC BY 4.0. It has 60 ten-second microphone recordings, converted here to 16 kHz WAV, from 20 bearings. Each bearing was recorded healthy, with a developing fault and faulty. There are five bearings per fault type: inner race, outer race, ball and cage. See the [attribution](01_uottawa_wavs/ATTRIBUTION.md).

## Authorship

Ramin Nassiri carried out the 2025 thesis work kept in [`records/`](records/) (the 104 logged runs of 93 pipeline configurations, the ESP32 prototype and the rebuild of the lab's MicroPython firmware) and wrote the thesis, with help from ChatGPT and Claude. He also started the 2026 follow-up (the re-evaluation, tests, frozen reference and documentation), supplied the original records, corrected historical details such as the firmware account, and decided what to publish. The follow-up was developed with substantial AI assistance: Claude generated most of its code and text, and ChatGPT reviewed it. The original records and thesis documents are kept unchanged, separate from the later re-evaluation.

## License and citation

- **Code:** [MIT](LICENSE).
- **Thesis PDF, presentation and supplementary index:** © Ramin Nassiri, all rights reserved.
- **Audio:** CC BY 4.0, as above.

To cite this work, use [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button.
