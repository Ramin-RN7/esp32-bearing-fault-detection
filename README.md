# ESP32 Acoustic Bearing Fault Detection

Python DSP and machine-learning ablation experiments, with an experimental MicroPython/ulab scoring prototype for an ESP32-based M5Stack Core2.

Supplementary materials for **Ramin Nassiri's master's thesis at OTH Regensburg (2025)**: *Audio-Based Bearing Fault Detection on Resource-Constrained Microcontrollers: An Ablation-Driven Study*.

**Technical review, 15 October 2026:** the archived device extractor produces different features from those used to train the exported GMM. The embedded audio-to-decision path therefore needs correction and validation. The offline scores below describe a small recording-level experiment. Read the [technical review and known issues](docs/TECHNICAL_REVIEW.md) before using the models or interpreting the timings.

## Overview

The study investigates acoustic bearing-fault detection for condition monitoring. An ablation framework compares preprocessing, handcrafted and spectral feature representations, scaling, PCA, GMM/MLP classifiers and file-level aggregation. The archive includes a desktop-to-JSON parameter exporter and a MicroPython scaler/PCA/GMM runtime.

The embedded prototype uses **WAV files stored in flash**. Its measurements concern the archived implementation, and embedded classification accuracy has not been established by this repository review.

## Historical offline results

| Archived configuration | File-level fault F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| A16 MLP | 0.923 | 0.889 | 0.833 |
| A8 GMM | 0.909 | 0.889 | 0.917 |

These results correspond to **9 test recordings: 3 normal and 6 fault**, with 81 overlapping two-second windows. Both models classify 8 of the 9 recordings correctly. The higher MLP fault F1 reflects a different error type, while GMM has higher balanced accuracy on this split. Always predicting fault gives F1 = 0.800.

A host reconstruction on 15 October 2026 reproduced these file-level scores using the archived model bundles, the available segmentation/split code and published WAV filenames. The original manifests are absent, so the reconstructed split is explicitly identified as such. These results do not establish accuracy on unseen bearings: the reconstructed split places 6 test recordings on physical bearings also represented in training or validation. Comparing many configurations on the same test set adds selection bias. A new evaluation must group all recordings and windows by physical bearing and select configurations inside training folds.

The desktop A8 model takes **25 handcrafted descriptors: 10 time-domain statistics, 8 band-power ratios and 7 spectral/autocorrelation descriptors**, then applies z-score scaling and PCA(4). The archived device extractor instead computes **25 short-frame PSD band energies**. Equal vector length does not make those feature definitions interchangeable.

See the [results tables](06_results_evaluation/), [thesis](Master_Thesis_Audio_ESP32.pdf) and [technical review](docs/TECHNICAL_REVIEW.md). The tables and thesis retain the historical experiment record.

## Timing interpretation

The thesis reports approximately **64 ms per predictor call over 500 calls**. In the archived code, each `predict_gmm()` call reads and parses two JSON files, constructs parameter arrays, and performs normalization, PCA and GMM scoring. The timing therefore includes repeated model loading and is not an isolated arithmetic benchmark. Spectral feature extraction is excluded, and the code does not measure live microphone latency.

The reported feature time of approximately 11.2 seconds for a 512-sample frame used the Python DFT fallback. The thesis contains an FFT erratum. Choosing a supported FFT API, caching parameters and re-measuring stages on the board are outstanding work. Hardware timings were not repeated in this review.

## Dataset and evaluation

- 60 ten-second WAV recordings from 20 physical bearings.
- 104 logged result rows with 93 distinct run IDs, including repeated IDs and file-aggregation results.
- The available standard splitter assigns all windows from a source recording to one subset. Using sorted published filenames reconstructs 42/9/9 train/validation/test recordings.
- Recording grouping prevents windows of the same recording crossing subsets, but the reconstructed subsets share physical bearings across different recordings.

The split implementation is in [make_splits.py](02_data_segmentation/make_splits.py).

The WAVs derive from the University of Ottawa dataset, version 5, by **Mert Sehri and Patrick Dumond**, licensed **CC BY 4.0**. See [dataset attribution](01_uottawa_wavs/ATTRIBUTION.md).

## Repository map

| Directory | Contents |
|---|---|
| [01_uottawa_wavs](01_uottawa_wavs/) | Converted source WAVs and attribution |
| [02_data_segmentation](02_data_segmentation/) | Segmentation and recording-grouped split scripts |
| [03_feature_extraction](03_feature_extraction/) | Desktop feature pipelines and archived device extractor |
| [04_model_training_ablation](04_model_training_ablation/) | Training/ablation scripts and embedded GMM runtime |
| [05_model_export](05_model_export/) | Export scripts and two archived desktop model bundles |
| [06_results_evaluation](06_results_evaluation/) | Historical evaluation tables and summaries |
| [07_confusion_matrices_full](07_confusion_matrices_full/) | Archived confusion matrices |
| [docs](docs/) | Setup, validation scope, technical review and review evidence |

Additional documents: [thesis](Master_Thesis_Audio_ESP32.pdf), [defense presentation](Presentation%20Audio-Based%20Bearing%20Fault%20Detection.pptx), and [supplementary-materials index](Supplementary_Materials_Index_Ramin_Nassiri_MEM_OTH_Regensburg.pdf).

## Getting started

Clone and inspect the archive with Python's standard library:

```bash
git clone https://github.com/Ramin-RN7/esp32-bearing-fault-detection.git
cd esp32-bearing-fault-detection
python scripts/check_snapshot.py
```

For host model diagnostics, install `requirements.txt` in a virtual environment and run:

```bash
python scripts/review_models.py
```

This reconstructs the test split, exports the GMM parameters, re-scores the archived models and demonstrates the device feature mismatch in a host emulation. It writes diagnostics under `artifacts/technical_review/`. It performs no new training and no hardware verification. See the [review evidence](docs/review/summary.json).

**Read [running the archived scripts](docs/RUNNING.md) before attempting full experiments.** The archive uses stage-specific paths and omitted intermediates, so it is not a one-command reproduction package. Some legacy evaluation scripts also change preprocessing between training and scoring; details are in the technical review.

## Source and licensing

The supplied supplementary archive contains the historical research scripts, models, results and thesis documents. Its [original README](docs/original-supplementary-readme.md) is retained. Setup guidance, snapshot checks and the later technical review accompany the archive.

No general software license has been assigned to the authored code or thesis documents. The separately credited dataset retains its **CC BY 4.0** license.
