> **Publication in progress:** The research code and setup notes are available. Remaining audio, figures and thesis documents are still uploading; some links below will become available when publication finishes.

# ESP32 Acoustic Bearing Fault Detection

Python DSP and machine-learning experiments, with a MicroPython/ulab inference prototype on an ESP32-based M5Stack Core2.

Supplementary materials for **Ramin Nassiri's master's thesis at OTH Regensburg (2025)**: *Audio-Based Bearing Fault Detection on Resource-Constrained Microcontrollers: An Ablation-Driven Study*.

## Overview

The study investigates acoustic bearing-fault detection for condition monitoring. An ablation framework compares preprocessing, spectral features, scaling, PCA, GMM/MLP classifiers and file-level aggregation. A selected PCA + GMM configuration was implemented on ESP32 using custom-built MicroPython firmware with ulab.

The embedded prototype processes **WAV files stored in flash**. Its reported timing is not a microphone-to-prediction latency measurement.

## Reported thesis results

| Evaluation | File-level F1 | Accuracy |
|---|---:|---:|
| Best offline MLP configuration | **0.923** | 0.889 |
| Strongest offline GMM configuration | **0.909** | 0.889 |

GMM was selected for its simpler on-device implementation. The MLP score is an **offline result**, not a deployed MLP result.

The deployed configuration reduces **25 spectral features to 4 PCA components**. Approximately **64 ms** was measured on ESP32 for **z-score normalization + PCA + GMM inference**, across **500 calls**. **Spectral feature extraction is excluded** and was the end-to-end bottleneck.

See the [results tables](06_results_evaluation/) and [thesis](Master_Thesis_Audio_ESP32.pdf) for configurations and measurement details. These are the original thesis results, not newly rerun benchmarks.

## Dataset and evaluation

- 60 ten-second recordings from 20 bearings; 104 logged experimental runs.
- Windows grouped by source WAV into nominal 70/15/15 train/validation/test subsets, with deterministic assignment within each class.
- Grouping keeps windows from a recording in one subset. **Strict bearing-level holdout is not established** by this grouping.

The split implementation is in [make_splits.py](02_data_segmentation/make_splits.py).

The WAVs are derived from the University of Ottawa bearing dataset, version 5, by **Mert Sehri and Patrick Dumond**, licensed **CC BY 4.0**. See [dataset attribution](01_uottawa_wavs/ATTRIBUTION.md).

## Repository map

| Directory | Contents |
|---|---|
| [01_uottawa_wavs](01_uottawa_wavs/) | Converted source WAVs and attribution |
| [02_data_segmentation](02_data_segmentation/) | Segmentation and recording-grouped split scripts |
| [03_feature_extraction](03_feature_extraction/) | DSP and feature-extraction pipelines |
| [04_model_training_ablation](04_model_training_ablation/) | Training/ablation scripts and embedded GMM runtime |
| [05_model_export](05_model_export/) | Export scripts and two archived desktop model bundles |
| [06_results_evaluation](06_results_evaluation/) | Evaluation tables and experiment summaries |
| [07_confusion_matrices_full](07_confusion_matrices_full/) | Archived confusion matrices |

Additional documents: [thesis](Master_Thesis_Audio_ESP32.pdf), [defense presentation](Presentation%20Audio-Based%20Bearing%20Fault%20Detection.pptx), and [supplementary-materials index](Supplementary_Materials_Index_Ramin_Nassiri_MEM_OTH_Regensburg.pdf).

## Getting started

Clone the repository and check the snapshot using Python's standard library:

```bash
git clone https://github.com/Ramin-RN7/esp32-bearing-fault-detection.git
cd esp32-bearing-fault-detection
python scripts/check_snapshot.py
```

For desktop experiments, create a virtual environment and install `requirements.txt`. **Read [running the archived scripts](docs/RUNNING.md) first**: the archive uses paths and intermediate files from the research workspace and is not a one-command reproduction package.

All 66 archived Python scripts passed a syntax check, and all 60 WAVs were checked for readability and duration. See [validation scope](docs/VALIDATION.md). The firmware, hardware timings and full training/evaluation pipeline have not been rerun for this publication.

The thesis erratum notes that ulab FFT was available in the firmware but was not used in the measured spectral-extraction runs, which used a Python DFT fallback.

## Source and licensing

The research scripts, models, results and thesis documents are preserved from the supplied supplementary archive. Its [original README](docs/original-supplementary-readme.md) is retained. Setup guidance and the snapshot checker were added for this GitHub publication.

No general software license has been assigned to the authored code or thesis documents. The separately credited dataset retains its **CC BY 4.0** license.
