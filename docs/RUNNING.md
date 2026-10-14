# Running the archived scripts

This repository preserves the thesis supplementary materials. The numbered directories organize the study by stage; they do not automatically connect every script's input and output paths.

## Inspect the snapshot

From the repository root:

```bash
python scripts/check_snapshot.py
```

This requires only Python's standard library. It checks syntax, WAV readability/durations and the archived best-results table. It does not run training or test an ESP32.

## Desktop environment

Create a virtual environment, for example with Python 3.11 or 3.12:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install the desktop dependencies:

```bash
python -m pip install -r requirements.txt
```

`scikit-learn==1.7.1` matches the version recorded in both archived model bundles. The remaining package list was derived from source imports; it is not a complete lockfile of the research environment. For the emlearn export experiments, use `requirements-export.txt`.

## Input paths and intermediates

- `01_uottawa_wavs/wav/normal/` and `01_uottawa_wavs/wav/fault/` contain source WAVs.
- `02_data_segmentation/prep_segments.py` expects inputs in `data/raw/normal/` and `data/raw/anomaly/`, relative to that script's directory. The archive's fault recordings correspond to the anomaly input category.
- That script generates standardized WAVs, overlapping segments and `manifests/segments.csv`.
- `make_splits.py` assigns all segments of an `orig_file` to one subset. Outputs include `segments_split.csv` and separate train/validation/test manifests.
- Feature and training scripts use their own paths, manifests and artifacts directories. Inspect each script's `ROOT` and path constants, and connect its inputs to preceding stage outputs.
- The full original collection of generated manifests and feature arrays is not included. These must be regenerated to rerun the complete pipeline.

## Embedded files

- `03_feature_extraction/feat_a8.py`: archived MicroPython feature implementation.
- `04_model_training_ablation/model_rt_gmm.py`: ulab scaler/PCA/GMM runtime.
- `05_model_export/a47_export_gmm_json.py`: desktop parameter exporter.

The runtime expects `/models/A8_GMM/preprocess.json` and `/models/A8_GMM/gmm.json` on the device. These JSON files are absent from the original archive. The exporter accepts an explicit model path:

```bash
python 05_model_export/a47_export_gmm_json.py --bundle 05_model_export/gmm_A8.joblib --out 05_model_export/exports/A8_GMM
```

Exported parameters and runtime scripts still need to be checked on compatible firmware before device use. This archive does not include a complete ESP-IDF firmware source tree or a tested one-command flashing procedure.

## Evaluation boundaries

Scores are file-level offline results. Recording-level grouping does not establish bearing-level holdout. The approximately 64 ms thesis measurement excludes spectral extraction. These scores and hardware timings are not reproduced by the snapshot checker.
