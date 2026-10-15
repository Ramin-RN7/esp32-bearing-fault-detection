# Running the archived scripts

This repository preserves the thesis supplementary materials. The numbered directories organize the study by stage; they do not automatically connect every script's input and output paths.

## Reproduce and re-evaluate (recommended)

The [`bearing_audio`](../src/bearing_audio/) package re-implements the two thesis pipelines with connected paths. It reads the WAVs in `01_uottawa_wavs/` and the archived split in `records/manifests/`.

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q                       # reproduces the thesis split exactly; about 3 minutes the first time
python scripts/evaluate_protocols.py      # leave-one-bearing-out and matched repeated splits; about 10 minutes
python scripts/build_ablation_table.py    # rebuilds results/ablation_runs.csv from the original run log
```

Window features are cached in `artifacts/features.npz`. Delete that file to force them to be recomputed.

## Inspect the snapshot

From the repository root:

```bash
python scripts/check_snapshot.py
```

This requires only Python's standard library. It checks syntax, WAV readability/durations and the archived best-results table. It does not run training or test an ESP32.

## Reproduce the later host review

Read the [technical review and known issues](TECHNICAL_REVIEW.md), then install the desktop requirements and run:

```bash
python scripts/review_models.py
```

This reconstructs the missing split, re-scores the two archived models, exports the GMM and demonstrates the desktop/device feature mismatch using a NumPy host emulation. It writes to `artifacts/technical_review/`. No retraining or ESP32 execution occurs. Other historical dependency versions are unknown, so matching the original environment fully is not established.

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
- The original manifests and baseline feature tables are in [`records/manifests/`](../records/manifests/). The scripts write to their own `manifests/` folders, so copy them there, or regenerate them, before running a stage. Segmented audio is not included; `prep_segments.py` regenerates it.

## Embedded files

- `03_feature_extraction/feat_a8.py`: archived MicroPython feature implementation.
- `04_model_training_ablation/model_rt_gmm.py`: ulab scaler/PCA/GMM runtime.
- `05_model_export/a47_export_gmm_json.py`: desktop parameter exporter.

The runtime expects `/models/A8_GMM/preprocess.json` and `/models/A8_GMM/gmm.json` on the device. The deployed copies are in [`records/device/exports/A8_GMM/`](../records/device/exports/A8_GMM/), together with the WAV evaluation, live and loop scripts and the device notes. The exporter can also regenerate the JSON files from the bundle:

```bash
python 05_model_export/a47_export_gmm_json.py --bundle 05_model_export/gmm_A8.joblib --out 05_model_export/exports/A8_GMM
```

The archived `feat_a8.py` is incompatible with the feature meanings used to train this GMM. Do not join that extractor to the bundle as a validated audio classifier. Define a matching feature contract and verify feature-to-score parity first. Device label 1 means normal, while offline label 1 means fault.

`predict_gmm()` currently loads both JSON files on every call. Cache parameters before measuring arithmetic separately. Verify the actual ulab FFT API and return format on the board; changing only the import path does not establish compatibility. The firmware build notes and configuration are in [`records/firmware/`](../records/firmware/). This archive still does not include a complete ESP-IDF firmware tree or a tested flashing procedure.

The optional `a46_export_mlp_MIN.py` path has a separate issue: its generated C predictor writes into the caller's input buffer despite accepting a `const` input pointer. Reusing that input without restoring it can change predictions. Repair the scratch-buffer swap and check input preservation and repeated calls before integrating this export. See the [second-pass review](TECHNICAL_REVIEW.md#second-pass-verification).

## Evaluation boundaries

The historical file-level scores use nine test recordings, which makes them noisy (about ±0.15 balanced accuracy between random splits of that size). The split shares physical bearings across subsets. Matched comparisons show this does not measurably change the scores, but only the leave-one-bearing-out results in [`results/`](../results/) estimate unseen-bearing performance. Some A16 aggregation and re-segmentation scripts also change preprocessing relative to training. Review their differences before interpreting them as controlled comparisons.

The approximately 64 ms thesis predictor measurement excludes feature extraction but includes repeated JSON loading/parsing. The snapshot checker does not reproduce predictions or hardware timings. The host review script reproduces selected offline predictions and exposes the feature mismatch; hardware timings remain unverified.
