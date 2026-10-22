# Running the archived scripts

This repository preserves the thesis supplementary materials. The numbered directories organize the study by stage; they do not automatically connect every script's input and output paths.

## Reproduce and re-evaluate (recommended)

The [`bearing_audio`](../src/bearing_audio/) package re-implements the two thesis pipelines with connected paths. It reads the WAVs in `01_uottawa_wavs/` and the archived split in `records/manifests/`.

Use Python 3.13, the version CI tests with `requirements-lock.txt`. Other versions can work: an independent re-run of `evaluate_protocols.py` on Python 3.12 with older NumPy, SciPy and librosa gave the same original-split and leave-one-bearing-out decisions for A8 GMM, A16 MLP and the baseline, with slightly different continuous scores and one of the 1,200 repeated-split rows changed.

```bash
python -m pip install -r requirements-lock.txt
python scripts/demo.py                    # A5 MLP on unseen bearings; about 2 minutes the first time
python -m pytest -q                       # includes the exact A8/A16 reproduction on the thesis split; about 4 minutes the first time
python scripts/freeze_reference.py        # the frozen A5 reference for a board port (reference/a5/); about 1 minute
python scripts/evaluate_protocols.py      # follow-up evaluations (see below); about 15 minutes
python scripts/build_ablation_table.py    # rebuilds results/ablation_runs.csv from the original run log
python scripts/evaluate_ablation.py       # every ablation pipeline on unseen bearings; about 75 minutes on 2 cores
```

`evaluate_protocols.py` writes `results/protocol_comparison.{json,md}` and `results/repeated_splits.csv`: leave-one-bearing-out with every fold, threshold and held-out score; bearing-level bootstrap intervals; a check with ten other choices of validation bearings; the same evaluation without the ball-fault bearings; and 200 matched repeated splits per protocol, each recorded. The JSON also records the commit, an audio hash, the feature-code hashes and the library versions. These full runs are made by hand. CI runs the tests, which include a one-fold check against the published leave-one-bearing-out results.

`evaluate_ablation.py` rebuilds every ablation pipeline from its run record ([`ablation.py`](../src/bearing_audio/ablation.py)). Each run's own `code.py` computes its features on the archived two-second segments, written exactly as `prep_segments.py` wrote them; only the script's imports, constants and functions execute. The model stage follows the run's settings. The script first checks every pipeline against its archived counts on the original split, then runs leave-one-bearing-out on all of them, then nested selection: each outer fold chooses a pipeline using only the other 19 bearings. Finally it checks the leading pipelines without the ball-fault bearings, with other validation bearings and with other MLP seeds. It writes `results/ablation_lobo.{json,csv,md}` and caches features in `artifacts/ablation_features/`.

Window features are cached in `artifacts/features.npz`. The cache is reused only if the audio content, the feature code, the feature-spec version and the feature libraries are all unchanged; otherwise the features are recomputed.

## The frozen A5 reference and a board port

[DECISION_CONTRACT.md](DECISION_CONTRACT.md) defines what a port of A5 MLP must compute and how it is checked.

```bash
python scripts/freeze_reference.py                    # rebuilds reference/a5/; publishes nothing if any check fails
python scripts/trace_recording.py H_13_0 --float32    # every window of one recording, with the float32 emulation
python scripts/trace_recording.py H_13_0 --window 3 --save w3.npz   # all stages of one window as arrays
python scripts/check_board.py board/*.json            # a board's output files against the reference (any subset)
python scripts/check_board.py --complete board/*.json # all 60, each with its fold model: the held-out result
```

`freeze_reference.py` writes the same files on the same environment; `tests/test_reference.py` checks them in CI (pushes to `main` and pull requests). The format of the board's output files is in the contract, section 6.

## Inspect the snapshot

From the repository root:

```bash
python scripts/check_snapshot.py
```

This requires only Python's standard library. It checks syntax, WAV readability/durations and the archived best-results table. It does not run training or test an ESP32.

## Host diagnostics

See [known issues](KNOWN_ISSUES.md). With the pinned environment from above installed, run:

```bash
python scripts/review_models.py
```

This rebuilds the test split from the file names, re-scores the two archived models, exports the GMM and demonstrates the desktop/device feature mismatch using a NumPy host emulation. It writes to `artifacts/technical_review/`. No retraining or ESP32 execution occurs. Other historical dependency versions are unknown, so matching the original environment fully is not established.

## Desktop environment

The package, the evaluation scripts and `review_models.py` need only the pinned environment above (`requirements-lock.txt`, Python 3.13). The archived experiment scripts in the numbered folders also import pandas, matplotlib and tqdm; for those, use `requirements.txt`.

Create a virtual environment, for example with Python 3.13:

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

Install the dependencies for the archived experiment scripts:

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

The archived `feat_a8.py` is incompatible with the feature meanings used to train this GMM. Do not join that extractor to the bundle as a validated audio classifier. Device label 1 means normal, while offline label 1 means fault. A new port should follow the [decision contract](DECISION_CONTRACT.md) for A5 MLP instead.

`predict_gmm()` currently loads both JSON files on every call. Cache parameters before measuring arithmetic separately. Verify the actual ulab FFT API and return format on the board; changing only the import path does not establish compatibility. The device ran the lab's custom MicroPython firmware, which is not included. The notes from rebuilding it with ESP-IDF 4.4 are in [`records/firmware/`](../records/firmware/); the rebuilt image booted, but the model-inference modules it was meant to add were not completed.

The optional `a46_export_mlp_MIN.py` path has a separate issue: its generated C predictor writes into the caller's input buffer despite accepting a `const` input pointer. Reusing that input without restoring it can change predictions. Repair the scratch-buffer swap and check input preservation and repeated calls before integrating this export. See [known issues](KNOWN_ISSUES.md#4-additional-archive-issues).

## Evaluation boundaries

The historical file-level scores use nine test recordings, which makes them noisy (about ±0.15 balanced accuracy between random splits of that size). The split shares physical bearings across subsets. Random splits of the same size that keep bearings apart scored within 0.02 on average. No large average difference was seen in these splits, but they reuse the same 20 bearings, so this does not prove the overlap is harmless. Only the leave-one-bearing-out results in [`results/`](../results/) estimate unseen-bearing performance. Some A16 aggregation and re-segmentation scripts also change preprocessing relative to training. Review their differences before interpreting them as controlled comparisons.

The approximately 64 ms thesis predictor measurement excludes feature extraction but includes repeated JSON loading/parsing. The device-runtime tests run the archived MicroPython code on the host with NumPy standing in for ulab; on-board parity has not been tested. The snapshot checker does not reproduce predictions or hardware timings. `review_models.py` reproduces selected offline predictions and exposes the feature mismatch; hardware timings remain unverified.
