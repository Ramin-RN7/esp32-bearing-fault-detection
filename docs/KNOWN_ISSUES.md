# Known issues

The known problems in the thesis archive: where the device prototype falls short, why the thesis test split can't support claims about unseen bearings, defects in the archived scripts, and where the thesis text differs from the code. The thesis PDF and slides stay as submitted; the corrections are collected here. Last updated 18 October 2026.

| Issue | Status | Details |
|---|---|---|
| The device computes different features from the ones the model was trained on | **Open on the device.** A strict expected-failure test tracks it. The [decision contract](DECISION_CONTRACT.md) and golden outputs are frozen (18 October 2026); a new board has to meet them | [§1](#1-desktop-and-device-features-have-different-meanings) |
| The predictor timing includes reloading the model on every call; feature extraction used a Python DFT | **Open.** To be re-measured on the new board | [§2](#2-predictor-timing-includes-repeated-model-loading) |
| The thesis test split has 9 recordings, shares bearings with training, and the configurations were compared on it | **Addressed by follow-up evaluations** on unseen bearings, with nested selection. The thesis's A8 GMM and A16 MLP results reproduce exactly | [§3](#3-the-split-is-too-weak-for-unseen-bearing-claims) |
| The thesis text differs from the code and records in seven places; slide 19 presents the PC accuracy as the board's, with a timing that is only partly measured | **Documented** | [§5](#5-thesis-text-versus-code-and-records) |
| Wrong configuration metadata in two supplementary CSV files | **Fixed** by [`results/ablation_runs.csv`](../results/ablation_runs.csv), rebuilt from the original run log | [§4](#4-additional-archive-issues) |
| `records/device/exports/A16_MLP/` holds the A9 model | **Documented** in the folder and checked by [`tests/test_exports.py`](../tests/test_exports.py) | [§4](#4-additional-archive-issues) |
| The MLP C exporter overwrites its input buffer | **Open** | [§4](#4-additional-archive-issues) |
| The evaluation package's feature cache could return stale features for changed audio of the same size | **Fixed.** The cache key covers the audio content, the feature code and the library versions; [`tests/test_feature_cache.py`](../tests/test_feature_cache.py) covers it | – |
| Smaller script defects: a legacy CSV writer, an FFT input crop in two baseline scripts, the A45 placeholder, label polarity | **Open** | [§4](#4-additional-archive-issues) |

## 1. Desktop and device features have different meanings

The desktop [`a8_bandpass_1_5k.py`](../03_feature_extraction/a8_bandpass_1_5k.py) filters a two-second window and computes 25 features in a fixed order: 10 time-domain statistics (mean, standard deviation, RMS, extrema, skewness, kurtosis, first-difference statistics, zero-crossing rate), 8 relative band powers and 7 spectral and autocorrelation descriptors. The archived GMM bundle expects exactly these: its scaler has 25 inputs, its PCA four components and its GMM three diagonal-covariance components.

The device [`feat_a8.py`](../03_feature_extraction/feat_a8.py) takes a short frame, removes its mean, applies a Hann window and pools the power spectrum between 1 and 5 kHz into 25 bands. Index 0, for example, is the waveform mean in training and a band energy on the device. Frame length, preprocessing, units and definitions all differ, so matching the vector length doesn't make the features compatible.

**Host emulation.** [`scripts/review_models.py`](../scripts/review_models.py) computes the device features on a PC from the last 256-, 512- and 1024-sample frame of each of the 81 test windows (243 vectors, 81 of them from healthy recordings) and scores them with the archived scaler, PCA and GMM:

- All 243 vectors fall on the fault side of the threshold, healthy ones included.
- Their training-normalized deviations reach about 16.5 to 2163 standard deviations.
- Log-likelihoods range from about −7506.8 to −91.7, against a threshold of −11.2183.

This is an emulation on a PC, not a measurement on the board. The [device report](../records/device/REPORT_DevicePhase.md) of 14 August 2025 agrees: it records "GMM predictions not meaningful", "Parity with PC not established" and that accuracy was not validated.

**What does work.** Given the correct desktop features, the archived MicroPython runtime, run on a PC with NumPy in place of ulab, makes the same decisions as scikit-learn:

- `scripts/review_models.py` (run by hand) compares all 81 test windows with two feature implementations: the same decisions in all 162 comparisons, with log-likelihoods within about 0.0014.
- On every push, [`tests/test_device_runtime.py`](../tests/test_device_runtime.py) checks a smaller case: 54 windows from six test recordings with one feature implementation, the same decisions and log-likelihoods within 0.01. It also checks that the deployed JSON files hold the archived A8 model.

Neither tests ulab or the board.

**Fix.** One decision contract covering PCM scaling, filters and their boundary handling, FFT and framing settings, feature definitions and order, scaler, PCA and model parameters, the window-to-recording aggregation, the threshold and the label meaning. Golden outputs for every stage, from PCM samples to the recording decision, that the board must reproduce within declared tolerances before any accuracy is measured on it.

**Status (18 October 2026).** Defined for A5 MLP, the best single pipeline, in [DECISION_CONTRACT.md](DECISION_CONTRACT.md), with the frozen files in [`reference/a5/`](../reference/a5/). The reference reproduces the archived A5 features in every bit. A float32 emulation of a port meets every tolerance on all 540 windows with the same decisions. What remains is the board itself (open work, item 2).

## 2. Predictor timing includes repeated model loading

[`time_gmm.py`](../04_model_training_ablation/time_gmm.py) times repeated calls to `predict_gmm()` in [`model_rt_gmm.py`](../04_model_training_ablation/model_rt_gmm.py). Each call opens and parses both model JSON files before scaling, PCA and GMM scoring, so 500 calls mean 1000 file loads. The archived ~64 ms per call (63.4 and 64.0 ms in two measurements) therefore includes model loading and excludes feature extraction. How much of it is file access, parsing or arithmetic is unknown. `infer_only()` times a zero feature vector and `end_to_end()` an all-zero frame; both are microbenchmarks, not classification benchmarks.

Feature extraction took about 11.2 s per 512-sample frame, because a broad import exception fell back to a pure-Python DFT. The firmware does provide `ulab.numpy.fft` (the thesis erratum corrects this), but the archived fast path assumes `rfft()` and complex results, while ulab's FFT interface depends on build options ([ulab documentation](https://micropython-ulab.readthedocs.io/en/latest/numpy-fft.html)). Changing the import alone is not enough.

**Fix.** Load the model once, use the board's FFT without a silent fallback, and time model loading, features, scaling and PCA, and scoring separately, together with correct predictions.

## 3. The split is too weak for unseen-bearing claims

[`prep_segments.py`](../02_data_segmentation/prep_segments.py) cuts two-second windows with a one-second hop, and [`make_splits.py`](../02_data_segmentation/make_splits.py) assigns whole recordings within each class (seed 42): 42 training, 9 validation and 9 test recordings. The archived manifests in [`records/manifests/`](../records/manifests/) contain exactly this split.

The test set has 3 healthy and 6 faulty recordings (81 overlapping windows) from 7 physical bearings. The dataset paper identifies the first number in each file name as the bearing ID ([Sehri and Dumond, §2.1](https://doi.org/10.1016/j.dib.2023.109327)). Six of the nine test recordings share a bearing with training or validation, and 11 of the 20 bearings cross a split boundary:

| Test recording | Recordings of the same bearing outside the test set |
|---|---|
| H_10_0 | O_10_1, O_10_2 in training |
| H_13_0 | B_13_1 in training; B_13_2 in validation |
| H_18_0 | None |
| B_11_2 | H_11_0 in training; B_11_1 in validation |
| B_14_2 | H_14_0 in training; B_14_1 in validation |
| B_15_1 | H_15_0, B_15_2 in training |
| C_18_1 | None |
| C_18_2 | None |
| C_19_2 | H_19_0, C_19_1 in training |

### The thesis scores reproduce

Rows are true labels, columns predictions, ordered `[healthy, fault]`:

| Model | Recording confusion matrix | Accuracy | Fault F1 | Balanced accuracy | MCC |
|---|---|---:|---:|---:|---:|
| A8 GMM | `[[3, 0], [1, 5]]` | 0.889 | 0.909 | 0.917 | 0.791 |
| A16 MLP | `[[2, 1], [0, 6]]` | 0.889 | 0.923 | 0.833 | 0.756 |

At window level, A8 gives `[[26, 1], [9, 45]]` (F1 0.900) and A16 `[[18, 9], [0, 54]]` (F1 0.923). A16's nine false-alarm windows are seven from H_13_0 and one each from H_10_0 and H_18_0; after averaging, only H_13_0 is a false alarm. The GMM misses C_19_2. [`tests/test_reproduction.py`](../tests/test_reproduction.py) checks these results.

### Why they can't support claims about unseen bearings

- **Nine recordings are few.** Between random splits of this size, balanced accuracy varies with a standard deviation of about 0.15. For 8 of 9 correct, a simple binomial interval is about 57–98%.
- **Bearings are shared** between training and test, as the table shows.
- **The configurations were compared on the same test recordings**, so the best scores are optimistic. With two faulty recordings per healthy one, a classifier that always says "fault" already gets F1 0.800 (balanced accuracy 0.500).
- **Load may be a shortcut cue.** The dataset paper (§3.2) gives a nominal 400 N load for the healthy, inner-race, outer-race and cage recordings, and no load for the ball faults (bearings 11–15). Three of the six test faults are ball faults.

### Follow-up evaluations

[`results/protocol_comparison.md`](../results/protocol_comparison.md) tests the two thesis pipelines with leave-one-bearing-out, bootstrap intervals over bearings, ten choices of validation bearings, the evaluation without the ball-fault bearings and 200 repeated splits. [`results/ablation_lobo.md`](../results/ablation_lobo.md) does the same for all 86 ablation pipelines and adds nested selection. On unseen bearings, A16 MLP reaches 0.850 balanced accuracy, A8 GMM 0.762 and an RMS + kurtosis baseline 0.762. Choosing among all 86 pipelines without seeing the test bearing (nested selection) gives 0.950. The best single pipeline, A5 MLP, reaches 0.963, but it was picked after seeing all 86 results.

## 4. Additional archive issues

| Issue | Details | Status |
|---|---|---|
| The MLP C exporter overwrites its input | The predictor generated by [`a46_export_mlp_MIN.py`](../05_model_export/a46_export_mlp_MIN.py) swaps a scratch pointer onto the caller's `const float *x_in` and writes the next layer's output there; the second scratch buffer is unused. A host-compiled export of the A16 model matched scikit-learn on the first call for all 81 test windows but modified every input. Calling it again without restoring the inputs changed 78 of 81 decisions. | Open. Repair before using this export path |
| The MLP export folder is mislabelled | `records/device/exports/A16_MLP/` holds the A9 MLP (z-score scaler, PCA(4), 4→10→1), not A16 (MinMax, PCA(8), 8→10→1). Its header comment and `"feature_set": "A16"` are wrong too. This export never ran: the device report records that MLP inference was not callable from MicroPython. | Documented in the [folder](../records/device/exports/A16_MLP/README.md); checked by [`tests/test_exports.py`](../tests/test_exports.py) |
| Configuration metadata | `06_results_evaluation/ablation_summary.csv` mislabels envelope and PCA settings: A16 as without envelope and PCA, and A8, A17 and A31 as without PCA. `best_of_table.csv` lists the A42-RF and A43-ET window scores (0.790, 0.820) under `file_F1`. The original run log is correct, and so are the numbers in the thesis tables, but thesis Table 3.1 labels some window-level scores as file-level ([§5](#5-thesis-text-versus-code-and-records)). | Fixed: [`results/ablation_runs.csv`](../results/ablation_runs.csv) rebuilds the table from the run log and notes where a run's code differs from its settings (all MLP runs decide at probability 0.5; A42, A43 and E3 use 23 features) |
| Logged rows are not configurations | `variants.csv` has 104 rows for 93 configuration IDs, because of repeated runs and aggregation entries. | Documented |
| Some A16 scoring helpers change the front end | `a35b_aggregate_A16_MLP_per_file.py` clamps its requested 5000 Hz cutoff to 3600 Hz instead of the training's 3500 Hz. The A40/A41 and B1/B2 helpers for A16 skip the training band-pass, smoothing, percentile normalization and RMS matching, so their results don't isolate aggregation or window length. | Documented; see [§5](#5-thesis-text-versus-code-and-records) |
| A42 and A43 use 23 features | Their feature functions return 10 time statistics, 6 band powers and 7 spectral and autocorrelation values, on a raw Hilbert magnitude with `log1p`. | Documented |
| A legacy feature writer | `run_variant_ov75.py` calls `writerow` inside the loop over feature values, producing 11 rows per window. | Open |
| FFT input crop | `extract_features.py` and `run_variant_ov75.py` pass a 4096-point FFT length for 32,000-sample windows, so NumPy crops the input. | Open |
| A45 is a placeholder | `a45_train_export_rf_small.py` exports the A44 ExtraTrees model under random-forest names; it doesn't train a random forest. | Documented |
| Label polarity | Offline, 1 means fault; the archived device runtime returns 1 for healthy. | Open; part of the feature contract |
| Embedded files | The WAV evaluation, live and loop scripts, the deployed JSON files and the notes from the firmware rebuild are in [`records/device/`](../records/device/) and [`records/firmware/`](../records/firmware/). The `wavtools` and `feat_a8.extract` names in thesis Listing 4.1 never existed: the listing is illustrative, and the measured runs used `a8_eval_wav_gmm.py` and `time_gmm.py`. The device ran the lab's custom MicroPython firmware, which belongs to the lab and is not included. | Documented |

## 5. Thesis text versus code and records

The thesis PDF stays as submitted. None of these points changes the reproduced scores in [§3](#3-the-split-is-too-weak-for-unseen-bearing-claims); some limit how far the conclusions reach. Unless stated otherwise, scores are window-level balanced accuracy on the 81-window test split, from [`results/ablation_runs.csv`](../results/ablation_runs.csv).

| Thesis statement | What the code and records show | Effect |
|---|---|---|
| **25-feature set** (§2.2.3, Table 2.1). Table 2.1 lists the SES lab's 25-feature set "adapted for audio signals". §2.2.3 applies "the same 25-feature methodology" and keeps "the feature definition" so that vibration and audio results can be compared directly. | [`features25`](../03_feature_extraction/a8_bandpass_1_5k.py) computes 14 of the table's features and 11 others. Not computed: variance; crest, shape and impulse factors; spectral skewness, kurtosis and entropy; dominant frequency; a separate bandwidth; and autocorrelation energy and entropy. Instead the code computes peak-to-peak, two first-difference statistics, seven more band powers (eight in total) and spectral slope. | Adapting the features to audio is expected, but the thesis doesn't list the 25 features actually used, and the comparison with the vibration features is not like for like. |
| **Data preparation** (§2.2, Listing 2.1). The WAVs were converted at 42 kHz. The listing shuffles two-second segments and splits them 70/15/15. | All 60 WAVs are 16 kHz ([attribution](../01_uottawa_wavs/ATTRIBUTION.md)). [`prep_segments.py`](../02_data_segmentation/prep_segments.py) cuts two-second windows with a one-second hop. [`make_splits.py`](../02_data_segmentation/make_splits.py) splits whole recordings within each class, 42/9/9. The archived [manifests](../records/manifests/) confirm this. | The real split is stricter than the listing: no recording contributes windows to two subsets. The listing is labelled "representative"; the code is the reference. |
| **One change per run** (§1.8, §3.1, §3.4). Every run changed exactly one parameter. | Several runs change two or more settings at once. For example, relative to A11, A12 lowers the sample rate to 8 kHz and the band-pass to 1–3.5 kHz. Relative to A16, A19 changes both back. | Effects attributed to one setting can be confounded with the other change. |
| **Envelope and log compression** (abstract, contribution 2, §3.6). Narrow-band features with envelope and log compression "consistently" improved separability and outperformed uncompressed alternatives, for GMM and MLP. | For the GMM, adding the Hilbert envelope to A8's band-pass lowered balanced accuracy from 0.898 (A8) to 0.556 (A10). Adding log compression gave 0.500 (A11). None of the 26 GMM runs with an envelope exceeded 0.667, and the best GMM (A8) uses neither envelope nor log. For the MLP, envelope and log alone did not help (A8 0.750, A11 0.741). A16 reached 0.833 only with new sample-rate, band-pass, PCA and scaler settings as well. | The claim does not hold for the GMM. For the MLP it holds only together with the other changes. |
| **Aggregation** (abstract, §3.5.2, §3.6). Mean aggregation clearly beat median and majority voting: average F1 ≈ 0.923 against ≈ 0.727. | For A8, all three rules give the same result: F1 0.909, balanced accuracy 0.917. The A16 median and majority runs (A40-A16, A41-A16: F1 0.545) used a helper with a different front end. It resamples, takes the Hilbert envelope and `log1p`, and computes the 25 baseline features, without the training band-pass, smoothing, percentile normalization or RMS matching ([§4](#4-additional-archive-issues)). | The data don't show that the aggregation rule matters. The lower A16 scores come from the front-end change. |
| **On-device validation** (contribution 3, §4.2, §4.5). The WAV test exercised the model "identically to the offline pipeline", and the A8→GMM path "worked end-to-end … and file-level decisions". | The device computes different features from the ones the model was trained on ([§1](#1-desktop-and-device-features-have-different-meanings)). The [device report](../records/device/REPORT_DevicePhase.md) of 14 Aug 2025 records "GMM predictions not meaningful" and "accuracy not validated". The ~64 ms in §4.4 includes reloading the model JSON on every call ([§2](#2-predictor-timing-includes-repeated-model-loading)). | The device work is an inference prototype with measured timings, not a validated end-to-end detector. |
| **Top configurations** (§3.5.3, Table 3.1), headed "Top configurations by file-level F1". | Some rows are window-level scores: A16-MLP, A17-MLP, A5-MLP and A8-GMM. A8-GMM's 0.900 is its window-level F1; its file-level F1 is 0.909 (row A8-GMM-FILE). Each number matches the run log at its own level. | The numbers are right, but rows at different levels are ranked together. |
| **Defense slide 19.** F1 ≈ 0.90–0.91, precision ≈ 1.00 and recall ≈ 0.83, "same accuracy as offline", at about 70–75 ms per frame ("FFT fast, but MicroPython PCA+GMM still ~64 ms"). | The accuracy figures match the PC results for A8 with per-file aggregation. The ~64 ms model step is measured; no FFT timing is archived, so the remaining 6–11 ms can't be traced to a measurement. The measured feature step used the Python DFT and took about 11.2 s per frame. The device features don't match the model ([§1](#1-desktop-and-device-features-have-different-meanings)), so the board's accuracy is unknown. | The slide presents the PC accuracy as the board's, and its timing is only partly measured; neither was measured end to end on the board. |

The erratum at the front of the thesis, dated 14 Aug 2025, corrects only the FFT wording. It states that "all figures, tables, and conclusions remain valid", so it does not cover the points above.

## Reproduce the diagnostics

With the pinned environment from [RUNNING.md](RUNNING.md) installed:

```bash
python scripts/check_snapshot.py
python scripts/review_models.py
```

The first command checks that the archived files are readable. The second rebuilds the nine-recording test set, re-scores the two archived models, exports the GMM, compares the runtime arithmetic with scikit-learn and runs the device-feature emulation from §1. Outputs go to `artifacts/technical_review/` (`--out PATH` for another folder). Neither command tests firmware or an ESP32.

## Open work

1. Done for A5 MLP (18 October 2026): the decision contract from §1 and golden outputs from PCM to the recording decision ([DECISION_CONTRACT.md](DECISION_CONTRACT.md)). The old device's mismatch is still tracked by a strict expected-failure test.
2. On the new board, with a pinned MicroPython + ulab build: complete the WAV path (recordings copied in batches, since the WAVs total 19.2 MB), load the model once, use the board's FFT, check each stage against the golden outputs with `scripts/check_board.py` and time each stage alongside correct predictions.
3. Repair the MLP C exporter's buffer handling and check repeated calls before using that export path.
4. Consolidate the duplicated experiment code and repair the legacy writers.

Done: evaluation on unseen bearings with out-of-fold predictions for the two thesis pipelines (15–16 October 2026) and for all 86 ablation pipelines with nested selection (17 October 2026); the rebuilt configuration table; the code license.
