# Technical review and known issues

There were three review passes, all on 15 October 2026:

- **First pass:** reviewed [`c03372e`](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/commit/c03372e3610b765f9a6acc555b584b81d72441f2).
- **Second pass:** reviewed [`8c9a5a9`](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/commit/8c9a5a9dc69d13e3eab17dea69bdcb3a268c8280).
- **Third pass:** published the original research records in [`records/`](../records/) and added the evaluation package [`src/bearing_audio/`](../src/bearing_audio/).

## Current status (third pass)

The original split manifests, all 101 run folders and the device-phase files are now available. They settle the questions the first two passes left open.

| Finding | Status |
|---|---|
| Thesis scores | **Reproduced exactly.** Retraining from the archived manifests gives the same confusion matrices and misclassified recordings ([`tests/test_reproduction.py`](../tests/test_reproduction.py)). The archived A8 model also reproduces the per-window scores logged on 14 Aug 2025 to within 1e-3. |
| Split ([§3](#3-the-split-is-too-weak-for-unseen-bearing-claims)) | The archived manifests are **identical** to the reconstruction the earlier passes used. |
| Bearing overlap ([§3](#3-the-split-is-too-weak-for-unseen-bearing-claims)) | The overlap is real but **has no measurable effect**. Over 200 matched splits of the thesis size, keeping bearings apart changes mean balanced accuracy by at most 0.02 (GMM 0.000, MLP 0.010, baseline 0.018). The real uncertainty comes from two other things: the 9-recording test set, where balanced accuracy varies by about ±0.15 between random splits, and the choice of configurations by their test scores. |
| Performance on unseen bearings | Leave-one-bearing-out balanced accuracy: A16 MLP 0.850, A8 GMM 0.762. A two-feature baseline (RMS and kurtosis) also reaches 0.762. See [`results/protocol_comparison.md`](../results/protocol_comparison.md). |
| Load confound ([§3](#required-re-evaluation)) | Without the ball-fault bearings 11–15, which were recorded without load, balanced accuracy is MLP 0.800, GMM 0.733 and baseline 0.817. |
| Desktop/device features ([§1](#1-desktop-and-device-features-have-different-meanings)) | **Confirmed by the device notes** of 14 Aug 2025. [`REPORT_DevicePhase.md`](../records/device/REPORT_DevicePhase.md) says "GMM predictions not meaningful" and "Parity with PC not established". The deployed JSON is the archived A8 model. A strict expected-failure test tracks the issue. **Open.** |
| Timing ([§2](#2-predictor-timing-includes-repeated-model-loading)) | Confirmed: both published runtime versions reload the model JSON on every call. **Open.** |
| Configuration metadata ([§4](#4-additional-archive-issues)) | The errors are confined to two supplementary files: `06_results_evaluation/ablation_summary.csv`, plus the evaluation-level mix-up in `best_of_table.csv`. The original run log and the thesis tables are correct. The corrected table is [`results/ablation_runs.csv`](../results/ablation_runs.csv). |
| Per-run artifacts ([§4](#4-additional-archive-issues)) | **Resolved.** All run folders are in [`records/runs/`](../records/runs/). |
| Embedded assets ([§4](#4-additional-archive-issues)) | **Partly resolved.** The WAV evaluation, live and loop scripts, the deployed JSON files and the firmware build notes and configuration are now in [`records/device/`](../records/device/) and [`records/firmware/`](../records/firmware/). `wavtools` and `feat_a8.extract` never existed: thesis Listing 4.1 is illustrative, and the measured runs used `a8_eval_wav_gmm.py` and `time_gmm.py`. A complete firmware source tree is still missing. |
| License ([§4](#4-additional-archive-issues)) | **Resolved.** The code is MIT-licensed; the thesis documents remain all rights reserved. |
| Still open ([§4](#4-additional-archive-issues)) | MLP C exporter buffer swap, legacy CSV writer, FFT input crop, A45 placeholder, label polarity. |

The sections below are the record of the first and second passes. Sentences that the third pass superseded are marked.

## First-pass summary

The historical offline scores can be reproduced from the supplied model bundles under a reconstruction of the split, which is now confirmed by the archived manifests. The archived embedded feature extractor is incompatible with those models. The archive supports an offline ablation study and an embedded scoring prototype; a validated audio-to-decision deployment remains outstanding work.

## Scope and evidence

All **236 files** in the reviewed publication were checked against their published Git blob hashes and inspected according to format. The later review files are additions to that baseline.

| Files | Checks performed |
|---|---|
| 67 Python files | Parsed all sources with AST, including 66 archived scripts and the snapshot checker; traced feature, split, training, export, scoring and timing paths |
| 60 WAV files | Decoded full PCM payloads; all mono 16-bit, 16 kHz, 10 seconds; checked nonempty samples |
| 92 PNG files | Decoded images and read the printed confusion-matrix counts; 91 match full CSV metric rows, and the A44 file matrix matches the rounded `fileF1=0.769` note |
| 3 CSV files | Parsed every row; checked metric ranges and precision/recall/F1 arithmetic; inspected IDs and metadata |
| 2 joblib bundles | Loaded using scikit-learn 1.7.1, checked finite model parameters and re-scored the reconstructed test data |
| 2 PDFs | Extracted and rendered all 63 pages; cross-checked the technical claims in the 57-page thesis and 6-page index |
| 1 PowerPoint and 1 Word file | Checked ZIP integrity and every XML part; read all 20 presentation slides and the supplementary index |
| 8 other text files | Read the five Markdown files, two dependency files and `.gitignore`; checked internal Markdown links |

See the [per-file inventory](review/file_inventory.json) and [machine-readable summary](review/summary.json). A format/integrity pass means the file is readable and matches the published bytes. It does not certify experiment validity or device behavior.

The host review used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.7.1, librosa 0.11.0 and soundfile 0.13.1. Only scikit-learn's original version was recorded in the archive. Full historical environment equivalence is not established.

## Second-pass verification

A second pass on 15 October 2026 checked all **240 files** at [`8c9a5a9`](https://github.com/Ramin-RN7/esp32-bearing-fault-detection/commit/8c9a5a9dc69d13e3eab17dea69bdcb3a268c8280), including the first review and its diagnostic script. Every file matched the published Git blob; format checks and internal Markdown links passed. All 92 confusion-matrix images were checked again, with the same 91 full-metric matches and one rounded A44 file-F1 match. The selected host diagnostic results repeated exactly. See the [second-pass evidence](review/recheck.json).

The original segmentation and split scripts were also run **without source changes** in an isolated directory. They produced 540 windows from all 60 recordings and the same 42/9/9 recording split (378/81/81 windows). Refitting only the A8/A16 scalers and PCA on the reconstructed training windows gave numerically close preprocessing parameters, with small differences from the saved historical objects. The A8 validation-threshold search, using the archived model, recovered **-11.218337179787783 exactly**. These checks strengthen the reconstruction; they do not recover the missing original manifests or establish an identical historical environment. No classifier was retrained.

This pass found additional issues that the first review did not identify:

- **Evaluation units are mixed in `best_of_table.csv`.** A42-RF's 0.789916 and A43-ET's 0.819672 appear under `file_F1`, but their source scripts and confusion matrices evaluate 81 segments. Their matrices are `[[9,18],[7,47]]` and `[[9,18],[4,50]]`. They must not be compared as file-level scores with the nine-recording MLP/GMM results.
- **A42/A43 use 23 features, not the commented 25.** Their actual functions return 10 time statistics + 6 band powers + 7 spectral/autocorrelation values. They also use raw Hilbert magnitude plus `log1p`, omitting A16's band-pass, smoothing, percentile normalization and RMS matching. Their train/test dimensions are internally consistent, but these runs do not isolate a classifier change on the exact A16 representation.
- **The summary metadata errors are widespread.** Besides A16, `ablation_summary.csv` labels A8, A17 and A31 as having no PCA despite their source pipelines using PCA(4), PCA(8) and PCA(8). Other envelope/PCA entries also require reconstruction from the source; the table is not reliable configuration metadata.
- **Presentation slide 19 is inconsistent with the archived evidence.** Its approximately 70–75 ms frame latency and “same accuracy as offline” claims are unsupported by the incompatible feature definitions and conflict with the thesis's approximately 11.2-second feature time. The presentation remains a historical document; those claims must not be reused as verified results.
- **The minimal MLP C exporter overwrites its input.** The generated buffer swap in [`a46_export_mlp_MIN.py`](../05_model_export/a46_export_mlp_MIN.py) assigns a scratch pointer to the caller's `const float *x_in`, then writes the next layer's result there. The second scratch buffer is unused. A host-compiled export of the archived A16 model matched scikit-learn on the first call for all 81 reconstructed test windows, but modified all 81 input vectors. Calling it again without restoring the input changed **78/81 decisions**. This requires a buffer-management repair before reuse, particularly with shared or read-only input memory. It is a separate C-export defect; it does not explain the GMM feature mismatch. The test used writable host arrays and did not execute an ESP32.

## 1. Desktop and device features have different meanings

The desktop [`a8_bandpass_1_5k.py`](../03_feature_extraction/a8_bandpass_1_5k.py) filters a two-second segment and computes:

- 10 time-domain statistics, including mean, standard deviation, RMS, extrema, skewness, kurtosis, derivative statistics and zero-crossing rate.
- 8 relative band powers.
- 7 spectral/autocorrelation descriptors.

The archived GMM bundle expects that feature ordering. Its scaler has 25 input dimensions, its PCA has four components, and its GMM has three diagonal-covariance components.

The device [`feat_a8.py`](../03_feature_extraction/feat_a8.py) takes a short frame, removes its mean, applies a Hann window and pools PSD values from 1-5 kHz into 25 bands. For example, index 0 is a waveform mean in training and a PSD band energy on the device. Frame length, preprocessing, feature units and feature definitions differ. Matching the vector length cannot resolve this mismatch.

### Numerical check

The review reconstructed 81 two-second test windows. It took the final 256-, 512- and 1024-sample frame from each window, giving **243 sampled feature vectors**, including 81 derived from healthy recordings. Using normalized PCM and the archived scaler/PCA/GMM:

- All **243/243** sampled vectors fell on the fault side of the threshold.
- Maximum absolute training-normalized deviations ranged from about **16.5 to 2163 standard deviations**.
- Log-likelihoods ranged from about **-7506.8 to -91.7**, against threshold **-11.2183**.

This is a **host emulation**, not a hardware accuracy measurement. The FFT-based host formula was checked against the archived Python DFT fallback on a deterministic frame, agreeing to approximately `4.1e-15` in maximum absolute feature error. The omitted WAV loader and full firmware still prevent verification of the historical board execution.

### What does work

The parameter exporter and downstream scoring arithmetic are substantially consistent when supplied with the **correct desktop feature vectors**. The archived runtime, using a NumPy host shim, made the same fault/normal decisions as scikit-learn in all 162 comparisons covering 81 distinct windows and two A8 feature implementations. Maximum absolute log-likelihood difference was approximately **0.001414**. Epsilon terms, dtypes and operation order prevent claiming exact numerical identity. This check does not establish ulab compatibility or validate the front end.

### Required repair

Define one feature contract covering sample units, filtering, windowing, feature order, dimensions and label meanings. Then either implement the desktop descriptors faithfully on the device or retrain a model on the actual device PSD features. Compare features, normalized values, PCA projections, scores and decisions on the same held-out WAV inputs before benchmarking the board.

## 2. Predictor timing includes repeated model loading

[`time_gmm.py`](../04_model_training_ablation/time_gmm.py) times repeated calls to `predict_gmm()`. In [`model_rt_gmm.py`](../04_model_training_ablation/model_rt_gmm.py), each call opens and parses both model JSON files and constructs parameter arrays before scoring. Thus 500 calls imply **1000 JSON file loads**, in addition to scaling, PCA and GMM arithmetic.

The approximately 64 ms thesis value should be described as the archived **predictor-call timing including model loading**, with feature extraction excluded. The source establishes what the timed path includes. It does not establish how many milliseconds belonged to I/O, parsing or arithmetic, and the review did not repeat hardware measurements.

`infer_only()` uses a zero feature vector. `end_to_end()` separately times features from an all-zero synthetic frame and repeated predictor calls. These are implementation microbenchmarks, not live audio acquisition or verified classification benchmarks.

The slow feature path uses a Python DFT after a broad import exception. The thesis erratum correctly points to `ulab.numpy.fft`. A namespace change alone is insufficient: the archived fast path assumes `rfft()` and complex-number results, while documented ulab FFT APIs and return formats depend on firmware build options. See the [ulab FFT documentation](https://micropython-ulab.readthedocs.io/en/latest/numpy-fft.html). Verify the actual board API, cache parameters, and measure model loading, features, preprocessing and scoring separately. Correct predictions must accompany timing results.

## 3. The split is too weak for unseen-bearing claims

The available [`prep_segments.py`](../02_data_segmentation/prep_segments.py) generates two-second windows with a one-second hop. [`make_splits.py`](../02_data_segmentation/make_splits.py) assigns recordings within each class using seed 42. With the sorted published filenames, this reconstructs 42 training, 9 validation and 9 test recordings.

*Superseded by the third pass:* the original manifests, now published in [`records/manifests/`](../records/manifests/), contain exactly this split.

The test set has 3 normal and 6 fault recordings, comprising **81 overlapping windows from 7 physical bearings**:

| Test recording | Same-bearing recordings outside test |
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

The [dataset authors' paper, section 2.1](https://doi.org/10.1016/j.dib.2023.109327) identifies the first numeric filename field as the physical bearing ID. An [author-uploaded copy](https://www.researchgate.net/publication/371782379_University_of_Ottawa_constant_load_and_speed_rolling-element_bearing_vibration_and_acoustic_fault_signature_datasets) supplies the same mapping. Consequently, **6/9 test recordings share a physical bearing with training or validation**, and 11 of the 20 bearing IDs cross some split boundary. This is actual group overlap. *Third pass:* matched splits that keep bearings apart score the same, so the overlap does not inflate these results. See the [current status](#current-status-third-pass).

### Reproduced historical scores

Rows are true labels and columns are predictions, ordered `[normal, fault]`.

| Model | File confusion matrix | Accuracy | Fault F1 | Balanced accuracy | MCC |
|---|---|---:|---:|---:|---:|
| A8 GMM | `[[3, 0], [1, 5]]` | 0.888889 | 0.909091 | 0.916667 | 0.790569 |
| A16 MLP | `[[2, 1], [0, 6]]` | 0.888889 | 0.923077 | 0.833333 | 0.755929 |

The matching A8 training features reproduce segment confusion matrix `[[26, 1], [9, 45]]`, with F1 0.900000. Matching A16 training features reproduce `[[18, 9], [0, 54]]`, with F1 0.923077. The A16 false-positive windows are distributed as seven from H_13_0, one from H_10_0 and one from H_18_0. After mean-probability aggregation, only H_13_0 is a file-level false positive. GMM's file-level false negative is C_19_2.

Correction, 15 October 2026: an earlier version of this review incorrectly assigned all nine A16 false-positive windows to H_13_0. Rechecking the per-window output confirmed the distribution above. The confusion matrices and headline file-level scores are unchanged. The distinct A16 file-scoring helper has seven false-positive windows (five H_13_0, one H_10_0 and one H_18_0), plus one false-negative window; it uses the different preprocessing described below.

The MLP/GMM F1 ordering is not convincing evidence that one model generalizes better: each makes one file error, in a different direction. Overlapping windows are correlated observations, not 81 independent test cases. They can still contain within-recording variation, as the A8 segment matrix shows.

An always-fault classifier achieves F1 0.800000 on this 2:1 class mix. Fifteen distinct ordinary GMM run IDs match that baseline, with additional repeated log rows and B1 results. The table contains test scores from many configurations, so using its maximum to select and advertise a winner creates selection bias. The individual A8 script correctly calibrates its threshold on validation data; that does not remove the later cross-configuration selection issue.

For scale, an independent-binomial Wilson interval for 8/9 is approximately **56.5%-98.0%**. Bearing dependence and model selection mean this is only an illustration of small-sample uncertainty, not a corrected generalization interval.

### Required re-evaluation

*Third pass:* done for the two thesis pipelines. [`scripts/evaluate_protocols.py`](../scripts/evaluate_protocols.py) runs leave-one-bearing-out (out-of-fold predictions included), matched repeated splits and the ball-fault sensitivity analysis. Nested selection across all ablation configurations is still open.

Group all health states, recordings and derived windows by physical bearing. Use outer held-out bearing groups for evaluation and inner groups for preprocessing/model/threshold selection. Fit scalers and PCA only on each training fold. Report out-of-fold recording-level predictions, balanced accuracy, MCC, fault F1 and uncertainty that respects bearing groups. The [scikit-learn nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) explains the selection-bias issue.

The source dataset also has limited operating-condition coverage. Section 3.2 of its paper specifies a nominal 400 N load for healthy, inner-race, outer-race and cage data, and no applied load for ball-fault data. Three of the six reconstructed test-fault recordings are ball faults. This is a potential load/label confound; these results do not establish how much the models use load-related cues. Re-evaluation should include a sensitivity analysis that removes all recordings of ball-fault bearings 11–15. Grouping or excluding these bearings cannot demonstrate robustness across loads; that requires suitable additional data. *Third pass:* the sensitivity analysis has been run. See the [current status](#current-status-third-pass).

## 4. Additional archive issues

| Issue | Evidence and implication |
|---|---|
| A16 scoring preprocessing differs from training | Training uses an upper band-pass cutoff of 3500 Hz. `a35b_aggregate_A16_MLP_per_file.py` clamps its requested 5000 Hz to 3600 Hz. In the reconstruction this changes segment predictions, although mean file decisions remain the same. |
| Some aggregation/re-segmentation comparisons change the front end | A40/A41 and B1/B2 A16 helpers use resampling, Hilbert magnitude and `log1p` without the full training band-pass/smoothing/percentile normalization/RMS matching. Their results do not isolate aggregation or window length alone. |
| Configuration metadata is wrong | `ablation_summary.csv` marks A16-MLP as no envelope and no PCA, contradicting the code, bundle dimensions and `variants.csv`. The second pass confirmed that PCA errors also affect A8, A17 and A31. Use source and model objects as evidence; do not infer ablation effects from this metadata without repair. *Third pass:* the original log ([`records/logs/`](../records/logs/)) and the thesis tables are correct. [`results/ablation_runs.csv`](../results/ablation_runs.csv) rebuilds the table from them, with evaluation level, balanced accuracy and MCC. |
| Logged rows are not unique configurations | `variants.csv` has 104 rows and 93 distinct IDs. Repeated IDs include A1, A14, B1/B2-A8 and A44-ET. Counts include repeat runs and aggregation entries. |
| Per-run artifacts are not all present | Absolute Windows paths in `variants.csv` refer to historical `params.json`, `metrics.json`, `code.py` and model outputs that are mostly absent. The logging code supports their creation; the published archive does not contain every run directory. *Third pass: resolved.* All run folders are in [`records/runs/`](../records/runs/). |
| A legacy feature writer emits incomplete rows | `run_variant_ov75.py` calls `writerow` inside the loop over 11 feature values, producing 11 rows per window, of which the first 10 have missing feature fields. That script needs repair before reuse. |
| Two baseline FFT calls crop long input | `extract_features.py` and `run_variant_ov75.py` pass a 4096-point FFT length for a 32000-sample window. NumPy crops the input for that transform. Confirm the intended baseline feature definition before treating it as a whole-window spectrum. |
| A45 is an export placeholder | `a45_train_export_rf_small.py` explicitly exports an A44 ExtraTrees model under RF names. It does not train a new Random Forest, despite its filename and supplementary-index description. |
| Device label polarity differs | Offline scoring uses 1=fault. The archived runtime returns 1=normal. A shared contract and wrapper tests are needed before combining outputs. |
| Embedded assets are incomplete | The archive has no original `wavtools`, `feat_a8.extract`, complete WAV evaluation helper, model JSON files, firmware source tree or build configuration. The A47 exporter can regenerate JSON from the supplied bundle. The thesis's illustrative JSON keys also differ from A47's actual schema. *Third pass:* the deployed JSON files, `a8_eval_wav_gmm.py` (which includes its own WAV parser), the live and loop scripts and the firmware build notes are now in [`records/device/`](../records/device/) and [`records/firmware/`](../records/firmware/). `wavtools` and `feat_a8.extract` never existed. |
| Software licensing is undeclared | The dataset's CC BY 4.0 attribution is supplied. The authored code/documents have no declared general license. *Third pass: resolved.* The code is [MIT](../LICENSE)-licensed; the thesis documents remain all rights reserved. |

The absent thesis and presentation noted in the earlier `0bc3373` review are **resolved**: both are present from `c03372e` onward, and README links work. The raw archived documents are historical evidence; the corrections above qualify their technical claims.

## Reproduce the host diagnostics

Install the desktop requirements in a virtual environment, then run:

```bash
python scripts/check_snapshot.py
python scripts/review_models.py
```

The first command performs lightweight archive checks. The second reconstructs the nine-file test set, uses the archived feature functions and trusted model bundles, exports the GMM JSON, compares downstream arithmetic and samples the incompatible device front end. Outputs go to `artifacts/technical_review/` by default. Use `--out PATH` for another output directory.

Successful completion means the diagnostic ran. It does not mean the known feature mismatch has been fixed or that firmware was tested. Full training of all variants, nested grouped evaluation, a clean installation of the entire environment and physical ESP32 validation remain outside this review.

## Repair order

1. Define and enforce one desktop/device feature and label contract. Add feature-to-decision parity fixtures. *(Open. The current mismatch is tracked by a strict expected-failure test.)*
2. Re-evaluate with nested physical-bearing grouping and publish explicit split manifests and out-of-fold predictions. *(Done for the thesis pipelines in the third pass. Nested selection across all configurations is open.)*
3. Complete the firmware/WAV path, cache model parameters, use the board's supported FFT API and benchmark correct predictions stage by stage. *(Open.)*
4. Repair the optional MLP C exporter's buffer handling and verify repeated-call behavior before using that export path. *(Open.)*
5. Consolidate duplicated experiment code, repair legacy writers and metadata/evaluation units, and declare an appropriate license after ownership and reuse terms are decided. *(License and metadata done. The rest is open.)*
