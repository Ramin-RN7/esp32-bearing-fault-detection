# Validation scope

Publication: 14-15 October 2026. Technical review: **15 October 2026**, against commit `c03372e3610b765f9a6acc555b584b81d72441f2`.

## Archive checks

All 236 files in that publication matched their published blob hashes and passed format-specific checks: all Python sources parsed, WAV payloads decoded, PNGs decoded, PDF pages extracted/rendered, Office ZIP/XML parts checked, CSV rows parsed and model parameter arrays inspected. See the [per-file inventory](review/file_inventory.json).

The original 228-file supplementary archive and its historical documents form the source record. Later setup/checking files and this technical review accompany it.

The standard-library snapshot checker remains available:

```bash
python scripts/check_snapshot.py
```

It checks the syntax of the archived Python sources (66 at the first review; 69 since the third pass added the A5, A7 and E3 scripts), all 60 WAVs and the archived best-results table. All WAVs are mono 16-bit PCM at 16,000 Hz and 10.000 seconds. A passing snapshot check establishes file readability, not scientific validity or embedded accuracy.

## Host model checks

```bash
python scripts/review_models.py
```

Using scikit-learn 1.7.1, the later host review reconstructed the missing split from sorted published filenames and the available seed-42 splitter. On nine test recordings it reproduced A8 GMM file F1 0.909091 and A16 MLP file F1 0.923077. It also reproduced their matching training-pipeline segment matrices.

A NumPy shim of the archived scaler/PCA/GMM runtime agreed on decisions in all 162 score comparisons covering 81 distinct test windows. Maximum absolute log-likelihood difference was approximately 0.001414. This is host arithmetic compatibility, not an ulab or firmware test.

The same review confirmed a critical front-end mismatch: the desktop GMM expects mixed handcrafted descriptors, while `feat_a8.py` computes PSD band energies. All 243 sampled host-emulated device vectors fell on the fault side, including healthy inputs. See [technical review](TECHNICAL_REVIEW.md) and [summary](review/summary.json).

## Second-pass checks

All 240 files at `8c9a5a9dc69d13e3eab17dea69bdcb3a268c8280` matched GitHub and passed repeated format checks. Internal Markdown links resolved, and the host diagnostic results and confusion-matrix cross-checks repeated. The original segmentation and split scripts were executed unchanged on all 60 WAVs in an isolated directory, reproducing the same split. A8/A16 preprocessing parameters were refitted for comparison and were numerically close to the saved objects; the A8 validation threshold was recovered exactly using the saved classifier.

The optional minimal MLP C export was generated and compiled on the host. First-call decisions matched scikit-learn on all 81 test windows, but the generated predictor overwrote every input. A second call without restoring the input changed 78 decisions. Additional table/feature metadata and presentation inconsistencies are recorded in the [second-pass review](TECHNICAL_REVIEW.md#second-pass-verification) and [evidence](review/recheck.json). These findings remain open defects in the historical sources.

## Third-pass checks

The original research records are now published in [`records/`](../records/): split manifests, 108 run folders, logs, per-window predictions, device files and firmware notes. Before publication, they were scanned for credentials and personal contact details.

- The archived manifests contain exactly the split the earlier passes reconstructed.
- [`src/bearing_audio/`](../src/bearing_audio/) re-implements the A8 and A16 pipelines from the archived scripts, emulating the intermediate WAV round trips. Retraining on the archived split reproduces both thesis confusion matrices at window and recording level, including which recordings are misclassified.
- With the archived A8 model, today's features reproduce the per-window scores logged on 14 Aug 2025 to within 1e-3.
- The archived MicroPython runtime, run on the host with NumPy standing in for ulab, matches scikit-learn on desktop features. This is host arithmetic parity; parity on the board is still untested.
- A strict expected-failure test records that the device feature extractor does not reproduce the desktop decisions.
- [`scripts/evaluate_protocols.py`](../scripts/evaluate_protocols.py) adds three evaluations: leave-one-bearing-out, the same without the ball-fault bearings, and 200 matched repeated splits by recording and by bearing. Results are in [`results/protocol_comparison.md`](../results/protocol_comparison.md).
- The tests run on every push via GitHub Actions ([`tests.yml`](../.github/workflows/tests.yml)) on Python 3.13, with the pinned versions in [`requirements-lock.txt`](../requirements-lock.txt). The full evaluations above are run by hand ([running notes](RUNNING.md)).

## Fourth-pass checks

Checked on 16 October 2026:

- The thesis text was compared with the code and records. Six groups of statements differ; [§5 of the technical review](TECHNICAL_REVIEW.md#5-thesis-text-versus-code-and-records) lists them. The reproduced scores are unaffected.
- `records/device/exports/A16_MLP/` holds the A9 MLP export. [`tests/test_exports.py`](../tests/test_exports.py) checks every exported parameter against `mlp_A9.joblib`.

## Fifth-pass checks

Checked on 16 October 2026, after a separate review of `5452756`:

- **The separate review's re-run** reproduced the original-split and leave-one-bearing-out results exactly, including every held-out decision. It reproduced the rebuilt ablation table byte for byte. The 200-split summaries matched except for a GMM mean difference of 0.0004 under a different software environment (Python 3.12, older NumPy, SciPy and librosa). A later review traced that difference to one of the 1,200 per-split rows (GMM, recording protocol, seed 180). Continuous scores also differ slightly between the two environments; the decisions above do not.
- **The feature cache** returned stale features for changed audio of the same size; the review confirmed this. It is fixed and covered by regression tests.
- **Rerun.** The full evaluation was rerun from a clean commit (recorded in the JSON) with the pinned environment. The original-split and leave-one-bearing-out predictions are unchanged.
- **Published outputs:**
  - every leave-one-bearing-out fold, with its threshold and held-out scores
  - bearing-level bootstrap intervals and paired differences
  - ten alternative choices of validation bearings
  - per-split repeated results, in `results/repeated_splits.csv`
- **Integrity tests** check that leave-one-bearing-out folds and bearing-grouped splits never share a bearing. Every recording is held out exactly once. One leave-one-bearing-out fold is recomputed in CI and compared with the published predictions and scores.

## Ablation re-evaluation

Checked on 17 October 2026. [`scripts/evaluate_ablation.py`](../scripts/evaluate_ablation.py) rebuilds every ablation pipeline from its run record and tests it on unseen bearings ([results](../results/ablation_lobo.md)).

- **Rebuild.** Each run's own `code.py` computes its features; the model stage follows its settings. On the original split, 84 of the 86 pipelines reproduce their archived confusion counts exactly. The other two, A40-A16 and A41-A16, were scored by helpers with a different front end ([§5](TECHNICAL_REVIEW.md#5-thesis-text-versus-code-and-records)); rebuilt with the training front end, they match A16-MLP-FILE.
- **Cross-checks.** For A16 MLP, A8 GMM and the baseline, the rebuilt leave-one-bearing-out decisions, the ten validation choices and the results without the ball-fault bearings equal those in `protocol_comparison.json` exactly. Tests check this, and an exact rebuild of A16 (features and counts) runs in CI.
- **Selection.** Nested selection chooses among all 86 pipelines in each outer fold using only the other 19 bearings. Its balanced accuracy, 0.950 [0.89–1.00], is close to the best single pipeline's 0.963, so selection inflated the top result only slightly here.

## Boundaries

- The split is now the original one, not a reconstruction. Recordings from the same physical bearing cross subsets. Random splits of the same size that keep bearings apart scored within 0.02 on average. No large average difference was seen in these splits, but they reuse the same 20 bearings, so this does not prove the overlap is harmless. Leave-one-bearing-out results estimate performance on unseen bearings.
- The bootstrap intervals resample bearings with the predictions fixed. They leave out the variation from refitting and from the earlier choice of pipeline settings. With 20 bearings, the MLP's lead over the two-feature baseline is not established.
- Comparing many configurations on the same test set introduces selection bias. The ablation re-evaluation adds nested selection over all 86 pipelines, but every result still comes from the same 20 bearings on one test rig. It does not replace an independent test set.
- Predictor timing includes two JSON loads per call; the approximately 64 ms thesis number is not isolated arithmetic time.
- The physical ESP32, ulab behavior, firmware build, live microphone path and historical hardware timings were not verified.
- The full original environment and every legacy script have not been executed end to end. Some archived evaluation scripts change preprocessing between training and scoring.

The scikit-learn pin matches both bundles. Other historical package versions are unknown. The diagnostic environment versions are recorded in the review summary; this is not a claim that a fresh installation of all requirements was tested.
