# Validation scope

Publication: 14-15 October 2026. Technical review: **15 October 2026**, against commit `c03372e3610b765f9a6acc555b584b81d72441f2`.

## Archive checks

All 236 files in that publication matched their published blob hashes and passed format-specific checks: all Python sources parsed, WAV payloads decoded, PNGs decoded, PDF pages extracted/rendered, Office ZIP/XML parts checked, CSV rows parsed and model parameter arrays inspected. See the [per-file inventory](review/file_inventory.json).

The original 228-file supplementary archive and its historical documents form the source record. Later setup/checking files and this technical review accompany it.

The standard-library snapshot checker remains available:

```bash
python scripts/check_snapshot.py
```

It checks syntax of 66 archived Python sources, all 60 WAVs and the archived best-results table. All WAVs are mono 16-bit PCM at 16,000 Hz and 10.000 seconds. A passing snapshot check establishes file readability, not scientific validity or embedded accuracy.

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

## Boundaries

- The original split manifests are absent; reconstructed filenames and reproduced scores do not make the reconstruction an original manifest.
- Recordings from the same physical bearing cross subsets in the reconstruction. The headline scores do not estimate unseen-bearing generalization.
- Comparing many configurations on the same test set introduces selection bias.
- Predictor timing includes two JSON loads per call; the approximately 64 ms thesis number is not isolated arithmetic time.
- The physical ESP32, ulab behavior, firmware build, live microphone path and historical hardware timings were not verified.
- No classifier retraining or nested grouped cross-validation was performed. Preprocessing estimators were refitted only for the second-pass comparison.
- The full original environment and every legacy script have not been executed end to end. Some archived evaluation scripts change preprocessing between training and scoring.

The scikit-learn pin matches both bundles. Other historical package versions are unknown. The diagnostic environment versions are recorded in the review summary; this is not a claim that a fresh installation of all requirements was tested.
