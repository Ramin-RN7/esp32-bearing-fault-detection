# Validation

What has been checked in this repository, how to repeat each check, and what has not been checked. The open problems are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## The thesis results

- **The two headline results reproduce exactly.** [`src/bearing_audio/`](../src/bearing_audio/) re-implements the A8 GMM and A16 MLP pipelines from the archived scripts, including the intermediate WAV round trips. Retraining on the archived split gives both thesis confusion matrices at window and recording level, including which recordings are misclassified ([`tests/test_reproduction.py`](../tests/test_reproduction.py)). For the other ablation configurations, see the 84-of-86 rebuild below.
- **The archived model gives the logged scores.** With the archived A8 model, today's features reproduce the per-window scores logged on 14 August 2025 to within 1e-3.
- **The split is the original one.** The manifests in [`records/manifests/`](../records/manifests/) hold the split every run used. [`tests/test_manifests.py`](../tests/test_manifests.py) checks its sizes, its window counts and the documented bearing overlap.

## The follow-up evaluations (2026)

- **The two thesis pipelines.** [`scripts/evaluate_protocols.py`](../scripts/evaluate_protocols.py) runs leave-one-bearing-out with every fold, threshold and held-out score; bearing-level bootstrap intervals and paired differences; ten other choices of validation bearings; the same evaluation without the ball-fault bearings; and 200 matched repeated splits by recording and by bearing ([results](../results/protocol_comparison.md)).
- **All 86 ablation pipelines.** [`scripts/evaluate_ablation.py`](../scripts/evaluate_ablation.py) rebuilds each pipeline from its run record: the run's own `code.py` computes the features and the model stage follows its settings. On the original split, 84 of the 86 reproduce their archived confusion counts exactly. The other two, A40-A16 and A41-A16, were scored by helpers with a different front end ([known issues §5](KNOWN_ISSUES.md#5-thesis-text-versus-code-and-records)); rebuilt with the training front end, they match A16-MLP-FILE. Every pipeline is then tested on unseen bearings, with nested selection over all 86 ([results](../results/ablation_lobo.md)).
- **Cross-checks.** For A16 MLP, A8 GMM and the baseline, the rebuilt pipelines give exactly the leave-one-bearing-out decisions, validation choices and no-ball-fault results of `protocol_comparison.json`. Tests check this.
- **Fold integrity.** Tests check that leave-one-bearing-out folds and bearing-grouped splits never share a bearing, that every recording is held out exactly once, and that nested selection never sees the held-out bearing.
- **Re-runs.** Both evaluations were re-run from a clean commit with the pinned environment (the commit is recorded in each JSON), with unchanged predictions.
- **Independent re-runs.** One re-ran `evaluate_protocols.py` on Python 3.12 with older NumPy, SciPy and librosa: the same original-split and leave-one-bearing-out decisions for A8 GMM, A16 MLP and the baseline, and the same configuration table ([`results/ablation_runs.csv`](../results/ablation_runs.csv)). Continuous scores differed slightly, and one of the 1,200 repeated-split rows changed (GMM, recording protocol, seed 180). Another re-ran A5 MLP's 20 leave-one-bearing-out folds and got every recording decision (balanced accuracy 0.9625, fault F1 0.975). The full 86-pipeline run and nested selection have not been re-run independently.

## The frozen A5 reference for a board port (2026)

[`scripts/freeze_reference.py`](../scripts/freeze_reference.py) builds [`reference/a5/`](../reference/a5/) for the [decision contract](DECISION_CONTRACT.md) in a staging copy and publishes it only if:

- the reference implementation ([`reference.py`](../src/bearing_audio/reference.py), no librosa) gives the archived A5 script's features **in every bit** for all 540 windows;
- the 20 fold models give exactly the published leave-one-bearing-out decisions and scores of A5 MLP (58 of 60 correct), and their JSON export computes P(fault) within 3e-7 of scikit-learn;
- a float32 emulation of a port ([`port32.py`](../src/bearing_audio/port32.py)) stays within every tolerance on all 540 windows (at most 0.28 of any) with the same 60 decisions.

In CI, [`tests/test_reference.py`](../tests/test_reference.py) repeats the bit-for-bit feature check on the golden recordings by running A5's archived `code.py`. It also recomputes the golden outputs and five of the models, checks the float32 emulation on the golden recordings, and checks that `scripts/check_board.py` fails a port with a forward-only notch. [`tests/test_check_board.py`](../tests/test_check_board.py) checks that the board checker rejects malformed files (missing, non-finite or wrongly shaped values; missing, repeated or out-of-range window IDs), fails a score or decision that disagrees with the board's own window probabilities, matches windows by ID, and requires all 60 recordings with their fold models in `--complete` mode. [`tests/test_freeze_reference.py`](../tests/test_freeze_reference.py) checks that a regeneration that fails after writing its models leaves the published reference unchanged. None of this involves a board.

## The device runtime (on a PC)

- The archived MicroPython runtime for scaler, PCA and GMM, run on a PC with NumPy in place of ulab, makes the same decisions as scikit-learn on the desktop features. The CI test ([`tests/test_device_runtime.py`](../tests/test_device_runtime.py)) covers 54 windows from six test recordings, with log-likelihoods within 0.01; `scripts/review_models.py`, run by hand, covers all 81 test windows with two feature implementations (within about 0.0014). This checks the arithmetic, not ulab or the board.
- A strict expected-failure test records that the device feature extractor does not give the desktop decisions ([known issues §1](KNOWN_ISSUES.md#1-desktop-and-device-features-have-different-meanings)).

## The archive

- `python scripts/check_snapshot.py` checks the syntax of the archived Python scripts, reads all 60 WAVs (mono, 16-bit, 16 kHz, 10 s) and checks the archived best-results table. It shows the files are readable, not that the science is right.
- The research records in [`records/`](../records/) were scanned for credentials and personal contact details before publication.

## Continuous integration

The tests run with GitHub Actions on pushes to `main` and on pull requests ([`tests.yml`](../.github/workflows/tests.yml)), on Python 3.13 with the versions pinned in [`requirements-lock.txt`](../requirements-lock.txt). They include the exact A8 and A16 reproduction on the thesis split, an exact rebuild of A16 from its archived feature code, a recomputation of one leave-one-bearing-out fold of the thesis pipelines, fold-integrity checks, consistency checks between the published result files and the frozen A5 reference checks above. CI does not re-run the 86-pipeline evaluation, nested selection or the bootstrap; those full evaluations are run by hand ([RUNNING.md](RUNNING.md)).

## Not checked

- **Independent data.** Every result comes from the same 20 bearings, recorded on one test rig with one microphone. Leave-one-bearing-out estimates performance on unseen bearings of this rig; it does not replace an independent test set.
- **Refitting and selection variation.** The bootstrap intervals resample bearings with the predictions fixed; they condition on the fitted predictions and don't capture the variation from retraining or from the pipeline selection. With that caveat:
  - Nested selection, which includes the choice among 86 pipelines, scores +0.088 to +0.287 above the RMS + kurtosis baseline (paired 95% interval).
  - A5 MLP's own interval (+0.100 to +0.300) ignores that A5 was picked as the best of 86 after seeing the results, so it is exploratory. On this dataset the hindsight and nested estimates differed by 0.0125.
  - The checks in [`ablation_lobo.md`](../results/ablation_lobo.md) §4 (other seeds, other validation bearings, no ball-fault bearings) test the leading pipelines one by one, not the nested procedure.
  - A16 MLP's lead over the baseline is not established (−0.025 to +0.200).
- **The bearing overlap in the thesis split.** Random splits of the same size that keep bearings apart scored within 0.02 on average, but they reuse the same 20 bearings, so this doesn't prove the overlap was harmless.
- **The board.** The physical ESP32, ulab behavior, the firmware build, a live microphone path and the historical hardware timings were not re-verified. The archived ~64 ms predictor timing includes two JSON loads per call. The frozen reference defines what a new board must reproduce, but no board has run it yet.
- **Legacy scripts.** Not every archived script has been run end to end, and some archived evaluation scripts change preprocessing between training and scoring.
- **The original environment.** scikit-learn 1.7.1 matches both archived model bundles; the other historical package versions are unknown.
