# Validation

What has been checked in this repository, how to repeat each check, and what has not been checked. The open problems are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## The thesis results

- **They reproduce exactly.** [`src/bearing_audio/`](../src/bearing_audio/) re-implements the A8 and A16 pipelines from the archived scripts, including the intermediate WAV round trips. Retraining on the archived split gives both thesis confusion matrices at window and recording level, including which recordings are misclassified ([`tests/test_reproduction.py`](../tests/test_reproduction.py)).
- **The archived model gives the logged scores.** With the archived A8 model, today's features reproduce the per-window scores logged on 14 August 2025 to within 1e-3.
- **The split is the original one.** The manifests in [`records/manifests/`](../records/manifests/) hold the split every run used. [`tests/test_manifests.py`](../tests/test_manifests.py) checks its sizes, its window counts and the documented bearing overlap.

## The follow-up evaluations (2026)

- **The two thesis pipelines.** [`scripts/evaluate_protocols.py`](../scripts/evaluate_protocols.py) runs leave-one-bearing-out with every fold, threshold and held-out score; bearing-level bootstrap intervals and paired differences; ten other choices of validation bearings; the same evaluation without the ball-fault bearings; and 200 matched repeated splits by recording and by bearing ([results](../results/protocol_comparison.md)).
- **All 86 ablation pipelines.** [`scripts/evaluate_ablation.py`](../scripts/evaluate_ablation.py) rebuilds each pipeline from its run record: the run's own `code.py` computes the features and the model stage follows its settings. On the original split, 84 of the 86 reproduce their archived confusion counts exactly. The other two, A40-A16 and A41-A16, were scored by helpers with a different front end ([known issues §5](KNOWN_ISSUES.md#5-thesis-text-versus-code-and-records)); rebuilt with the training front end, they match A16-MLP-FILE. Every pipeline is then tested on unseen bearings, with nested selection over all 86 ([results](../results/ablation_lobo.md)).
- **Cross-checks.** For A16 MLP, A8 GMM and the baseline, the rebuilt pipelines give exactly the leave-one-bearing-out decisions, validation choices and no-ball-fault results of `protocol_comparison.json`. Tests check this.
- **Fold integrity.** Tests check that leave-one-bearing-out folds and bearing-grouped splits never share a bearing, that every recording is held out exactly once, and that nested selection never sees the held-out bearing.
- **Re-runs.** Both evaluations were re-run from a clean commit with the pinned environment (the commit is recorded in each JSON), with unchanged predictions. An independent re-run on Python 3.12 with older NumPy, SciPy and librosa gave the same decisions on the original split and in leave-one-bearing-out, and the same configuration table ([`results/ablation_runs.csv`](../results/ablation_runs.csv)). Continuous scores differ slightly between the environments, and one of the 1,200 repeated-split rows changed (GMM, recording protocol, seed 180).

## The device runtime (on a PC)

- The archived MicroPython runtime for scaler, PCA and GMM, run on a PC with NumPy in place of ulab, makes the same decisions as scikit-learn on the desktop features ([`tests/test_device_runtime.py`](../tests/test_device_runtime.py)). This checks the arithmetic, not ulab or the board.
- A strict expected-failure test records that the device feature extractor does not give the desktop decisions ([known issues §1](KNOWN_ISSUES.md#1-desktop-and-device-features-have-different-meanings)).

## The archive

- `python scripts/check_snapshot.py` checks the syntax of the archived Python scripts, reads all 60 WAVs (mono, 16-bit, 16 kHz, 10 s) and checks the archived best-results table. It shows the files are readable, not that the science is right.
- The research records in [`records/`](../records/) were scanned for credentials and personal contact details before publication.

## Continuous integration

The tests run on every push with GitHub Actions ([`tests.yml`](../.github/workflows/tests.yml)), on Python 3.13 with the versions pinned in [`requirements-lock.txt`](../requirements-lock.txt). They include an exact rebuild of A16 from its archived feature code and a recomputation of one leave-one-bearing-out fold. The full evaluations are run by hand ([RUNNING.md](RUNNING.md)).

## Not checked

- **Independent data.** Every result comes from the same 20 bearings, recorded on one test rig with one microphone. Leave-one-bearing-out estimates performance on unseen bearings of this rig; it does not replace an independent test set.
- **Refitting variation.** The bootstrap intervals resample bearings with the predictions fixed, so they leave out the variation from refitting. Within that limit, the lead of A5 MLP and of nested selection over the two-feature baseline is clear (paired 95% intervals +0.100 to +0.300 and +0.088 to +0.287); the lead of A16 MLP is not (−0.025 to +0.200).
- **The bearing overlap in the thesis split.** Random splits of the same size that keep bearings apart scored within 0.02 on average, but they reuse the same 20 bearings, so this doesn't prove the overlap was harmless.
- **The board.** The physical ESP32, ulab behavior, the firmware build, a live microphone path and the historical hardware timings were not re-verified. The archived ~64 ms predictor timing includes two JSON loads per call.
- **Legacy scripts.** Not every archived script has been run end to end, and some archived evaluation scripts change preprocessing between training and scoring.
- **The original environment.** scikit-learn 1.7.1 matches both archived model bundles; the other historical package versions are unknown.
