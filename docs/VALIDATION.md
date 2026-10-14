# Validation scope

Repository preparation date: **14 October 2026**.

The supplementary ZIP passed its archive-integrity check. Its 228 original files were preserved: the research code, data, models, results and documents retain their original bytes; the original README is stored under `docs/original-supplementary-readme.md` and the repository root has an expanded README.

The snapshot checker verifies:

- Syntax of 66 archived Python source files, without importing or executing them.
- Readability of the full PCM payload in all 60 WAV recordings.
- Recording sample rates: 16,000 Hz for all 60 WAVs.
- Recording durations: 10.000 seconds for all 60 WAVs.
- Best file-level F1 values in the archived results table: MLP 0.923077 and GMM 0.909091.

Run it from the repository root:

```bash
python scripts/check_snapshot.py
```

The values read from the results table are archived values, not newly computed predictions. Syntax and file-integrity checks do not reproduce the full ML experiments or validate embedded behavior. Original dataset splitting, model training, JSON export, firmware building/flashing and hardware latency measurements were not rerun during this publication.

The scikit-learn pin in `requirements.txt` is based on version 1.7.1 recorded inside both archived model bundles. The dependency file is not a complete reconstruction of the original environment, and a clean installation of all dependencies has not been tested as part of these snapshot checks.
