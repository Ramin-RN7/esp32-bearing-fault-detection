# Frozen A5 reference

What an ESP32 port of A5 MLP is checked against. The rules, the tolerances and the board's output format are in the [decision contract](../../docs/DECISION_CONTRACT.md). Frozen on 18 October 2026 by [`scripts/freeze_reference.py`](../../scripts/freeze_reference.py) and checked in CI (pushes to `main` and pull requests) by [`tests/test_reference.py`](../../tests/test_reference.py).

| File | Contents |
|---|---|
| [`contract.json`](contract.json) | Constants, filter coefficients, feature order and spread, tolerances, probe indices, SHA-256 of the 60 WAVs, library versions |
| [`models/`](models/) | `lobo_bearing_01.json` … `lobo_bearing_20.json`: the leave-one-bearing-out models (each excludes the bearing in its name). `all_bearings.json`: trained on all 20 bearings |
| [`golden/`](golden/) | H_10_0, B_11_2, H_13_0, C_19_2 and H_2_0: every stage of all 9 windows, from the PCM samples to the decision |
| [`window_features.csv`](window_features.csv) | The 25 features of all 540 windows, the autocorrelation lag gap, and P(fault) from the fold model and the all-bearings model |
| [`decisions.csv`](decisions.csv) | The 60 recording scores and decisions of both models; the fold models give the published 0.9625 |
| [`float32_check.json`](float32_check.json) | A float32 emulation of a port, as a fraction of each tolerance (at most 0.28) |

Each recording's fold model is `lobo_bearing_NN` for its bearing NN (`decisions.csv`, column `fold_model`). The all-bearings model's decisions on these recordings are on its training data, so they are not an accuracy.
