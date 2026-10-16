# `A16_MLP`: holds the A9 model, not A16

The folder name, the header comment and `"feature_set": "A16"` in `preprocess.json` all say A16. The files are actually the export of the **A9 MLP**, [`mlp_A9.joblib`](../../../runs/A9-MLP_20250813_234816/): a z-score scaler on 25 features, PCA(4) and an MLP with 4→10→1 units. The scaler means, the PCA components and all weights and biases match that model.

The A16 model is a different shape. It uses a MinMax scaler, PCA(8) and an 8→10→1 MLP.

**Why it happened:** the exporter [`a46_export_mlp_MIN.py`](../../../../05_model_export/a46_export_mlp_MIN.py) gives `mlp_A9.joblib` in its usage line, but it always writes to a default folder named `A16_MLP`.

**Why the name stays:** the device report and [`commands_used.ps1`](../../commands_used.ps1) refer to this folder by that name.

**Effect:** this export was never run. `preprocess.json` was copied to the board, but the device report records that MLP inference was not callable from MicroPython. Only the A8 GMM ran on the device.

[`tests/test_exports.py`](../../../../tests/test_exports.py) checks the match.
