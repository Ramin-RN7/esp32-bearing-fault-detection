"""The device export folder named A16_MLP holds the A9 model (see its README)."""
import json
import re

import joblib
import numpy as np


def _header_arrays(text):
    arrays = {}
    for name, body in re.findall(r"static const float (\w+)\[\d+\]=\{([^}]*)\};", text):
        arrays[name] = np.array([float(v) for v in body.split(",") if v.strip()])
    return arrays


def _same(flat, values):
    values = np.asarray(values)
    return flat.size == values.size and (
        np.allclose(flat, values.ravel(), atol=1e-6) or np.allclose(flat, values.T.ravel(), atol=1e-6)
    )


def test_a16_mlp_export_folder_holds_the_a9_model(root):
    folder = root / "records" / "device" / "exports" / "A16_MLP"
    pre = json.loads((folder / "preprocess.json").read_text())
    header = _header_arrays((folder / "model_mlp.h").read_text())
    a9 = joblib.load(root / "records" / "runs" / "A9-MLP_20250813_234816" / "mlp_A9.joblib")
    a16 = joblib.load(root / "records" / "runs" / "A16-MLP_20250814_003605" / "mlp_A16.joblib")

    # Every exported parameter is A9's: z-score scaler, PCA(4), MLP 4-10-1.
    assert pre["scale"]["type"] == "z"
    np.testing.assert_allclose(pre["scale"]["mean"], a9["scaler"].mean_)
    np.testing.assert_allclose(pre["pca"]["components"], a9["pca"].components_)
    assert len(a9["mlp"].coefs_) == 2
    for i, (weights, biases) in enumerate(zip(a9["mlp"].coefs_, a9["mlp"].intercepts_)):
        assert _same(header[f"L{i}_W"], weights)
        assert _same(header[f"L{i}_B"], biases)

    # A16 has a MinMax scaler and PCA(8), so it cannot be the source.
    assert type(a16["scaler"]).__name__ == "MinMaxScaler"
    assert np.shape(pre["pca"]["components"]) != a16["pca"].components_.shape
