"""The re-implementation reproduces the archived thesis results exactly."""
import csv

import joblib
import numpy as np

from bearing_audio.data import load_original_split
from bearing_audio.evaluation import original


def test_thesis_results_reproduce_from_the_archived_split(table):
    res = original(table, load_original_split())
    gmm, mlp = res["a8_gmm"], res["a16_mlp"]
    assert gmm["recording"]["confusion_matrix"] == [[3, 0], [1, 5]]
    assert gmm["window"]["confusion_matrix"] == [[26, 1], [9, 45]]
    assert gmm["misclassified"] == ["C_19_2"]
    assert round(gmm["recording"]["fault_f1"], 6) == 0.909091
    assert mlp["recording"]["confusion_matrix"] == [[2, 1], [0, 6]]
    assert mlp["window"]["confusion_matrix"] == [[18, 9], [0, 54]]
    assert mlp["misclassified"] == ["H_13_0"]
    assert round(mlp["recording"]["fault_f1"], 6) == 0.923077


def test_features_match_the_original_environment(table, root):
    """Archived A8 model + today's features -> the per-window scores logged on 14 Aug 2025."""
    bundle = joblib.load(root / "05_model_export" / "gmm_A8.joblib")
    logged = list(csv.DictReader(open(root / "records" / "predictions" / "segments_A8_GMM_20250814_025831.csv")))
    by_recording = {}
    for row in logged:
        by_recording.setdefault(row["group_id"].split("_mic")[0], []).append(float(row["score"]))
    for name, scores in by_recording.items():
        X = table.a8[table.rec == name]
        ours = bundle["gmm"].score_samples(bundle["pca"].transform(bundle["scaler"].transform(X)))
        np.testing.assert_allclose(ours, scores, atol=1e-3, err_msg=name)
