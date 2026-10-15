"""The archived MicroPython files, run on the host with NumPy standing in for ulab."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pytest

from bearing_audio.data import list_recordings, load_windows


def _load_device_module(root, name, monkeypatch):
    ulab = types.ModuleType("ulab")
    ulab.numpy = np                      # no ulab.fft attribute -> feat_a8 uses its DFT fallback, as on the board
    monkeypatch.setitem(sys.modules, "ulab", ulab)
    monkeypatch.setitem(sys.modules, "ujson", json)
    spec = importlib.util.spec_from_file_location(name, root / "records" / "device" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(root, monkeypatch):
    rt = _load_device_module(root, "model_rt_gmm", monkeypatch)
    exports = root / "records" / "device" / "exports" / "A8_GMM"
    rt.jload = lambda p: json.loads((exports / Path(p).name).read_text())
    return rt


def test_deployed_json_is_the_archived_a8_model(root):
    bundle = joblib.load(root / "05_model_export" / "gmm_A8.joblib")
    exports = root / "records" / "device" / "exports" / "A8_GMM"
    pre = json.loads((exports / "preprocess.json").read_text())
    gmm = json.loads((exports / "gmm.json").read_text())
    np.testing.assert_array_equal(pre["scale"]["mean"], bundle["scaler"].mean_)
    np.testing.assert_array_equal(pre["pca"]["components"], bundle["pca"].components_)
    np.testing.assert_array_equal(gmm["means"], bundle["gmm"].means_)
    assert gmm["thr"] == bundle["thr"]


def test_device_runtime_matches_scikit_learn_on_desktop_features(table, runtime, root):
    bundle = joblib.load(root / "05_model_export" / "gmm_A8.joblib")
    X = table.a8[np.isin(table.rec, ["H_10_0", "H_13_0", "H_18_0", "B_11_2", "C_18_1", "C_19_2"])]
    expected = bundle["gmm"].score_samples(bundle["pca"].transform(bundle["scaler"].transform(X)))
    got = np.array([runtime.predict_gmm(list(map(float, x)))[1] for x in X])
    np.testing.assert_allclose(got, expected, atol=0.01)
    device_says_normal = np.array([runtime.predict_gmm(list(map(float, x)))[0] for x in X])
    np.testing.assert_array_equal(device_says_normal, (expected >= bundle["thr"]).astype(int))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Known issue: feat_a8.py computes 25 PSD band energies, not the 25 desktop "
                                       "features the exported model expects (records/device/REPORT_DevicePhase.md).")
def test_device_features_give_the_desktop_decision(runtime, root, monkeypatch):
    feat = _load_device_module(root, "feat_a8", monkeypatch)
    recs = {r.name: r for r in list_recordings()}
    for name, healthy in (("H_10_0", True), ("B_11_2", False)):     # the two WAVs used on the board
        frame_ll = []
        window = load_windows(recs[name])[0]                         # seg0001, as copied to /wav/
        for start in range(0, len(window) - 256 + 1, 4096):          # 8 frames of 256 samples
            y, ll, thr = runtime.predict_gmm(feat.feat25(list(map(float, window[start:start + 256])), 16000, 256))
            frame_ll.append(ll)
        assert (np.mean(frame_ll) >= thr) == healthy, name
