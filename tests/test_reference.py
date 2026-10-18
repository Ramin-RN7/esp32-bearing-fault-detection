"""The frozen A5 reference (reference/a5/, docs/DECISION_CONTRACT.md): it equals the archived pipeline, the frozen
files reproduce, a float32 implementation can meet the contract, and the board check catches porting mistakes."""
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import filtfilt, sosfilt

from bearing_audio import ablation
from bearing_audio import port32 as P
from bearing_audio import reference as R
from bearing_audio.data import list_recordings, load_windows
from bearing_audio.evaluation import lobo_folds
from bearing_audio.features import FEATURE_NAMES, file_sha256

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "a5"
RECS = {r.name: r for r in list_recordings()}
GOLDEN = ("H_10_0", "B_11_2", "H_13_0", "C_19_2", "H_2_0")
CONTRACT = json.loads((REF / "contract.json").read_text())
SPREAD = np.asarray(CONTRACT["features"]["spread"])


def _json(path):
    return json.loads(Path(path).read_text())


def _csv(name):
    with open(REF / name, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def ref_features():
    """Reference features of all 540 windows, in list_recordings() order (about 10 s)."""
    X, names = [], []
    for r in RECS.values():
        for w in R.windows(R.normalize(R.read_pcm(r.path))[0]):
            X.append(R.features25(R.front_end(R.window_signal(w))["notch"]))
            names.append(r.name)
    return np.vstack(X), np.array(names)


def test_windows_and_notch_equal_the_thesis_pipeline():
    rec = RECS["B_11_2"]
    ours = [R.window_signal(w) for w in R.windows(R.normalize(R.read_pcm(rec.path))[0])]
    assert all(np.array_equal(a, b) for a, b in zip(ours, load_windows(rec))) and len(ours) == 9
    p = R.preemphasis(R.highpass(ours[0]))
    np.testing.assert_allclose(R.notch(p), filtfilt(R.NOTCH_B, R.NOTCH_A, p), rtol=0, atol=1e-15)
    np.testing.assert_allclose(sosfilt(R.HPF_SOS, ours[0]), R.highpass(ours[0]), rtol=0, atol=1e-9)


def test_reference_features_equal_the_archived_a5_code_in_every_bit():
    """Runs A5's own code.py (librosa, filtfilt, np.correlate) on the golden recordings' segments."""
    recs = [RECS[n] for n in GOLDEN]
    run = next(c for c in ablation.configs() if c.id == "A5-MLP").run
    with tempfile.TemporaryDirectory() as tmp:
        names = ablation.export_segments(recs, Path(tmp))
        X, _, files = ablation.script_features(ROOT / run / "code.py", Path(tmp))
    order = {n: i for i, n in enumerate(names)}
    archived = X[np.argsort([order[f] for f in files], kind="stable")]
    ours = np.vstack([R.features25(R.front_end(R.window_signal(w))["notch"])
                      for r in recs for w in R.windows(R.normalize(R.read_pcm(r.path))[0])])
    assert ours.dtype == archived.dtype == np.float32
    assert np.array_equal(ours, archived)


def test_frozen_window_features_and_decisions(ref_features):
    X, names = ref_features
    rows = _csv("window_features.csv")
    assert [r["recording"] for r in rows] == names.tolist()
    frozen = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows], dtype=np.float32)
    np.testing.assert_allclose(frozen, X, rtol=1e-6, atol=1e-12)
    assert np.array_equal(frozen[:, -1], X[:, -1])                       # the autocorrelation lag exactly
    lobo = _json(ROOT / "results" / "ablation_lobo.json")["lobo"]["A5-MLP"]
    published = {p["recording"]: p for p in lobo["predictions"]}
    decisions = _csv("decisions.csv")
    assert len(decisions) == 60
    for d in decisions:
        assert int(d["lobo_decision"]) == published[d["recording"]]["pred"]
        assert float(d["lobo_score"]) == pytest.approx(published[d["recording"]]["score"], abs=1e-6)
    assert sum(int(d["lobo_correct"]) for d in decisions) == 58          # A5 MLP: 0.9625 balanced accuracy
    assert min(float(d["lobo_margin"]) for d in decisions) > 0.1


def test_frozen_models_retrain_from_the_reference_features(ref_features):
    X, names = ref_features
    labels = np.array([RECS[n].label for n in names])
    cfg = next(c for c in ablation.configs() if c.id == "A5-MLP")
    for fold in lobo_folds(RECS)[::5] + [None]:                           # 4 of the 20 folds, and all bearings
        tr = np.isin(names, fold["train"]) if fold else np.ones(len(names), bool)
        key = f"lobo_bearing_{fold['test_bearing']:02d}" if fold else "all_bearings"
        frozen = _json(REF / "models" / f"{key}.json")
        fit = ablation.Fitted(cfg, X[tr], labels[tr], X[:0], labels[:0])
        fresh = R.export_model(fit)
        assert frozen["trained_on"] == (sorted(fold["train"]) if fold else sorted(RECS))
        for part in ("standardize", "pca", "mlp"):
            for k, v in fresh[part].items():
                if isinstance(v, str):
                    assert frozen[part][k] == v
                else:
                    np.testing.assert_allclose(frozen[part][k], v, rtol=1e-4, atol=1e-6, err_msg=f"{key} {part} {k}")
        sk = fit.window_scores(X[tr][:20])
        ours = [R.model_stages(x, frozen)["probability"][0] for x in X[tr][:20]]
        np.testing.assert_allclose(ours, sk, atol=1e-5)


@pytest.mark.parametrize("name", GOLDEN)
def test_golden_outputs_reproduce(name):
    g = _json(REF / "golden" / f"{name}.json")
    rec = RECS[name]
    assert g["wav_sha256"] == file_sha256(rec.path) == CONTRACT["audio_sha256"][name]
    tr = R.trace(rec.path, _json(REF / g["model"]))
    assert float(tr["gain"]) == g["gain"] and tr["decision"] == g["decision"]
    assert tr["score"] == pytest.approx(g["score"], abs=1e-6)
    for i, w in enumerate(tr["windows"]):
        fresh, frozen = R.golden_window(w, i), g["windows"][i]
        assert fresh["pcm"] == frozen["pcm"]
        for s in ("highpass", "preemphasis", "notch"):
            np.testing.assert_allclose(fresh[s]["probes"], frozen[s]["probes"], rtol=1e-9, atol=1e-15)
            np.testing.assert_allclose([fresh[s][k] for k in ("peak", "rms")], [frozen[s][k] for k in ("peak", "rms")],
                                       rtol=1e-9)
        np.testing.assert_allclose(fresh["features"], frozen["features"], rtol=1e-6, atol=1e-12)
        assert fresh["features"][-1] == frozen["features"][-1]
        for s in ("standardized", "pca", "hidden", "probability"):
            np.testing.assert_allclose(fresh[s], frozen[s], rtol=1e-5, atol=1e-6)


def test_contract_file_matches_the_reference():
    t = CONTRACT["tolerances"]
    assert t["feature_fraction_of_spread"] == R.FEATURE_TOLERANCE and t["signal_fraction_of_peak"] == R.SIGNAL_TOLERANCE
    assert t["probability"] == R.PROBABILITY_TOLERANCE and CONTRACT["probes"] == list(R.PROBES)
    assert CONTRACT["features"]["names"] == list(FEATURE_NAMES)
    np.testing.assert_array_equal(CONTRACT["highpass"]["b"], R.HPF_B)
    np.testing.assert_array_equal(CONTRACT["notch"]["zi"], R.NOTCH_ZI)
    assert set(CONTRACT["audio_sha256"]) == set(RECS)
    assert all((REF / m).exists() for m in CONTRACT["models"] + CONTRACT["golden"]) and len(CONTRACT["models"]) == 21


def _port_window(w, model, notch=P.notch):
    st = {"pcm": w, **P.front_end(R.window_signal(w))}
    st["notch"] = notch(st["preemphasis"])
    st["features"] = P.features25(st["notch"])
    st["probability"] = P.model_stages(st["features"], model)["probability"]
    return st


def test_float32_port_meets_the_contract_on_the_golden_recordings():
    """The tolerances can be met in single precision: every check below 1 and the same decisions."""
    for name in GOLDEN:
        g = _json(REF / "golden" / f"{name}.json")
        model = _json(REF / g["model"])
        normalized, gain = P.normalize(R.read_pcm(RECS[name].path))
        assert float(gain) == g["gain"]
        probs = []
        for i, w in enumerate(R.windows(normalized)):
            st = _port_window(w, model)
            checks = R.check_window(st, g["windows"][i], SPREAD)
            assert max(checks.values()) < 1, (name, i, max(checks, key=checks.get))
            ms = P.model_stages(np.asarray(g["windows"][i]["features"], dtype=np.float32), model)
            assert max(R.check_model(ms, g["windows"][i]).values()) < 1, (name, i)
            probs.append(float(st["probability"][0]))
        assert abs(np.mean(probs) - g["score"]) < R.PROBABILITY_TOLERANCE["recording"]
        assert (np.mean(probs) >= R.THRESHOLD) == bool(g["decision"])


def _check_board():
    spec = importlib.util.spec_from_file_location("check_board", ROOT / "scripts" / "check_board.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _board_file(tmp, name, notch=P.notch):
    key = {d["recording"]: d["fold_model"] for d in _csv("decisions.csv")}[name]
    model = _json(REF / "models" / f"{key}.json")
    windows, probs = [], []
    for i, w in enumerate(R.windows(P.normalize(R.read_pcm(RECS[name].path))[0])):
        st = _port_window(w, model, notch)
        probs.append(float(st["probability"][0]))
        windows.append({"window": i + 1, "pcm": R.pcm_summary(w),
                        **{s: R.signal_summary(st[s]) for s in ("highpass", "preemphasis", "notch")},
                        "features": [float(v) for v in st["features"]], "probability": probs[-1]})
    path = Path(tmp) / f"{name}.json"
    path.write_text(json.dumps({"recording": name, "model": f"models/{key}.json", "windows": windows,
                                "score": float(np.mean(probs)), "decision": int(np.mean(probs) >= 0.5)}))
    return path


def test_board_check_passes_a_correct_port_and_catches_a_forward_only_notch(tmp_path):
    cb = _check_board()
    table = {"decisions": {d["recording"]: d for d in _csv("decisions.csv")},
             "windows": {(r["recording"], int(r["window"])): r for r in _csv("window_features.csv")}}
    for name in ("B_11_2", "O_8_1"):                      # a golden recording, and one checked from the tables
        ok, lines = cb.check_file(_board_file(tmp_path, name), SPREAD, table)
        assert ok, lines
    forward_only = lambda p: sosfilt(P.NOTCH_SOS, p)     # a common porting mistake: no backward pass
    ok, lines = cb.check_file(_board_file(tmp_path, "B_11_2", forward_only), SPREAD, table)
    assert not ok and any("notch" in line for line in lines[1:])
