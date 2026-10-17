"""The thesis ablation configurations, rebuilt from their run records for a stricter evaluation (2026 follow-up).

Each of the 82 window-level configurations (A0-A39 with GMM and MLP, A42-RF, A43-ET, A44-ET, E3-GMM) is rebuilt
from its archived run folder:

* **Features:** the run's own feature code (``code.py``) runs on the two-second segments, written exactly as
  ``prep_segments.py`` wrote them. Only its imports, upper-case constants and function definitions execute; the
  training and logging part of the script is skipped.
* **Model stage:** scaler, PCA, model, outlier removal and decision rule follow the run's ``params.json`` and code.

``reproduce_original`` compares every configuration with its archived confusion counts on the original split.
The aggregation variants A40 (majority vote) and A41 (median) are added for A8 GMM and A16 MLP, which gives 86
pipelines. The remaining recording-level IDs are not separate pipelines: A8-GMM-FILE, A16-MLP-FILE and
A31-MLP-FILE use the mean aggregation that every pipeline uses here, and B1/B2 re-score the A8 and A16 models on
re-segmented test audio.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .data import Recording, load_windows, repo_root
from .features import cache_key

SEED = 42
ABLATION_TABLE = Path("results") / "ablation_runs.csv"
AGGREGATION_VARIANTS = {          # id: (base configuration, recording rule)
    "A40-A8": ("A8-GMM", "majority"), "A40-A16": ("A16-MLP", "majority"),
    "A41-A8": ("A8-GMM", "median"), "A41-A16": ("A16-MLP", "median"),
}
NOT_SEPARATE = {
    "A8-GMM-FILE": "mean aggregation of A8-GMM, the rule used for every pipeline here",
    "A16-MLP-FILE": "mean aggregation of A16-MLP, the rule used for every pipeline here",
    "A31-MLP-FILE": "mean aggregation of A31-MLP, the rule used for every pipeline here",
    "B1-A8": "re-scores the A8 model on re-segmented test audio", "B1-A16": "re-scores the A16 model on re-segmented test audio",
    "B2-A8": "re-scores the A8 model on re-segmented test audio", "B2-A16": "re-scores the A16 model on re-segmented test audio",
}
# Constants that only the model stage reads; they don't change the features.
_MODEL_ONLY_CONSTANTS = {"PCA_DIMS", "OUTLIER_PCT", "FIXED_PCT", "N_TREES", "MAX_DEPTH", "RNG", "HERE", "DATA",
                         "DATA_CAND", "ART", "ART_CAND", "RUNS", "EXPO", "OUT", "OUTDIR"}


@dataclass(frozen=True)
class Config:
    id: str
    description: str
    run: str                       # archived run folder, relative to the repository root
    model: str                     # GMM, MLP, RF, ET (or LR for the reference baseline)
    scaler: str                    # zscore, minmax, robust
    pca: int                       # 0 = no PCA
    threshold: str = "val-opt"     # GMM: val-opt or fixed-p5; the classifiers use their own 0.5 rule
    outlier: str = "none"          # GMM training only: none, mahalanobis or gmm-trim
    outlier_pct: float = 0.0
    trees: tuple = ()              # (n_estimators, max_depth) for RF and ET
    window_rule: str = "predict"   # classifiers: estimator.predict, or probability >= 0.5 (A44)
    aggregation: str = "mean"      # recording decision: mean, median or majority of the windows
    base: str = ""                 # configuration whose trained model is shared (aggregation variants)
    features_from: str = ""        # configuration whose features are used, if not its own
    seed: int = SEED               # MLP initialisation; the archived runs all used 42
    archived: dict = field(default_factory=dict, compare=False)   # archived counts and level

    @property
    def model_key(self) -> str:
        return self.base or self.id

    @property
    def feature_key(self) -> str:
        return self.features_from or self.base or self.id


def _constant(code: str, name: str):
    m = re.search(rf"^{name}\s*=\s*([0-9.]+)", code, re.M)
    return float(m.group(1)) if m else None


def configs(root: Path | None = None) -> list[Config]:
    """The 82 window-level configurations plus the 4 aggregation variants, in ablation-table order."""
    root = Path(root or repo_root())
    rows = [r for r in csv.DictReader(open(root / ABLATION_TABLE, newline="")) if r["latest_for_id"] == "True"]
    out, by_id = [], {}
    for r in rows:
        if r["level"] != "window":
            continue
        run = Path(r["run_record"])
        params = json.loads((root / run / "params.json").read_text())
        code = (root / run / "code.py").read_text(errors="replace")
        model = r["model"]
        scaler = {"zscore": "zscore", "minmax": "minmax", "robust(p25-p75)": "robust"}[params["scaler"]]
        kw = {}
        if model == "GMM":
            if re.search(r"=\s*keep_mask_outliers\(", code):
                kw.update(outlier="mahalanobis", outlier_pct=_constant(code, "OUTLIER_PCT"))
            elif "gmm0 = GaussianMixture" in code:
                kw.update(outlier="gmm-trim", outlier_pct=_constant(code, "OUTLIER_PCT"))
            kw["threshold"] = "fixed-p5" if "FIXED_PCT" in code else "val-opt"
        elif model in ("RF", "ET"):
            hp = params.get("model_hparams", {})
            n = _constant(code, "N_TREES") or hp.get("n_estimators")
            d = _constant(code, "MAX_DEPTH") or hp.get("max_depth")
            kw["trees"] = (int(n), int(d))
        if model != "GMM" and re.search(r"probs\s*>=\s*0\.5", code):
            kw["window_rule"] = "proba"
        cfg = Config(r["id"], r["description"], str(run), model, scaler, int(params.get("pca_dims") or 0),
                     archived={k: int(r[k]) for k in ("tp", "fp", "tn", "fn")} | {"level": "window"}, **kw)
        out.append(cfg)
        by_id[cfg.id] = cfg
    for r in rows:
        if r["id"] in AGGREGATION_VARIANTS:
            base, rule = AGGREGATION_VARIANTS[r["id"]]
            b = by_id[base]
            out.append(Config(r["id"], f"{b.description} — {rule} of window decisions" if rule == "majority"
                              else f"{b.description} — median window score", b.run, b.model, b.scaler, b.pca,
                              b.threshold, b.outlier, b.outlier_pct, b.trees, b.window_rule, rule, base=base,
                              archived={k: int(r[k]) for k in ("tp", "fp", "tn", "fn")} | {"level": "recording"}))
    return out


BASELINE = Config("BASELINE", "Reference: logistic regression on RMS and kurtosis of the A8 window (2026 follow-up)",
                  "", "LR", "zscore", 0, features_from="A8-GMM")


# --------------------------------------------------------------------------------------------- features

def export_segments(recordings: list[Recording], folder: Path) -> list[str]:
    """Write every two-second segment as the int16 WAV that prep_segments.py wrote; return the file names."""
    names = []
    for r in recordings:
        sub = Path(folder) / "all" / ("normal" if r.label == 0 else "anomaly")
        sub.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(load_windows(r), 1):
            seg = np.round(w.astype(np.float64) * 32768.0).astype(np.int16)
            name = f"{r.path.stem}__seg{i:04d}.wav"
            sf.write(sub / name, seg, 16000, subtype="PCM_16")
            names.append(name)
    return names


def _definitions(code: str) -> list[ast.stmt]:
    """Imports, upper-case constants and function definitions of an archived run script."""
    keep = []
    for node in ast.parse(code).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = [n for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if names and all(n.id.isupper() for n in names):
                keep.append(node)
    return keep


def _imports_only(node: ast.stmt, package: str) -> bool:
    if isinstance(node, ast.Import):
        return all(a.name.split(".")[0] == package for a in node.names)
    return isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package


def feature_signature(code: str) -> str:
    """Hash of the parts of a script that can affect its features (model-only constants and paths left out)."""
    parts = []
    for node in _definitions(code):
        if isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if names <= _MODEL_ONLY_CONSTANTS:
                continue
        parts.append(ast.unparse(node))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def script_features(code_path: Path, segments: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run an archived script's own load_split() on the exported segments: features, labels and file names."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    code = Path(code_path).read_text(errors="replace")
    ns = {"__file__": str(Path(segments) / "script.py"), "__name__": "archived_run"}
    for node in _definitions(code):
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(code_path), "exec"), ns)
        except NameError:          # a constant computed from training data, e.g. HEALTHY = (y_train == 0)
            continue
        except ImportError:        # matplotlib: the scripts only plot confusion matrices with it
            if not _imports_only(node, "matplotlib"):
                raise
    ns["DATA"] = Path(segments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ns["load_split"]("all")
    # Keep the script's dtype (float32 for all of them): scaling float64 instead changes later digits and can flip
    # a few MLP window decisions.
    return np.asarray(out[0]), np.asarray(out[1], dtype=int), [Path(p).name for p in out[2]]


def ablation_features(recordings: list[Recording], cfgs: list[Config], cache_dir: Path | None = None,
                      workers: int = 1, progress=None) -> dict[str, np.ndarray]:
    """Window features for every configuration (keyed by model_key), computed once per distinct feature code.

    Rows follow the window order of FeatureTable: recordings in list order, windows in time order. Results are
    cached per feature signature together with the audio, feature-code and library key of ``cache_key``.
    """
    root = repo_root()
    audio_key = cache_key(recordings)
    by_sig: dict[str, list[str]] = {}
    code_of: dict[str, Path] = {}
    for c in cfgs:
        if c.base or c.features_from or not c.run:
            continue
        code = (root / c.run / "code.py").read_text(errors="replace")
        sig = feature_signature(code)
        by_sig.setdefault(sig, []).append(c.id)
        code_of.setdefault(sig, root / c.run / "code.py")
    feats: dict[str, np.ndarray] = {}
    todo = []
    for sig, ids in by_sig.items():
        path = Path(cache_dir) / f"{sig[:20]}.npz" if cache_dir else None
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == audio_key:
                    for i in ids:
                        feats[i] = z["X"]
                    continue
        todo.append(sig)
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            names = export_segments(recordings, Path(tmp))
            expected = sorted(n for n in names if n.startswith("H_")) + sorted(n for n in names if not n.startswith("H_"))
            order = {n: i for i, n in enumerate(names)}
            jobs = [(sig, code_of[sig], Path(tmp)) for sig in todo]
            if workers > 1:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
                    results = list(ex.map(_features_job, jobs))
            else:
                results = [_features_job(j) for j in jobs]
            for done, (sig, X, files) in enumerate(results, 1):
                if files != expected:
                    raise RuntimeError(f"{code_of[sig]}: unexpected window order")
                X = X[np.argsort([order[f] for f in files], kind="stable")]     # back to recording order
                for i in by_sig[sig]:
                    feats[i] = X
                if cache_dir:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.savez(Path(cache_dir) / f"{sig[:20]}.npz", key=audio_key, X=X)
                if progress:
                    progress(done, len(results))
    for c in cfgs:
        feats.setdefault(c.id, feats[c.feature_key])
    return feats


def _features_job(job):
    sig, code_path, segments = job
    X, _, files = script_features(code_path, segments)
    return sig, X, files


# --------------------------------------------------------------------------------------------- models

def _scaler(name: str):
    return {"zscore": StandardScaler(), "minmax": MinMaxScaler(),
            "robust": RobustScaler(quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True)}[name]


def _keep_mahalanobis(Z: np.ndarray, pct: float) -> np.ndarray:
    """A14/A17: keep the healthy windows inside the (1 - pct) quantile of squared Mahalanobis distance."""
    if pct <= 0.0:
        return np.ones(len(Z), dtype=bool)
    Z = Z.astype(np.float64)
    D = Z - Z.mean(axis=0, keepdims=True)
    S = np.cov(Z, rowvar=False)
    Si = np.linalg.inv(S + 1e-6 * np.eye(S.shape[0], dtype=np.float64))
    d2 = np.einsum("ij,jk,ik->i", D, Si, D)
    return d2 <= np.percentile(d2, 100.0 * (1.0 - pct))


class Fitted:
    """One configuration trained on one training set (and validation set, for the GMM threshold)."""

    def __init__(self, cfg: Config, Xtr, ytr, Xva, yva):
        self.cfg = cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if cfg.model == "LR":                                    # reference baseline: RMS and kurtosis
                self.scaler = StandardScaler().fit(Xtr[:, [2, 6]])
                self.est = LogisticRegression(class_weight="balanced").fit(self.scaler.transform(Xtr[:, [2, 6]]), ytr)
                self.threshold = 0.5
                return
            self.scaler = _scaler(cfg.scaler).fit(Xtr)
            self.pca = PCA(n_components=cfg.pca, random_state=SEED).fit(self.scaler.transform(Xtr)) if cfg.pca else None
            Ztr = self._z(Xtr)
            if cfg.model == "GMM":
                Zh = Ztr[ytr == 0]
                if cfg.outlier == "mahalanobis":
                    Zh = Zh[_keep_mahalanobis(Zh, cfg.outlier_pct)]
                elif cfg.outlier == "gmm-trim":
                    s = GaussianMixture(n_components=3, covariance_type="diag", random_state=SEED).fit(Zh).score_samples(Zh)
                    Zh = Zh[s > np.percentile(s, 100 * cfg.outlier_pct)]
                self.est = GaussianMixture(n_components=3, covariance_type="diag", random_state=SEED).fit(Zh)
                sv = self.est.score_samples(self._z(Xva))
                if cfg.threshold == "fixed-p5":
                    self.threshold = float(np.percentile(sv[yva == 0], 5.0))
                else:
                    best = -1.0
                    for t in np.percentile(sv, np.linspace(0.5, 99.5, 200)):
                        f1 = f1_score(yva, (sv < t).astype(int), zero_division=0)
                        if f1 > best:
                            best, self.threshold = f1, float(t)
            else:
                if cfg.model == "MLP":
                    self.est = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", solver="adam", max_iter=500,
                                             random_state=cfg.seed)
                elif cfg.model == "RF":
                    self.est = RandomForestClassifier(n_estimators=cfg.trees[0], max_depth=cfg.trees[1],
                                                      random_state=SEED)
                elif cfg.model == "ET":
                    self.est = ExtraTreesClassifier(n_estimators=cfg.trees[0], max_depth=cfg.trees[1],
                                                    random_state=SEED)
                else:
                    raise ValueError(cfg.model)
                self.est.fit(Ztr, ytr)
                self.threshold = 0.5

    def _z(self, X):
        Z = self.scaler.transform(X)
        return self.pca.transform(Z) if self.pca is not None else Z

    def window_scores(self, X) -> np.ndarray:
        """GMM: log-likelihood (lower = more fault-like). Classifiers: fault probability."""
        if self.cfg.model == "LR":
            return self.est.predict_proba(self.scaler.transform(X[:, [2, 6]]))[:, 1]
        Z = self._z(X)
        return self.est.score_samples(Z) if self.cfg.model == "GMM" else self.est.predict_proba(Z)[:, 1]

    def window_pred(self, X) -> np.ndarray:
        s = self.window_scores(X)
        if self.cfg.model == "GMM":
            return (s < self.threshold).astype(int)
        if self.cfg.model == "LR" or self.cfg.window_rule == "proba":
            return (s >= 0.5).astype(int)
        return np.asarray(self.est.predict(self._z(X)), dtype=int)


def recording_decision(cfg: Config, scores: np.ndarray, window_pred: np.ndarray, threshold: float):
    """Recording score and decision from its window scores, by the configuration's aggregation rule."""
    if cfg.aggregation == "majority":
        s = float(np.mean(window_pred))
        return s, int(s >= 0.5)
    s = float(np.median(scores) if cfg.aggregation == "median" else np.mean(scores))
    return s, int(s < threshold) if cfg.model == "GMM" else int(s >= threshold)


# --------------------------------------------------------------------------------------------- evaluation

def run_split(cfgs: list[Config], feats: dict[str, np.ndarray], table, train, val, test) -> dict:
    """Train every configuration on one split (one fit per shared model) and decide the test recordings.

    Returns {config id: {"recordings": [(name, true, pred, score)], "window_true": [...], "window_pred": [...],
    "threshold": t}}.
    """
    tr, va = table.mask(train), table.mask(val)
    fits = {}
    out = {}
    for cfg in cfgs:
        key = cfg.model_key
        X = feats[cfg.feature_key]
        if key not in fits:
            base = next(c for c in cfgs if c.id == key) if cfg.base else cfg
            fits[key] = Fitted(base, X[tr], table.label[tr], X[va], table.label[va])
        fit = fits[key]
        recs, wt, wp = [], [], []
        for name in test:
            m = table.rec == name
            scores = fit.window_scores(X[m])
            pred_w = fit.window_pred(X[m])
            score, pred = recording_decision(cfg, scores, pred_w, fit.threshold)
            recs.append((str(name), int(table.recordings[name].label), pred, score))
            wt += table.label[m].tolist()
            wp += pred_w.tolist()
        out[cfg.id] = {"recordings": recs, "window_true": wt, "window_pred": wp, "threshold": float(fit.threshold)}
    return out


def confusion(true, pred) -> dict:
    t, p = np.asarray(true), np.asarray(pred)
    return {"tp": int(((t == 1) & (p == 1)).sum()), "fp": int(((t == 0) & (p == 1)).sum()),
            "tn": int(((t == 0) & (p == 0)).sum()), "fn": int(((t == 1) & (p == 0)).sum())}


def balanced_accuracy(c: dict) -> float:
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    spec = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0
    return (rec + spec) / 2


def fault_f1(c: dict) -> float:
    return 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] else 0.0


def pooled(results: list[dict], cfg_id: str) -> dict:
    """Recording- and window-level confusion counts pooled over folds."""
    rt, rp, wt, wp = [], [], [], []
    for r in results:
        rr = r[cfg_id]
        rt += [x[1] for x in rr["recordings"]]
        rp += [x[2] for x in rr["recordings"]]
        wt += rr["window_true"]
        wp += rr["window_pred"]
    return {"recording": confusion(rt, rp), "window": confusion(wt, wp)}


def select(cfgs: list[Config], inner: list[dict]) -> tuple[str, dict]:
    """Nested selection: highest recording-level balanced accuracy on the inner folds; ties go to the higher
    window-level balanced accuracy, then to the earlier configuration in the ablation table."""
    scores = {}
    for i, c in enumerate(cfgs):
        p = pooled(inner, c.id)
        scores[c.id] = (balanced_accuracy(p["recording"]), balanced_accuracy(p["window"]), -i)
    best = max(scores, key=scores.get)
    return best, {k: {"recording_ba": v[0], "window_ba": v[1]} for k, v in scores.items()}


def bootstrap(predictions: dict[str, list[dict]], recordings: dict, pairs=(), n_resamples: int = 10000,
              seed: int = 42) -> dict:
    """Bearing-level percentile intervals for recording balanced accuracy, predictions fixed (as in evaluation.py)."""
    keys = list(predictions)
    bearings = sorted({recordings[p["recording"]].bearing for p in predictions[keys[0]]})
    index = {b: i for i, b in enumerate(bearings)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(bearings), size=(n_resamples, len(bearings)))
    counts = np.zeros((n_resamples, len(bearings)))
    np.add.at(counts, (np.repeat(np.arange(n_resamples), len(bearings)), draws.ravel()), 1)
    resampled = {}
    for k in keys:
        tally = np.zeros((4, len(bearings)))
        for p in predictions[k]:
            i, row = index[recordings[p["recording"]].bearing], 2 * int(p["true"])
            tally[row, i] += 1
            tally[row + 1, i] += int(p["pred"] == p["true"])
        resampled[k] = (counts @ tally[1] / (counts @ tally[0]) + counts @ tally[3] / (counts @ tally[2])) / 2
    interval = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"n_resamples": n_resamples, "seed": seed, "n_bearings": len(bearings),
            "balanced_accuracy_95": {k: interval(v) for k, v in resampled.items()},
            "paired_difference_95": {f"{a} - {b}": interval(resampled[a] - resampled[b]) for a, b in pairs}}
