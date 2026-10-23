"""scripts/freeze_reference.py builds into a staging copy and publishes it only after every check has passed, so a
failed regeneration leaves the frozen reference exactly as it was. Without --write it only checks that the
reference reproduces, allowing the last-bit differences another CPU makes but no other decision."""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bearing_audio import reference as R
from bearing_audio.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _freeze_module():
    spec = importlib.util.spec_from_file_location("freeze_reference", ROOT / "scripts" / "freeze_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FZ = _freeze_module()


def snapshot(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def published(tmp_path):
    """A copy of reference/a5/ to regenerate, so the real one is never touched."""
    out = tmp_path / "a5"
    shutil.copytree(ROOT / "reference" / "a5", out)
    return out


def test_failure_after_the_models_are_written_leaves_the_reference_unchanged(published, monkeypatch):
    """The real build with stand-in features: it writes all 21 model files into the staging copy, then the
    comparison with the published leave-one-bearing-out run fails. None of it reaches the published directory."""
    before = snapshot(published)
    made = []

    def stand_in_features(signal):
        made.append(np.random.default_rng(len(made)).normal(size=25).astype(np.float32))
        return made[-1]

    monkeypatch.setattr(R, "features25", stand_in_features)
    monkeypatch.setattr(FZ, "ablation_features", lambda *a, **k: {"A5-MLP": np.vstack(made)})
    overwritten = []

    def build(stage):
        try:
            FZ.build(stage, skip_float32=True)
        except FZ.FreezeError:
            staged = snapshot(stage)
            overwritten.extend(k for k in staged if k.startswith("models/") and staged[k] != before[k])
            raise

    with pytest.raises(FZ.FreezeError, match="differs from results/ablation_lobo.json"):
        FZ.freeze(published, build)
    assert len(overwritten) == 21                       # all model files had been rewritten in the staging copy
    assert snapshot(published) == before                # but the published reference is byte for byte the same
    assert [p.name for p in published.parent.iterdir()] == ["a5"]     # and no staging copy is left behind


def test_success_replaces_the_reference_and_keeps_files_the_build_does_not_write(published):
    readme = (published / "README.md").read_text()

    def build(stage):
        (stage / "models" / "lobo_bearing_01.json").write_text("{}\n")

    FZ.freeze(published, build)
    assert (published / "models" / "lobo_bearing_01.json").read_text() == "{}\n"
    assert (published / "README.md").read_text() == readme
    assert [p.name for p in published.parent.iterdir()] == ["a5"]


def test_verify_accepts_the_frozen_reference_and_rejects_an_incomplete_one(published):
    FZ.verify(published)
    (published / "golden" / "H_2_0.json").unlink()
    with pytest.raises(FZ.FreezeError, match="incomplete"):
        FZ.verify(published)


# --------------------------------------------------------------------------------------------- other CPUs
# Without AVX-512 (or AVX2), float32 sums round differently in the last bits. Retrained on such a CPU, the fold
# models make the same 60 decisions with recording scores up to about 2.5e-6 from the published run (GPT's review
# and our runs with NumPy's AVX-512 and AVX2 paths switched off, 23 October 2026). A check must accept that and
# still reject anything else.

PUBLISHED_RUN = {p["recording"]: p for p in json.loads((ROOT / "results" / "ablation_lobo.json").read_text())
                 ["lobo"]["A5-MLP"]["predictions"]}


def test_published_run_check_allows_last_bit_differences_and_nothing_else():
    exact = {n: (p["score"], p["pred"]) for n, p in PUBLISHED_RUN.items()}
    assert FZ.check_published_run(exact, PUBLISHED_RUN) == 0
    other_cpu = {n: (s + (2.5e-6 if i % 2 else -2.5e-6), d) for i, (n, (s, d)) in enumerate(exact.items())}
    assert FZ.check_published_run(other_cpu, PUBLISHED_RUN) == pytest.approx(2.5e-6)
    for name, change in (("H_10_0", lambda s, d: (s + 1e-3, d)),          # a real change in a score
                         ("H_13_0", lambda s, d: (s, 1 - d)),             # another decision
                         ("C_19_2", lambda s, d: (float("nan"), d))):     # not a number
        broken = {**exact, name: change(*exact[name])}
        with pytest.raises(FZ.FreezeError, match=f"{name}: leave-one-bearing-out result differs"):
            FZ.check_published_run(broken, PUBLISHED_RUN)
    with pytest.raises(FZ.FreezeError, match="other recordings"):
        FZ.check_published_run({n: v for n, v in exact.items() if n != "H_10_0"}, PUBLISHED_RUN)


def _edit_csv(path: Path, change):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    change(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_compare_allows_last_bit_differences_and_nothing_else(published):
    """``published`` plays the freshly built copy; the real reference/a5/ is the published one."""
    real = ROOT / "reference" / "a5"
    spread = json.loads((real / "contract.json").read_text())["features"]["spread"]
    j = FEATURE_NAMES.index("skewness")
    assert "the same 60 decisions" in FZ.compare(published, real)

    def other_cpu_scores(rows):
        for r in rows:
            r["lobo_score"] = repr(float(r["lobo_score"]) + 2.5e-6)

    def other_cpu_features(rows):
        for r in rows:
            r["skewness"] = repr(float(r["skewness"]) + 2e-6 * spread[j])
            r["p_lobo"] = repr(float(r["p_lobo"]) - 2.5e-6)

    _edit_csv(published / "decisions.csv", other_cpu_scores)
    _edit_csv(published / "window_features.csv", other_cpu_features)
    assert "the same 60 decisions" in FZ.compare(published, real)

    def flip(rows):
        rows[3]["lobo_decision"] = str(1 - int(rows[3]["lobo_decision"]))

    def feature_off(rows):
        rows[7]["kurtosis"] = repr(float(rows[7]["kurtosis"]) * 1.5)

    for name, change, message in (("decisions.csv", flip, "lobo_decision differs"),
                                  ("window_features.csv", feature_off, "kurtosis differs")):
        copy = published.parent / f"copy-{name}"
        shutil.copytree(published, copy)
        _edit_csv(copy / name, change)
        with pytest.raises(FZ.FreezeError, match=message):
            FZ.compare(copy, real)
