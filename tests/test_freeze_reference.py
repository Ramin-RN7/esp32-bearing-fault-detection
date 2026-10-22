"""scripts/freeze_reference.py builds into a staging copy and publishes it only after every check has passed, so a
failed regeneration leaves the frozen reference exactly as it was."""
import hashlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from bearing_audio import reference as R

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
