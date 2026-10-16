"""The feature cache must never return features for audio or code that has changed."""
import shutil

import numpy as np

from bearing_audio import features
from bearing_audio.data import Recording, list_recordings
from bearing_audio.features import FeatureTable


def _copy(tmp_path, rec):
    path = tmp_path / rec.path.name
    shutil.copyfile(rec.path, path)
    return Recording(rec.name, path, rec.condition, rec.bearing, rec.state, rec.label)


def test_changed_audio_with_the_same_name_and_size_is_recomputed(tmp_path):
    recs = {r.name: r for r in list_recordings()}
    original, other = recs["H_10_0"], recs["O_10_2"]
    assert original.path.stat().st_size == other.path.stat().st_size
    rec, cache = _copy(tmp_path, original), tmp_path / "features.npz"
    before = FeatureTable([rec], cache=cache).a8
    shutil.copyfile(other.path, rec.path)                 # same file name and byte count, different audio
    after = FeatureTable([rec], cache=cache).a8
    np.testing.assert_array_equal(after, FeatureTable([rec]).a8)
    assert not np.allclose(before, after)


def test_a_feature_spec_change_invalidates_the_cache(tmp_path, monkeypatch):
    rec, cache = _copy(tmp_path, list_recordings()[0]), tmp_path / "features.npz"
    FeatureTable([rec], cache=cache)
    calls, real = [], features.features25
    monkeypatch.setattr(features, "features25", lambda *a, **k: calls.append(1) or real(*a, **k))
    FeatureTable([rec], cache=cache)
    assert not calls                                      # nothing changed: served from the cache
    monkeypatch.setattr(features, "FEATURE_SPEC_VERSION", "changed for this test")
    FeatureTable([rec], cache=cache)
    assert calls                                          # spec changed: recomputed


def test_a_cache_that_does_not_line_up_with_the_audio_is_ignored(tmp_path):
    rec, cache = _copy(tmp_path, list_recordings()[0]), tmp_path / "features.npz"
    good = FeatureTable([rec], cache=cache)
    np.savez(cache, key=features.cache_key([rec]), a8=good.a8[:-1], a16=good.a16[:-1], rec=good.rec[:-1])
    reloaded = FeatureTable([rec], cache=cache)           # right key, one window missing
    assert reloaded.a8.shape == good.a8.shape
    np.testing.assert_array_equal(reloaded.a8, good.a8)
