"""The archived split manifests: sizes, class balance and the documented bearing overlap."""
import csv

from bearing_audio.data import list_recordings, load_original_split
from bearing_audio.evaluation import matched_splits


def test_split_covers_all_recordings_with_thesis_sizes():
    split = load_original_split()
    recs = {r.name: r for r in list_recordings()}
    assert set(split) == set(recs)
    for subset, (healthy, fault) in {"train": (14, 28), "val": (3, 6), "test": (3, 6)}.items():
        names = [n for n, s in split.items() if s == subset]
        assert sum(recs[n].label == 0 for n in names) == healthy
        assert sum(recs[n].label == 1 for n in names) == fault


def test_manifest_window_counts(root):
    for subset, n in {"train": 378, "val": 81, "test": 81}.items():
        with open(root / "records" / "manifests" / f"split_{subset}.csv", newline="") as f:
            assert sum(1 for _ in csv.DictReader(f)) == n


def test_documented_bearing_overlap():
    split = load_original_split()
    recs = {r.name: r for r in list_recordings()}
    test = [n for n, s in split.items() if s == "test"]
    shared = sorted(n for n in test
                    if any(split[m] != "test" and recs[m].bearing == recs[n].bearing for m in split))
    assert shared == ["B_11_2", "B_14_2", "B_15_1", "C_19_2", "H_10_0", "H_13_0"]


def test_matched_splits_have_thesis_sizes_and_disjoint_bearings(table):
    for seed in range(10):
        by_recording, by_bearing = matched_splits(table, seed)
        for train, val, test in (by_recording, by_bearing):
            assert (len(train), len(val), len(test)) == (42, 9, 9)
            assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
            assert sum(table.recordings[n].label for n in test) == 6
        tr, va, te = ({table.recordings[n].bearing for n in s} for s in by_bearing)
        assert not (tr & va or tr & te or va & te)
