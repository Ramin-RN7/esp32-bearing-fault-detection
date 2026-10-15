"""Recordings, two-second windows and the archived split, reproduced exactly as the thesis scripts created them.

The thesis pipeline wrote intermediate WAV files (`prep_segments.py`) and read them back in each ablation
script. The int16 round trips are emulated here so that features match the archived runs bit for bit.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
WINDOW_S = 2.0
HOP_S = 1.0
TARGET_PEAK_DBFS = -1.0
_NAME = re.compile(r"^([HIOBC])_(\d+)_(\d)_mic_16000Hz\.wav$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Recording:
    name: str        # e.g. "B_11_2"
    path: Path
    condition: str   # H healthy, I inner race, O outer race, B ball, C cage (the bearing's final condition)
    bearing: int     # physical bearing ID (first number in the dataset file name)
    state: int       # 0 healthy, 1 fault developing, 2 faulty
    label: int       # 0 healthy, 1 fault (developing or faulty)


def list_recordings(wav_root: Path | None = None) -> list[Recording]:
    """All 60 recordings under 01_uottawa_wavs/wav/{normal,fault}."""
    wav_root = Path(wav_root or repo_root() / "01_uottawa_wavs" / "wav")
    recs = []
    for folder, label in (("normal", 0), ("fault", 1)):
        for path in sorted((wav_root / folder).glob("*.wav")):
            m = _NAME.match(path.name)
            if not m:
                raise ValueError(f"Unexpected file name: {path.name}")
            condition, bearing, state = m.group(1), int(m.group(2)), int(m.group(3))
            if (condition == "H") != (label == 0) or (state == 0) != (label == 0):
                raise ValueError(f"Label/name mismatch: {path}")
            recs.append(Recording(f"{condition}_{bearing}_{state}", path, condition, bearing, state, label))
    if not recs:
        raise FileNotFoundError(f"No WAV files under {wav_root}")
    return recs


def load_windows(rec: Recording) -> list[np.ndarray]:
    """Two-second windows (1 s hop) as float32, matching what the ablation scripts read from disk."""
    y, sr = sf.read(rec.path, always_2d=False)                 # prep_segments.standardize()
    if sr != SAMPLE_RATE:
        raise ValueError(f"{rec.path}: expected {SAMPLE_RATE} Hz, got {sr}")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    peak = float(np.max(np.abs(y)) + 1e-12)
    y = y * min(1.0, (10.0 ** (TARGET_PEAK_DBFS / 20.0)) / peak)
    std_i16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)          # data/wav16k/*.wav
    x = std_i16.astype(np.float32) / 32767.0                                # prep_segments.segment_file()
    win, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    windows = []
    for start in range(0, len(x) - win + 1, hop):
        seg_i16 = (np.clip(x[start:start + win], -1.0, 1.0) * 32767.0).astype(np.int16)   # data/segments/*.wav
        windows.append(seg_i16.astype(np.float32) / 32768.0)                              # sf.read(dtype="float32")
    return windows


def load_original_split(manifest_dir: Path | None = None) -> dict[str, str]:
    """Recording name -> 'train' / 'val' / 'test', from the archived manifests used by every thesis run."""
    manifest_dir = Path(manifest_dir or repo_root() / "records" / "manifests")
    split: dict[str, str] = {}
    for subset in ("train", "val", "test"):
        with open(manifest_dir / f"split_{subset}.csv", newline="") as f:
            for row in csv.DictReader(f):
                name = row["orig_file"].split("_mic")[0]
                if split.setdefault(name, subset) != subset:
                    raise ValueError(f"{name} appears in more than one subset")
    return split
