from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import csv
from tqdm import tqdm

SR = 16000
WIN_SEC = 2.0
HOP_SEC = 1.0          # 50% overlap
TARGET_PEAK_DBFS = -1.0

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
WAV16K = ROOT / "data" / "wav16k"
SEGS = ROOT / "data" / "segments"
MANIFESTS = ROOT / "manifests"
CLASSES = ["normal", "anomaly"]

def to_mono(x):
    if x.ndim == 2:
        return x.mean(axis=1)
    return x

def norm_to_dbfs(x, target_dbfs):
    t = 10.0 ** (target_dbfs / 20.0)
    peak = float(np.max(np.abs(x)) + 1e-12)
    scale = min(1.0, t / peak)
    return x * scale

def standardize(in_path: Path, out_path: Path):
    y, sr = sf.read(in_path, always_2d=False)
    y = to_mono(np.asarray(y, dtype=np.float32))
    if sr != SR:
        y = resample_poly(y, up=SR, down=sr).astype(np.float32)
    y = norm_to_dbfs(y, TARGET_PEAK_DBFS)
    y = np.clip(y, -1.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, (y * 32767.0).astype(np.int16), SR, subtype="PCM_16")

def segment_file(in_path: Path, out_dir: Path, orig_name: str, cls: str, writer):
    y, sr = sf.read(in_path, dtype="int16")
    assert sr == SR, f"Unexpected SR in {in_path}: {sr}"
    y = y.astype(np.float32) / 32767.0
    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)
    n = len(y)
    seg_idx = 0
    for start in range(0, n - win + 1, hop):
        seg = y[start:start + win]
        seg_idx += 1
        fname = f"{Path(orig_name).stem}__seg{seg_idx:04d}.wav"
        out_path = out_dir / cls / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16), SR, subtype="PCM_16")
        writer.writerow({
            "filepath": str(out_path.relative_to(ROOT)),
            "class": cls,
            "orig_file": orig_name,
            "segment_id": f"{seg_idx:04d}",
            "duration_s": f"{WIN_SEC:.2f}",
        })
    return seg_idx

def main():
    print("== Standardizing to 16 kHz mono, 16-bit PCM, -1 dBFS ==")
    for cls in CLASSES:
        in_dir = RAW / cls
        out_dir = WAV16K / cls
        files = sorted(list(in_dir.glob("*.wav")))
        if not files:
            print(f"WARNING: no WAVs in {in_dir}")
        for p in tqdm(files, desc=f"std:{cls}"):
            standardize(p, out_dir / p.name)

    print("\n== Segmenting to 2.0 s with 50% overlap ==")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS / "segments.csv"
    with open(manifest_path, "w", newline="") as f:
        fieldnames = ["filepath", "class", "orig_file", "segment_id", "duration_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        summary = {c: 0 for c in CLASSES}
        low_seg_files = []

        for cls in CLASSES:
            wav_dir = WAV16K / cls
            for p in tqdm(sorted(wav_dir.glob("*.wav")), desc=f"seg:{cls}"):
                nsegs = segment_file(p, SEGS, p.name, cls, writer)
                summary[cls] += nsegs
                if nsegs < 2:
                    low_seg_files.append(str(p.relative_to(ROOT)))

    print("\n== Summary ==")
    print(f"Segments (normal): {summary['normal']}")
    print(f"Segments (anomaly): {summary['anomaly']}")
    if low_seg_files:
        print("Files with <2 segments:")
        for s in low_seg_files:
            print("  -", s)
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
