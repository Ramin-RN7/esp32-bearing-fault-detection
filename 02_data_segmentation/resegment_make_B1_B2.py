# Make B1/B2 re-segmented datasets from raw WAVs.
# Input  : data/raw/{train,val,test}/{normal,anomaly}/*.wav
# Output : data/segments_split_B1 and data/segments_split_B2 with files named <origprefix>__segXXXX.wav
# B1 = 1.0 s window, 50% overlap @16 kHz
# B2 = 2.0 s window, 75% overlap @16 kHz (hop=0.5 s)

from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import json

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT_B1 = HERE / "data" / "segments_split_B1"
OUT_B2 = HERE / "data" / "segments_split_B2"
SR = 16000

SPLITS = ["train","val","test"]
CLASSES = ["normal","anomaly"]

# B1 config
B1_WIN_S = 1.0
B1_HOP_S = 0.5
# B2 config
B2_WIN_S = 2.0
B2_HOP_S = 0.5  # 75% overlap on 2.0 s -> 0.5 s hop

def load_mono_16k(fp):
    y, sr = sf.read(fp, dtype="float32", always_2d=False)
    if y.ndim > 1: y = y[:,0]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def base_prefix(name:str):
    # strip any existing "__segXXXX"
    if "__seg" in name:
        name = name.split("__seg")[0]
    return name.rsplit(".",1)[0]

def write_segments(y, out_dir:Path, prefix:str, win_s:float, hop_s:float):
    out_dir.mkdir(parents=True, exist_ok=True)
    win = int(SR*win_s)
    hop = int(SR*hop_s)
    if len(y) < win:
        return 0
    n = 0
    for start in range(0, len(y)-win+1, hop):
        seg = y[start:start+win]
        n += 1
        out_fp = out_dir / f"{prefix}__seg{n:04d}.wav"
        sf.write(out_fp.as_posix(), seg, SR, subtype="PCM_16")
    return n

def main():
    if not RAW.exists():
        raise SystemExit(f"[ERR] Missing raw folder: {RAW}")

    stats = {"B1": {}, "B2": {}}

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = RAW / split / cls
            if not in_dir.exists():
                print(f"[WARN] skip missing: {in_dir}")
                continue

            # prepare out dirs
            o1 = OUT_B1 / split / cls
            o2 = OUT_B2 / split / cls
            o1.mkdir(parents=True, exist_ok=True)
            o2.mkdir(parents=True, exist_ok=True)

            cnt_b1 = 0
            cnt_b2 = 0
            files = sorted(in_dir.glob("*.wav"))
            for fp in files:
                y = load_mono_16k(fp)
                pref = base_prefix(fp.name)
                cnt_b1 += write_segments(y, o1, pref, B1_WIN_S, B1_HOP_S)
                cnt_b2 += write_segments(y, o2, pref, B2_WIN_S, B2_HOP_S)

            stats["B1"].setdefault(split, {})[cls] = cnt_b1
            stats["B2"].setdefault(split, {})[cls] = cnt_b2
            print(f"[OK] {split}/{cls}: B1={cnt_b1} segs, B2={cnt_b2} segs")

    # save a small manifest
    man = {
        "sr": SR,
        "inputs": str(RAW),
        "outputs": {"B1": str(OUT_B1), "B2": str(OUT_B2)},
        "b1": {"win_s": B1_WIN_S, "hop_s": B1_HOP_S},
        "b2": {"win_s": B2_WIN_S, "hop_s": B2_HOP_S},
        "counts": stats,
    }
    (HERE / "artifacts").mkdir(parents=True, exist_ok=True)
    manifest = HERE / "artifacts" / "resegment_manifest_B1_B2.json"
    manifest.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote manifest: {manifest}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
