
#!/usr/bin/env python3
"""
UOttawa -> WAV converter (v2) with strict label mapping.

- Reads CSV/XLSX/MAT (mic column) and writes mono 16 kHz WAVs.
- Labels:
    normal = last numeric token in filename == 0
    fault  = last numeric token in filename in {1,2}
  (e.g., H-1-0 -> normal; I-2-1 -> fault)
- If no numeric token is found, falls back to keywords: "healthy"/"normal" -> normal, else fault.
"""

import argparse, sys
from pathlib import Path

def _imports():
    try:
        import numpy as np
        import pandas as pd
        from scipy.signal import resample_poly
        from scipy.io.wavfile import write as wavwrite
        from scipy.io import loadmat
    except Exception:
        print("Please install: pip install numpy pandas scipy", file=sys.stderr); raise
    return np, pd, resample_poly, wavwrite, loadmat

def read_table(path, pd, loadmat):
    p = Path(path); suf = p.suffix.lower()
    if suf in (".csv",".txt"):
        try: df = pd.read_csv(p, header=None)
        except Exception: df = pd.read_csv(p, header=0)
        return df.values
    if suf in (".xlsx",".xls"):
        try: df = pd.read_excel(p, header=None)
        except Exception: df = pd.read_excel(p, header=0)
        return df.values
    if suf == ".mat":
        mat = loadmat(p, squeeze_me=True)
        # pick first 2D array with >=2 cols
        import numpy as np
        for k,v in mat.items():
            if k.startswith("__"): continue
            if isinstance(v, np.ndarray) and v.ndim==2 and v.shape[1] >= 2:
                return v
        raise ValueError("No 2D array with >=2 columns found in MAT file.")
    raise ValueError(f"Unsupported file type: {suf}")

def strict_label_from_name(name:str)->str:
    import re
    base = name.lower()
    toks = re.split(r'[^0-9a-z]+', base)  # split on non-alnum
    toks = [t for t in toks if t]
    # last numeric token
    last_num = None
    for t in reversed(toks):
        if t.isdigit():
            last_num = int(t)
            break
    if last_num is not None:
        return "normal" if last_num == 0 else "fault"
    # fallback keywords
    if ("healthy" in base) or ("normal" in base):
        return "normal"
    return "fault"

def peak_normalize(x, peak_db=-12.0):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0: return x
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12: return x
    target = 10.0**(peak_db/20.0)  # -12 dBFS ~ 0.251
    return x * (target/peak)

def resample_array(x, src_sr, dst_sr):
    from fractions import Fraction
    from scipy.signal import resample_poly
    if src_sr == dst_sr: return x
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)

def float_to_int16(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def process_file(path, out_root, mic_col_index=1, src_sr=42000, dst_sr=16000, peak_db=-12.0, verbose=True):
    np, pd, resample_poly, wavwrite, loadmat = _imports()
    data = read_table(path, pd, loadmat)
    if mic_col_index >= data.shape[1]:
        raise ValueError(f"{path}: mic_col_index={mic_col_index} out of range; cols={data.shape[1]}")
    mic = data[:, mic_col_index].astype("float32")
    # scale if looks like int range
    if np.max(np.abs(mic)) > 10:
        mic = mic / (np.max(np.abs(mic)) + 1e-9)
    mic = resample_array(mic, src_sr, dst_sr)
    mic = peak_normalize(mic, peak_db)
    y16 = float_to_int16(mic)
    label = strict_label_from_name(Path(path).stem)
    out_dir = Path(out_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / (Path(path).stem + f"_mic_{dst_sr}Hz.wav")
    wavwrite(out_wav.as_posix(), dst_sr, y16)
    if verbose:
        print("Wrote", out_wav)
    return out_wav, label

def main():
    ap = argparse.ArgumentParser(description="UOttawa CSV/XLSX/MAT -> WAV (strict label mapping)")
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder")
    ap.add_argument("--mic-col", type=int, default=1, help="0-based mic column index (default 1 = 2nd col)")
    ap.add_argument("--src-sr", type=int, default=42000, help="Source sample rate (default 42000)")
    ap.add_argument("--dst-sr", type=int, default=16000, help="Target sample rate (default 16000)")
    ap.add_argument("--peak-db", type=float, default=-12.0, help="Peak normalize to dBFS (default -12)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    files = []
    for suf in ("*.csv","*.txt","*.xlsx","*.xls","*.mat"):
        files += list(in_dir.rglob(suf))
    if not files:
        print("No CSV/XLSX/MAT files found in", in_dir, file=sys.stderr); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_norm = n_fault = 0
    for f in files:
        try:
            _, lab = process_file(f, out_dir, args.mic_col, args.src_sr, args.dst_sr, args.peak_db, verbose=False)
            if lab=="normal": n_norm += 1
            else: n_fault += 1
        except Exception as e:
            print("[skip]", f, "->", e, file=sys.stderr)
    print(f"Done. normal={n_norm}, fault={n_fault}, total={n_norm+n_fault}")

if __name__ == "__main__":
    main()
