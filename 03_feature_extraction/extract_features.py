from pathlib import Path
import csv, math
import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SR = 16000
NFFT = 4096  # long FFT for better band resolution on 2 s clips

def load_audio(p: Path):
    y, sr = sf.read(p, dtype="int16", always_2d=False)
    if sr != SR:
        raise RuntimeError(f"Unexpected SR {sr} in {p}")
    y = y.astype(np.float32) / 32767.0
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y

def features_from_signal(y: np.ndarray, sr: int = SR):
    # time-domain
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    zcr = float(((y[:-1] * y[1:]) < 0).mean())
    mean = float(np.mean(y))
    std = float(np.std(y) + 1e-12)
    kurt = float(np.mean(((y - mean) / std) ** 4))  # Pearson kurtosis
    crest = float((np.max(np.abs(y)) + 1e-12) / (rms + 1e-12))

    # spectrum (whole-segment periodogram)
    n = len(y)
    w = np.hanning(n)
    Y = np.fft.rfft(y * w, n=NFFT)
    mag = np.abs(Y) + 1e-12
    psd = (mag ** 2)

    freqs = np.fft.rfftfreq(NFFT, d=1.0 / sr)
    psd_sum = np.sum(psd) + 1e-20

    # spectral centroid & bandwidth
    centroid = float(np.sum(freqs * psd) / psd_sum)
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))

    # spectral flatness (geometric / arithmetic mean)
    flatness = float(np.exp(np.mean(np.log(psd))) / (np.mean(psd) + 1e-12))

    # bandpowers (0-1k, 1-3k, 3-8k)
    def band_power(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        return float(np.sum(psd[idx]) / psd_sum)

    bp_0_1k = band_power(0, 1000)
    bp_1_3k = band_power(1000, 3000)
    bp_3_8k = band_power(3000, 8000)
    hf_ratio = float((bp_3_8k) / (bp_0_1k + bp_1_3k + 1e-12))

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid_hz": centroid,
        "bandwidth_hz": bw,
        "flatness": flatness,
        "kurtosis": kurt,
        "crest": crest,
        "bp_0_1k": bp_0_1k,
        "bp_1_3k": bp_1_3k,
        "bp_3_8k": bp_3_8k,
        "hf_ratio": hf_ratio,
    }

def process_split(in_csv: Path, out_csv: Path):
    rows = []
    with open(in_csv, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    field_base = ["filepath","class","orig_file","segment_id","duration_s"]
    feat_names = ["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest",
                  "bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_base + feat_names)
        w.writeheader()
        for row in tqdm(rows, desc=f"feat:{in_csv.stem}"):
            p = (ROOT / row["filepath"]).resolve()
            y = load_audio(p)
            feats = features_from_signal(y, SR)
            out = {k: row[k] for k in field_base}
            out.update(feats)
            w.writerow(out)
    return len(rows)

def main():
    man = ROOT / "manifests"
    in_tr = man / "split_train.csv"
    in_va = man / "split_val.csv"
    in_te = man / "split_test.csv"
    out_tr = man / "features_train.csv"
    out_va = man / "features_val.csv"
    out_te = man / "features_test.csv"

    ntr = process_split(in_tr, out_tr)
    nva = process_split(in_va, out_va)
    nte = process_split(in_te, out_te)

    print("\\n== Features summary ==")
    print(f"train rows: {ntr}")
    print(f"val rows:   {nva}")
    print(f"test rows:  {nte}")
    print("Wrote:", out_tr, out_va, out_te)

if __name__ == "__main__":
    main()
