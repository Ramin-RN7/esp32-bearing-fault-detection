from pathlib import Path
import pandas as pd, numpy as np

ROOT=Path(__file__).resolve().parent
MAN=ROOT/"manifests"
FEATS=["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest","bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]

def load(name):
    df=pd.read_csv(MAN/f"features_{name}.csv")
    # force numeric, coerce bad to NaN
    for c in FEATS:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    # replace inf with NaN
    for c in FEATS:
        m=np.isfinite(df[c].to_numpy())
        df.loc[~m, c]=np.nan
    return df

tr=load("train"); va=load("val"); te=load("test")

# report NaNs
def report(df, name):
    n_row=len(df)
    n_nan=int(df[FEATS].isna().any(axis=1).sum())
    print(f"{name}: rows={n_row}, rows_with_NaN={n_nan}")
    if n_nan:
        bad=df[df[FEATS].isna().any(axis=1)][["filepath","class","orig_file","segment_id"]]
        bad.to_csv(MAN/f"nan_{name}.csv", index=False)
        print(f"  -> listed in manifests/nan_{name}.csv")

report(tr,"train"); report(va,"val"); report(te,"test")

# impute using TRAIN medians only (avoid leakage)
med=tr[FEATS].median(numeric_only=True)

for df,name in [(tr,"train_clean"),(va,"val_clean"),(te,"test_clean")]:
    d=df.copy()
    d[FEATS]=d[FEATS].fillna(med)
    # verify
    assert np.isfinite(d[FEATS].to_numpy()).all(), f"{name} still has NaN/Inf"
    d.to_csv(MAN/f"features_{name}.csv", index=False)
    print(f"wrote: manifests/features_{name}.csv")
