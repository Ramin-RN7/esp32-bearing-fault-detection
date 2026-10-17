# Original research records

These are unchanged copies from the original research workspace (`bearing-ml`, last modified 14 August 2025). They are the primary evidence for the data split, the ablation runs and the on-device phase. Only two files were added later: this README and the note in [`device/exports/A16_MLP/`](device/exports/A16_MLP/README.md).

| Folder | Contents |
|---|---|
| [`manifests/`](manifests/) | The exact split used by every run: `segments.csv`, `segments_split.csv` and `split_{train,val,test}.csv`. That is 42/9/9 recordings and 378/81/81 two-second windows. The folder also holds the baseline feature tables, the 0.5 s-hop variant (`ov75_win2.0_hop0.5/`, used by B2) and the B1/B2 re-segmentation manifest. |
| [`runs/`](runs/) | 108 folders: one for each of the 104 logged runs (93 configurations: A0–A44, B1/B2 and E3), plus three unlogged reruns and the first baseline bundle. Each holds `params.json`, `metrics.json` (B runs: `metrics_file.json`), a snapshot of the code that produced the run (`code.py`), confusion matrices and, for most runs, the fitted model. `early_baseline_20250813_1416/` is the first baseline bundle. |
| [`logs/`](logs/) | The running experiment log (`variants.csv`, plus its backup and queue). Also the cleaned tables cited in the thesis (`ablation_clean.csv`, top-10 tables), the appendix table and the run-integrity audit. |
| [`predictions/`](predictions/) | Per-window and per-recording test predictions of the recording-level aggregation runs (A8 GMM, A16 MLP and A31 MLP). |
| [`device/`](device/) | Files deployed to the M5Stack Core2 on 14 August 2025: the feature extractor, GMM runtime, timing and WAV-evaluation scripts, and the exported models (`exports/`). It also holds the device notes (`MODEL_INFO_A8_GMM.md`, `REPORT_DevicePhase.md`) and the copy commands. `earlier_versions/` holds the first device iteration. |
| [`firmware/`](firmware/) | Notes, scripts and a placeholder C module from rebuilding the lab's custom MicroPython v1.20 firmware with ESP-IDF 4.4 to add model-inference modules. The rebuilt image booted, but those modules were not completed; the device measurements ran on the lab's original build (v1.20.0-53). Also the module list of that build. The lab's firmware itself is not included. |

## Reading these records

- **Every metric in these records is a test-set metric**, computed on 81 windows from 9 recordings. No validation metrics were logged. Validation data were used only to set the GMM decision threshold inside each run.
- `logs/variants_backup_20250814_050536.csv` has one malformed row (line 81 has 14 fields instead of 15). It is kept as recorded; the active log `logs/variants.csv` is complete.
- Paths such as `C:\Users\ramin\Desktop\bearing-ml\...` refer to the original workspace. [`scripts/evaluate_protocols.py`](../scripts/evaluate_protocols.py) reads the published copies instead.
- **The on-device features don't match the model.** The device feature extractor computes 25 short-frame PSD band energies, but the exported A8 model expects the 25 desktop features. The device notes record this ("parity with PC not established"). On-device classification accuracy was never measured. See [known issues](../docs/KNOWN_ISSUES.md#1-desktop-and-device-features-have-different-meanings).
- **`device/exports/A16_MLP/` holds the A9 MLP, not A16.** The exporter writes to that folder name by default. A [note in the folder](device/exports/A16_MLP/README.md) explains; the original files are unchanged.
- **Not included:**
  - raw and segmented audio, which can be regenerated from [`01_uottawa_wavs/`](../01_uottawa_wavs/)
  - the Python virtual environment
  - third-party code
  - personal notes
