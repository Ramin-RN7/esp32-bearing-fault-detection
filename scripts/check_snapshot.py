#!/usr/bin/env python3
"""Inspect the thesis snapshot without importing ML or device libraries."""

import ast
import csv
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "01_uottawa_wavs",
    "02_data_segmentation",
    "03_feature_extraction",
    "04_model_training_ablation",
    "05_model_export",
)


def main():
    problems = []
    sources = sorted(
        path
        for directory in SOURCE_DIRS
        for path in (ROOT / directory).rglob("*.py")
    )
    if not sources:
        problems.append("No archived Python sources were found.")
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: {exc}")
    print(f"Archived Python sources checked: {len(sources)}")

    recordings = sorted((ROOT / "01_uottawa_wavs" / "wav").rglob("*.wav"))
    rates, durations = Counter(), []
    if not recordings:
        problems.append("No source WAV recordings were found.")
    for path in recordings:
        try:
            with wave.open(str(path), "rb") as audio:
                rate, frames = audio.getframerate(), audio.getnframes()
                if rate <= 0 or frames <= 0:
                    raise ValueError("Empty recording or invalid sample rate")
                expected = frames * audio.getnchannels() * audio.getsampwidth()
                if len(audio.readframes(frames)) != expected:
                    raise ValueError("Truncated PCM payload")
                rates[rate] += 1
                durations.append(frames / rate)
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: {exc}")
    print(f"Source WAV recordings checked: {len(recordings)}")
    print(f"Sample rates (Hz -> recordings): {dict(sorted(rates.items()))}")
    if durations:
        print(f"Duration range: {min(durations):.3f}-{max(durations):.3f} seconds")

    table = ROOT / "06_results_evaluation" / "best_of_table.csv"
    try:
        with table.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for family in ("MLP", "GMM"):
            values = [
                float(row["file_F1"])
                for row in rows
                if family in row.get("label", "")
            ]
            if not values:
                raise ValueError(f"No {family} file-level result found")
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Invalid {family} F1 value")
            print(f"Archived best {family} file-level F1: {max(values):.6f}")
    except (OSError, KeyError, ValueError) as exc:
        problems.append(f"Results table: {exc}")

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print("PASS: snapshot checks completed; training and hardware were not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
