"""
prepare_dataset.py

Scans the data/ folder and builds train/val/test manifest CSVs
(filepath,label) that train_kws_dscnn.py reads from.

Expected folder layout (see README.md for where to get each piece):

  data/
    keyword/              <- your recorded clips (record_keyword.py)
    unknown/              <- other-word clips, e.g. from Speech Commands
    silence/               <- pure silence / near-silence clips
    _background_noise_/    <- long _background_noise_ clips (e.g. from MUSAN), used only
                              for augmentation during training, not as its
                              own class

Usage:
    python prepare_dataset.py
"""

import csv
import os
import random
from pathlib import Path
DATA_ROOT = "data"
LABELS = ["keyword", "unknown", "silence"]  # index order = model output order
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 1337


def list_wavs(folder: str) -> list[str]:
    folder_path = Path(folder)

    if not folder_path.is_dir():
        return []

    return sorted(
        str(wav_path)
        for wav_path in folder_path.rglob("*.wav")
        if wav_path.is_file()
    )

def build_manifest():
    rows = []
    for label_idx, label in enumerate(LABELS):
        folder = os.path.join(DATA_ROOT, label)
        files = list_wavs(DATA_ROOT)
        print(f"  {label:10s}: {len(files)} clips found in {folder}/")
        for f in files:
            rows.append((f, label_idx, label))

    if not rows:
        raise RuntimeError(
            "No .wav files found under data/. Populate data/keyword, "
            "data/unknown and data/silence first (see README.md)."
        )

    random.Random(SEED).shuffle(rows)

    n = len(rows)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)

    val_rows = rows[:n_val]
    test_rows = rows[n_val:n_val + n_test]
    train_rows = rows[n_val + n_test:]

    return train_rows, val_rows, test_rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label_idx", "label_name"])
        writer.writerows(rows)
    print(f"  wrote {len(rows):5d} rows -> {path}")


def main():
    print("Scanning data/ ...")
    train_rows, val_rows, test_rows = build_manifest()

    print("\nWriting manifests:")
    write_csv(train_rows, "manifests/manifest_train.csv")
    write_csv(val_rows, "manifests/manifest_val.csv")
    write_csv(test_rows, "manifests/manifest_test.csv")

    print("\nClass balance check (train set):")
    for label_idx, label in enumerate(LABELS):
        count = sum(1 for _, li, _ in train_rows if li == label_idx)
        print(f"  {label:10s}: {count}")
    print(
        "\nIf 'unknown' or 'silence' massively outnumber 'keyword', consider "
        "capping how many unknown/silence files you draw in, so the model "
        "isn't trivially rewarded for always predicting the majority class."
    )


if __name__ == "__main__":
    main()
