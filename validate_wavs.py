"""
validate_wavs.py

Scans .wav files under data/ (or a specific subfolder) and tries to
open each with soundfile, the same library librosa.load uses under
the hood. Reports any file that fails, along with its size.

By default this only REPORTS bad files -- nothing is deleted unless
you pass --delete.

Usage (run from the PythonProject root):

    # just report, scan everything under data/
    python validate_wavs.py

    # report only, scan just the keyword folder
    python validate_wavs.py --folder keyword

    # actually delete bad files found in the keyword folder
    python validate_wavs.py --folder keyword --delete

    # delete bad files anywhere under data/
    python validate_wavs.py --delete
"""

import argparse
import os
import soundfile as sf

DATA_ROOT = "C:/Users/reddy/PycharmProjects/PythonProject/data"


def find_bad_files(scan_root):
    bad_files = []
    total = 0

    for dirpath, _, filenames in os.walk(scan_root):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            total += 1
            path = os.path.join(dirpath, fname)
            size = os.path.getsize(path)
            try:
                with sf.SoundFile(path) as f:
                    _ = f.frames  # touch it to force a real read check
            except Exception as e:
                bad_files.append((path, size, str(e)))

    return total, bad_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Subfolder under data/ to scan, e.g. 'keyword'. "
             "Omit to scan all of data/.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete bad files found (default: report only).",
    )
    args = parser.parse_args()

    scan_root = os.path.join(DATA_ROOT, args.folder) if args.folder else DATA_ROOT

    if not os.path.isdir(scan_root):
        print(f"Folder not found: {scan_root}")
        return

    total, bad_files = find_bad_files(scan_root)

    print(f"Checked {total} wav file(s) under {scan_root}")
    if not bad_files:
        print("All files opened successfully. Nothing to delete.")
        return

    print(f"\n{len(bad_files)} BAD file(s) found:\n")
    for path, size, err in bad_files:
        print(f"  {path}  ({size} bytes)")
        print(f"    -> {err}")

    if not args.delete:
        print(
            "\nReport-only mode -- nothing deleted. "
            "Re-run with --delete to remove these files."
        )
        return

    print(f"\nDeleting {len(bad_files)} bad file(s)...")
    deleted = 0
    for path, _, _ in bad_files:
        try:
            os.remove(path)
            deleted += 1
        except Exception as e:
            print(f"  FAILED to delete {path}: {e}")

    print(f"Deleted {deleted}/{len(bad_files)} bad file(s).")
    print(
        "\nRemember: your manifest CSVs still reference these paths. "
        "Re-run prepare_dataset.py to regenerate manifests before training again."
    )


if __name__ == "__main__":
    main()