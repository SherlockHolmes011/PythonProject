"""
validate_wavs.py

Scans every .wav file under data/ and tries to open it with soundfile,
the same library librosa.load uses under the hood. Reports any file
that fails, along with its size, so you can spot 0-byte / truncated
recordings before training crashes on them mid-run.

Usage (run from the PythonProject root):
    python validate_wavs.py
"""

import os
import soundfile as sf

DATA_ROOT = "C:/Users/reddy/PycharmProjects/PythonProject/data"

bad_files = []
total = 0

for dirpath, _, filenames in os.walk(DATA_ROOT):
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

print(f"Checked {total} wav files.")
if bad_files:
    print(f"\n{len(bad_files)} BAD file(s) found:\n")
    for path, size, err in bad_files:
        print(f"  {path}  ({size} bytes)")
        print(f"    -> {err}")
else:
    print("All files opened successfully.")