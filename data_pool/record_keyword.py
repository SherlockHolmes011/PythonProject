"""
record_keyword.py

Records N one-second clips of your custom keyword straight from the
microphone and saves them as WAV files in data/keyword/.

Usage:
    python record_keyword.py --num_clips 200 --keyword mykeyword

Tips while recording (see the guide for the full explanation):
  - Vary your distance from the mic and your tone slightly between clips.
  - Leave a beat of silence before/after speaking; this script auto-pads.
  - If possible, use the same microphone hardware you'll deploy with.
"""

import argparse
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from features import SAMPLE_RATE, CLIP_DURATION_S


def record_one_clip(duration_s=CLIP_DURATION_S, sample_rate=SAMPLE_RATE):
    print("  Recording in: 1...")
    time.sleep(0.4)
    print("        ...GO!")
    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clips", type=int, default=200)
    parser.add_argument("--keyword", type=str, default="keyword")
    parser.add_argument("--out_dir", type=str, default="data/keyword")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(args.out_dir) if f.endswith(".wav")])

    print(f"Recording {args.num_clips} clips of '{args.keyword}'.")
    print("Press Ctrl+C at any time to stop early -- clips already saved are kept.\n")

    for i in range(existing, existing + args.num_clips):
        input(f"[{i + 1}/{existing + args.num_clips}] Press Enter, then say '{args.keyword}'...")
        audio = record_one_clip()
        peak = np.max(np.abs(audio))
        if peak < 0.02:
            print("  ! That sounded very quiet / silent -- re-recording this one.")
            audio = record_one_clip()

        out_path = os.path.join(args.out_dir, f"{args.keyword}_{i:04d}.wav")
        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"  saved -> {out_path}\n")

    print("Done. Now run prepare_dataset.py to build the training manifest.")


if __name__ == "__main__":
    main()
