"""
sweep_mel_bins.py

Trains and evaluates the DS-CNN model at several different mel-bin
counts, so you can pick the point that balances accuracy against RAM
instead of guessing. For each candidate N_MEL_BINS, this:

  1. Re-extracts features at that resolution (using librosa.feature.
     melspectrogram directly, independent of the fixed N_MEL_BINS in
     features.py, so nothing else in the project needs to change to
     run this sweep).
  2. Trains a model with that input width.
  3. Reports validation accuracy, quantized flash size, and estimated
     activation RAM (the same proxy measurement used for the MFCC vs
     log-mel comparison).

IMPORTANT: this reads manifest_train.csv / manifest_val.csv -- the
real ones your project produces once you've recorded your keyword and
populated data/unknown, data/silence, data/_background_noise_. Without
those, there is no meaningful "accuracy" to report; this script will
tell you exactly that instead of faking a number.

Usage (once you have real data):
    python sweep_mel_bins.py --candidates 10,13,16,20,26,32 --epochs 20
"""

import argparse
import csv
import os

import numpy as np
import librosa
import tensorflow as tf

from features import (
    SAMPLE_RATE,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    NUM_FRAMES,
    load_and_fix_length,
    add_background_noise,
    time_shift,
    volume_jitter,
)
from train_kws_dscnn import build_ds_cnn


def extract_log_mel_variable(audio, n_mel_bins):
    """Same as features.extract_log_mel, but with n_mel_bins as a
    parameter instead of the fixed constant, so the sweep can try
    several values without touching features.py."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=n_mel_bins,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    log_mel = np.log(mel_spec + 1e-6).T
    if log_mel.shape[0] > NUM_FRAMES:
        log_mel = log_mel[:NUM_FRAMES, :]
    elif log_mel.shape[0] < NUM_FRAMES:
        pad = NUM_FRAMES - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)))
    return log_mel[..., np.newaxis].astype(np.float32)


def read_manifest(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["filepath"], int(r["label_idx"])))
    return rows


def load_noise_clips(folder="data/_background_noise_"):
    import glob
    clips = []
    for p in glob.glob(os.path.join(folder, "*.wav")):
        try:
            clips.append(load_and_fix_length(p, clip_samples=SAMPLE_RATE * 5))
        except Exception:
            pass
    return clips


def build_dataset(manifest_path, n_mel_bins, noise_clips, batch_size, augment):
    rows = read_manifest(manifest_path)

    def gen():
        for filepath, label_idx in rows:
            audio = load_and_fix_length(filepath)
            if augment:
                audio = time_shift(audio)
                audio = volume_jitter(audio)
                audio = add_background_noise(audio, noise_clips)
            yield extract_log_mel_variable(audio, n_mel_bins), label_idx

    shape = (NUM_FRAMES, n_mel_bins, 1)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if augment:
        ds = ds.shuffle(buffer_size=min(len(rows), 2000), seed=1337)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(rows)


def measure_ram_flash(model, input_shape, n_samples=8):
    def rep_gen():
        for _ in range(n_samples):
            yield [np.random.randn(1, *input_shape).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    activation_bytes = sum(
        int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        for t in interp.get_tensor_details()
    )
    return len(tflite_model), activation_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="10,13,16,20,26,32")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not (os.path.exists("manifest_train.csv") and os.path.exists("manifest_val.csv")):
        print("No manifest_train.csv / manifest_val.csv found.")
        print("This sweep needs your real recorded keyword plus negative/_background_noise_")
        print("data (run prepare_dataset.py first) -- without it, any")
        print("'accuracy' number here would be meaningless, so this won't")
        print("fabricate one. Architecture-only stats (flash/RAM) don't need")
        print("real data and are safe to report; run this again with real")
        print("manifests for the accuracy column too.")
        return

    candidates = [int(x) for x in args.candidates.split(",")]
    noise_clips = load_noise_clips()

    results = []
    for n_mel_bins in candidates:
        print(f"\n=== N_MEL_BINS = {n_mel_bins} ===")
        train_ds, n_train = build_dataset(
            "manifest_train.csv", n_mel_bins, noise_clips, args.batch_size, augment=True
        )
        val_ds, n_val = build_dataset(
            "manifest_val.csv", n_mel_bins, noise_clips, args.batch_size, augment=False
        )

        model = build_ds_cnn(input_shape=(NUM_FRAMES, n_mel_bins, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6, restore_best_weights=True)],
        )
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)

        flash_bytes, activation_bytes = measure_ram_flash(
            model, (NUM_FRAMES, n_mel_bins, 1)
        )

        results.append((n_mel_bins, val_acc, flash_bytes, activation_bytes))
        print(f"val_accuracy={val_acc:.4f}  flash={flash_bytes/1024:.1f}KB  "
              f"activation~{activation_bytes/1024:.1f}KB")

    print("\n=== Summary ===")
    print(f"{'mel_bins':>8} | {'val_acc':>8} | {'flash_KB':>9} | {'activation_KB':>14}")
    for n_mel_bins, val_acc, flash_bytes, activation_bytes in results:
        print(f"{n_mel_bins:>8} | {val_acc:>8.4f} | {flash_bytes/1024:>9.1f} | "
              f"{activation_bytes/1024:>14.1f}")

    print("\nPick the smallest mel-bin count whose val_accuracy doesn't drop")
    print("meaningfully below your best result -- that's your best")
    print("accuracy-per-KB point, not necessarily the single highest accuracy.")


if __name__ == "__main__":
    main()
