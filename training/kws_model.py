"""
kws_model.py

Builds and trains the depthwise-separable CNN keyword-spotting model
described in the architecture walkthrough:

    input (49,10,1)
      -> Conv2D                (full conv, mixes all channels once)
      -> SeparableConv2D x3    (the cheap "DS-CNN" blocks, each downsamples)
      -> GlobalAveragePooling2D
      -> Dense + softmax       (3 classes: keyword / unknown / silence)

Usage:
    python train_kws_dscnn.py --epochs 30
"""

import argparse
import csv
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from mfcc_extract.features import (
    INPUT_SHAPE,
    load_and_fix_length,
    extract_mfcc,
    add_background_noise,
    time_shift,
    volume_jitter,
)

NUM_CLASSES = 3
LABELS = ["keyword", "unknown", "silence"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_ds_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    # Layer 1: one full Conv2D -- mixes across all input channels once,
    # before we switch to the cheap depthwise-separable blocks.
    x = layers.Conv2D(32, (10, 4), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Layers 2-4: depthwise-separable conv blocks. Keras's SeparableConv2D
    # implements exactly the depthwise-then-pointwise pattern discussed
    # earlier in one call. Each block also downsamples via MaxPooling2D.
    for filters in (64, 64, 64):
        x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    # Layer 5: collapse each channel to a single number instead of a large
    # flatten + dense, which is what keeps the parameter count tiny.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Layer 6: final classification head.
    outputs = layers.Dense(num_classes, activation="softmax", name="scores")(x)

    return models.Model(inputs, outputs, name="ds_cnn_kws")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def read_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["filepath"], int(r["label_idx"])))
    return rows


def load_noise_clips(folder="data/_background_noise_"):
    clips = []
    for path in glob.glob(os.path.join(folder, "*.wav")):
        try:
            clips.append(load_and_fix_length(path, clip_samples=16000 * 5))
        except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
            print(f"  skipping unreadable noise file {path}: {e}")
    print(f"  loaded {len(clips)} background noise clips")
    return clips


def make_example_generator(rows, noise_clips, augment):
    def gen():
        for filepath, label_idx in rows:
            audio = load_and_fix_length(filepath)
            if augment:
                audio = time_shift(audio)
                audio = volume_jitter(audio)
                audio = add_background_noise(audio, noise_clips)
            features = extract_mfcc(audio)
            yield features, label_idx

    return gen


def build_dataset(manifest_path, noise_clips, batch_size, augment, shuffle):
    rows = read_manifest(manifest_path)
    gen = make_example_generator(rows, noise_clips, augment)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(rows), 2000), seed=1337)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="saved_model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading background noise for augmentation...")
    noise_clips = load_noise_clips()

    print("\nBuilding data pipelines...")
    train_ds, n_train = build_dataset(
        "manifest_train.csv", noise_clips, args.batch_size, augment=True, shuffle=True
    )
    val_ds, n_val = build_dataset(
        "manifest_val.csv", noise_clips, args.batch_size, augment=False, shuffle=False
    )
    print(f"  train examples: {n_train}   val examples: {n_val}")

    print("\nBuilding model...")
    model = build_ds_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(args.out_dir, "final_model.keras")
    model.save(final_path)
    print(f"\nSaved trained model -> {final_path}")
    print("Next: run quantize_and_export.py to produce the int8 .tflite "
          "file and the C array for TFLite Micro deployment.")


if __name__ == "__main__":
    main()