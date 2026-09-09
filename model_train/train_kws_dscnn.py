"""
train_kws_dscnn.py

Builds and trains the final merged DS-CNN keyword-spotting model --
combining the depthwise/pointwise block structure and ReLU6 activations
from the reference model.py with this project's dropout regularization
and audio-shaped first-layer kernel:

    input (49, N_MEL_BINS, 1)                  log-mel spectrogram
      -> Conv2D(16, 5x4, stride 2)              audio-shaped first layer
      -> [DepthwiseConv2D -> BN -> ReLU6 -> Conv2D(1x1) -> BN -> ReLU6] x3
      -> GlobalAveragePooling2D -> Dropout(0.3)
      -> Dense + softmax                        3 classes

Usage:
    python train_kws_dscnn.py --epochs 30
"""

import argparse
import csv
import glob
import os

import tensorflow as tf
from tensorflow.keras import layers, models
import sys
sys.path.insert(0, "C:/Users/reddy/PycharmProjects/PythonProject")
from log_mel_Extract.features import (
    INPUT_SHAPE,
    load_and_fix_length,
    extract_log_mel,
    add_background_noise,
    time_shift,
    volume_jitter,
)

NUM_CLASSES = 3
LABELS = ["keyword", "unknown", "silence"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def depthwise_block(x: tf.Tensor, filters: int, stride: int) -> tf.Tensor:
    """One MobileNet-style depthwise-separable block, expanded into its
    two explicit steps (rather than Keras's fused SeparableConv2D) so
    BatchNorm + ReLU6 are applied between the depthwise and pointwise
    convolutions, not just once at the end -- this gives the network
    more non-linearity and matches the published DS-CNN design.
    ReLU6 (bounded to [0,6]) is used instead of plain ReLU so int8
    post-training quantization has a fixed, predictable activation
    range to calibrate against.
    """
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(max_value=6.0)(x)


def build_ds_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name="log_mel_input")

    # Layer 1: one full Conv2D -- mixes across all input channels once,
    # before switching to the cheap depthwise-separable blocks. Kept
    # asymmetric (more time-axis reach than frequency-axis) since time
    # and frequency carry different kinds of structure in a spectrogram,
    # unlike a generic square image kernel.
    x = layers.Conv2D(16, (5, 4), strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    # Layers 2-4: three depthwise-separable blocks, widened versus the
    # reference two-block version for a bit more capacity -- affordable
    # here since the RAM budget has headroom for it.
    x = depthwise_block(x, 32, stride=1)
    x = depthwise_block(x, 48, stride=2)
    x = depthwise_block(x, 64, stride=1)

    # Collapse each channel to a single number instead of a large
    # flatten + dense, which is what keeps the parameter count tiny.
    x = layers.GlobalAveragePooling2D()(x)
    # Dropout guards against overfitting on a small custom-keyword dataset
    # -- the reference model has none, which is a real risk with only a
    # few hundred of your own recorded clips.
    x = layers.Dropout(0.3)(x)

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
            print(f"  skipping unreadable _background_noise_ file {path}: {e}")
    print(f"  loaded {len(clips)} background _background_noise_ clips")
    return clips


def make_example_generator(rows, noise_clips, augment):
    def gen():
        for filepath, label_idx in rows:
            audio = load_and_fix_length(filepath)
            if augment:
                audio = time_shift(audio)
                audio = volume_jitter(audio)
                audio = add_background_noise(audio, noise_clips)
            features = extract_log_mel(audio)
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

    print("Loading background _background_noise_ for augmentation...")
    noise_clips = load_noise_clips()

    print("\nBuilding data pipelines...")
    train_ds, n_train = build_dataset(
        "C:/Users/reddy/PycharmProjects/PythonProject/manifests/manifest_train.csv", noise_clips, args.batch_size, augment=True, shuffle=True
    )
    val_ds, n_val = build_dataset(
        "C:/Users/reddy/PycharmProjects/PythonProject/manifests/manifest_val.csv", noise_clips, args.batch_size, augment=False, shuffle=False
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
            monitor="val_accuracy", patience=15, restore_best_weights=True
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
