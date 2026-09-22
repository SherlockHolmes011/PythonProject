"""
quantize_and_export.py

Takes the trained Keras model and:
  1. Runs full integer (int8) post-training quantization.
  2. Saves the resulting .tflite file.
  3. Exports it as a C byte array (.cc/.h) ready to compile into
     TFLite Micro firmware on the ESP32.

Usage:
    python quantize_and_export.py --model saved_model/final_model.keras
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, "C:/Users/reddy/PycharmProjects/PythonProject")

from log_mel_Extract.features import load_and_fix_length, extract_log_mel


def representative_dataset_gen(manifest_path, num_samples=300):
    """The converter needs real example inputs to calibrate int8 ranges.
    We draw a few hundred samples from the training set for this."""
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = [r["filepath"] for r in reader]

    rng = np.random.default_rng(1337)
    chosen = rng.choice(rows, size=min(num_samples, len(rows)), replace=False)

    def gen():
        for path in chosen:
            audio = load_and_fix_length(path)
            features = extract_log_mel(audio)
            yield [features[np.newaxis, ...]]  # add batch dim

    return gen


def quantize_model(keras_model_path, manifest_path, out_tflite_path):
    model = tf.keras.models.load_model(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(manifest_path)
    # Force full int8: every op, plus the model's input/output tensors,
    # is quantized -- this is what TFLite Micro on the ESP32 expects.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(out_tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Quantized model saved -> {out_tflite_path}  ({size_kb:.1f} KB)")
    return tflite_model


def tflite_to_c_array(tflite_bytes, var_name, out_cc_path, out_h_path):
    """Equivalent of `xxd -i`, but naming things the way TFLite Micro
    examples expect (a byte array + length constant)."""
    with open(out_cc_path, "w") as f:
        f.write(f'#include "{os.path.basename(out_h_path)}"\n\n')
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"  {line},\n")
        f.write("};\n")
        f.write(f"const int {var_name}_len = {len(tflite_bytes)};\n")

    guard = os.path.basename(out_h_path).upper().replace(".", "_")
    with open(out_h_path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("extern const unsigned char " + var_name + "[];\n")
        f.write("extern const int " + var_name + "_len;\n\n")
        f.write(f"#endif  // {guard}\n")

    print(f"C array exported -> {out_cc_path}, {out_h_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="saved_model/final_model.keras")
    parser.add_argument("--manifest", type=str, default="manifest_train.csv")
    parser.add_argument("--out_dir", type=str, default="exported")
    parser.add_argument("--var_name", type=str, default="g_kws_model_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tflite_path = os.path.join(args.out_dir, "kws_model_int8.tflite")

    tflite_bytes = quantize_model(args.model, args.manifest, tflite_path)

    cc_path = os.path.join(args.out_dir, "model_data.cc")
    h_path = os.path.join(args.out_dir, "model_data.h")
    tflite_to_c_array(tflite_bytes, args.var_name, cc_path, h_path)

    print(
        "\nDone. Copy model_data.cc / .h into your ESP32 firmware "
        "project alongside the TFLite Micro interpreter, and point it at "
        f"`{args.var_name}` / `{args.var_name}_len` when loading the model."
    )


if __name__ == "__main__":
    main()
