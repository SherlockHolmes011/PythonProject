# Edge KWS + Cloud ASR Handoff -- Full Working Flow

This project trains the local "wake-up" model discussed in the design:
a tiny keyword-spotting (KWS) model that runs continuously on a
microcontroller, and only when it hears the custom keyword does the
device hand audio off to a cloud ASR server. This README explains how
every file connects, in the order things actually happen.

## 1. The two halves of the system, and where this code sits

```
   [ TRAINING -- this repo, runs on your PC in PyCharm ]
        |
        v
   kws_model_data.cc/.h  (quantized model as a C array)
        |
        v
   [ DEPLOYMENT -- ESP32 firmware, separate step ]
        record audio -> extract MFCC -> run this model ->
        if keyword detected -> stream audio to cloud ASR
```

Everything in this folder produces one artifact: a trained, quantized
`.tflite` model exported as a C array. Getting that artifact onto the
ESP32 and wiring it into the always-on listening loop is the next
phase (ask if you want that firmware code next).

## 2. Step-by-step flow, file by file

**Step 1 -- `record_keyword.py`**
Records N one-second WAV clips of your chosen keyword through your
mic and saves them to `data/keyword/`. This is your positive class.

**Step 2 -- gather negative data (manual download, see below)**
You need two more things Google didn't record for you:
- `data/unknown/` -- clips of *other* words, so the model learns what
  is speech-but-not-the-keyword, not just silence-vs-keyword.
- `data/silence/` -- short clips of near-silence / room tone.
- `data/_background_noise_/` -- longer noise clips (traffic, chatter,
  music) used only for augmentation, not as a training label of their
  own.

**Step 3 -- `prepare_dataset.py`**
Scans all of `data/`, splits everything into train/val/test, and
writes `manifest_train.csv` / `manifest_val.csv` / `manifest_test.csv`
(just filepath + label). Every later script reads from these
manifests instead of re-scanning folders, so the exact same split is
used consistently.

**Step 4 -- `features.py`** (imported by everything, not run directly)
Defines the *one* audio-to-features function used everywhere:
load a wav -> force it to exactly 1 second -> compute a 10-coefficient
MFCC over 49 time frames -> return a `(49, 10, 1)` array. It also
holds the three augmentation functions (background noise mixing, time
shift, volume jitter) applied only during training.

**Step 5 -- `train_kws_dscnn.py`**
This is the core script:
- Builds a `tf.data` pipeline that reads each manifest row, loads the
  wav, applies augmentation (train set only), and extracts MFCC
  features -- all lazily, so nothing huge is held in memory at once.
- Builds the DS-CNN model (architecture detail below).
- Trains with early stopping and learning-rate reduction on
  plateau, saving the best-validation-accuracy checkpoint.
- Outputs `saved_model/final_model.keras`.

**Step 6 -- `quantize_and_export.py`**
- Loads the trained float32 model.
- Runs full int8 post-training quantization, using a few hundred real
  training examples as the "representative dataset" the converter
  needs to calibrate int8 ranges correctly.
- Saves `exported/kws_model_int8.tflite`.
- Converts that file into a C byte array
  (`exported/kws_model_data.cc` / `.h`), the exact format TFLite
  Micro expects to compile into firmware.

## 3. The model architecture, and why each layer is there

```
Input (49, 10, 1)              <- one second of MFCC features
   -> Conv2D(32)               <- one full conv, mixes all channels once
   -> [SeparableConv2D + MaxPool] x3   <- the cheap "DS-CNN" blocks
   -> GlobalAveragePooling2D   <- collapses each channel to one number
   -> Dense(3) + softmax       <- keyword / unknown / silence
```

- **Why MFCC and not raw waveform**: turns 16000 raw samples/sec into
  a much smaller, more structured 49x10 grid that highlights the
  frequency patterns speech actually varies in -- easier for a tiny
  model to learn from.
- **Why one regular Conv2D first**: it needs to mix information across
  all input channels/frequencies once before switching to the cheap
  depthwise-separable layers, which don't do full channel mixing.
- **Why SeparableConv2D (the DS-CNN part)**: splits a normal
  convolution into a per-channel depthwise pass plus a cheap 1x1
  pointwise mix, cutting compute roughly 8-9x versus a normal conv of
  the same size -- this is what makes the model small enough to fit a
  microcontroller. Keras's `SeparableConv2D` layer implements exactly
  this depthwise-then-pointwise pattern in one call.
- **Why GlobalAveragePooling2D instead of Flatten+Dense**: avoids a
  huge weight matrix at the transition to the classifier head, which
  would dominate the model's size otherwise.
- **Why int8 quantization**: roughly 4x smaller than float32, faster
  integer math on a microcontroller without a strong FPU, and it's
  what determines the RAM "tensor arena" size TFLite Micro reports
  against your 256KB budget.

## 4. Technologies used, and why each one

| Technology | Role |
|---|---|
| **TensorFlow / Keras** | Defines and trains the model in Python |
| **librosa** | Loads audio and computes MFCC features |
| **sounddevice / soundfile** | Records and saves your keyword clips |
| **tf.data** | Streams training examples lazily instead of loading everything into RAM at once |
| **TFLite Converter** | Performs int8 post-training quantization |
| **TensorFlow Lite for Microcontrollers (next phase)** | The C++ runtime that actually executes the quantized model on the ESP32 |

## 5. Where to get the negative/background data

- **Speech Commands dataset** (for `data/unknown/`): ~105,000 clips of
  35 common words. Pick a handful of word-folders from the extracted
  archive and copy their wavs into `data/unknown/`.
- **MUSAN corpus** (for `data/_background_noise_/`): noise, music, and
  speech recordings, good for both augmentation and your false-positive
  stress test later.
- For `data/silence/`, a few minutes of your own room-tone recordings
  (using `record_keyword.py` with `--keyword silence`) works fine.

(Links to both were shared earlier in this conversation.)

## 6. Running it end to end

```bash
pip install -r requirements.txt

python record_keyword.py --keyword yourword --num_clips 250
# ... populate data/unknown, data/silence, data/_background_noise_ ...

python prepare_dataset.py
python train_kws_dscnn.py --epochs 30
python quantize_and_export.py
```

At the end, `exported/kws_model_data.cc` and `.h` are what get copied
into `esp32_firmware/kws_firmware/model_data.h` and `.cc`.

## 7. The ESP32 deployment firmware

`esp32_firmware/kws_firmware/` contains the second half of the project:
the always-on inference loop, the threshold + smoothing decision logic,
and the buffer/compress/stream handoff to the cloud ASR server. See
`esp32_firmware/kws_firmware/README_ESP32.md` for wiring, required
libraries, and flashing instructions. It ships with a placeholder model
so it compiles as a template before you've trained anything -- swap in
your real exported model per that README once training is done.
