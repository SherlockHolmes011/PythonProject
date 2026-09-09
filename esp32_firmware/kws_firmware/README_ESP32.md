# ESP32 KWS Firmware

This is the deployment half of the project: the always-on inference loop,
threshold+smoothing decision, and cloud streaming handoff, built to run
your trained model from `quantize_and_export.py`.

## Important, read first

I could not compile or run this on real hardware from here -- I don't
have an ESP32 or the Arduino toolchain available in this environment.
Every file is written carefully and the logic mirrors well-established
patterns (TFLite Micro's own examples use the same interpreter setup,
I2S is standard ESP32 mic capture), but **treat this as a strong
starting scaffold to build and debug on your actual board, not
guaranteed-correct code.** Expect to fix small issues (library API
version differences, pin numbers for your specific dev board, timing)
the way you would with any first hardware bring-up.

## 1. Hardware wiring

This firmware supports two mic types, switchable with one line in
`config.h` (`MIC_TYPE`). Wire whichever one you're currently using.

### Option A: INMP441 I2S digital mic (`MIC_TYPE_I2S`)

| INMP441 pin | ESP32 pin (default in config.h) |
|---|---|
| VDD | 3.3V |
| GND | GND |
| WS (L/R clock) | GPIO 15 |
| SCK (bit clock) | GPIO 14 |
| SD (data out) | GPIO 32 |
| L/R | GND (selects left channel) |

### Option B: 2-pin electret condenser capsule (`MIC_TYPE_ANALOG`)

Unlike the I2S mic, this is NOT plug-and-play -- the bare capsule needs
an external bias/coupling circuit before it produces anything usable:

```
3.3V ----[4.7k bias resistor]----+---- capsule pin 1 (signal)
                                  |
                                  +----[2.2uF coupling cap]---- GPIO 34
GND  ------------------------------ capsule pin 2
```

- The bias resistor is what powers the capsule at all -- without it,
  you'll read pure noise or nothing.
- The coupling capacitor strips off the DC bias voltage, leaving just
  the audio signal for the ADC to read.
- GPIO 34 was chosen because it's an ADC1 pin -- ADC2 pins conflict
  with WiFi and become unusable the moment WiFi connects, which would
  break the cloud streaming step entirely.
- For noticeably better signal quality, add a small op-amp gain stage
  between the capacitor and the ADC pin (see the chat discussion for
  why -- the raw capsule signal is only millivolts, and the ESP32's
  ADC alone is fairly noisy). Not strictly required to get it running,
  but worth it if detection accuracy looks poor.

**Calibration step, required before this mic will work correctly:**
`ANALOG_MIC_ADC_MIDPOINT` in `config.h` is currently a default guess
(2048). Your actual circuit's silence baseline will differ. To measure
it: flash the firmware, sit in silence, and add a temporary line
printing the raw ADC value (`adc1_get_raw(ANALOG_MIC_ADC_CHANNEL)`) to
Serial every second or so. Average several readings while silent, and
put that number into `ANALOG_MIC_ADC_MIDPOINT`. Skipping this means
every sample is offset by a constant error, which will show up as
either clipping or a flattened, quiet-sounding signal.

## 2. Switching between the two mic types later

Change exactly one line in `config.h`:
```cpp
#define MIC_TYPE   MIC_TYPE_I2S       // or MIC_TYPE_ANALOG
```
Re-wire the hardware per the matching table above, re-flash, and
nothing else in the firmware needs to change -- both drivers
(`audio_provider_i2s.h` / `audio_provider_analog.h`) expose the exact
same four functions, which is why `kws_firmware.ino` never needs
editing when you switch.

**Worth knowing when comparing the two**: the analog driver costs more
CPU than the I2S one. I2S offloads sample capture to dedicated DMA
hardware essentially for free; the analog driver has no equivalent and
uses a 16kHz hardware timer actively reading the ADC each time instead.
If you're tracking your idle-CPU budget closely (see the earlier RAM
and CPU discussions), expect a real, measurable difference between the
two -- not just theoretical.

## 3. Required Arduino libraries

Install via Arduino IDE -> Tools -> Manage Libraries:
- **ESP32 board support** (esp32 by Espressif Systems) -- if not already installed
- **TensorFlowLite_ESP32** (search "TensorFlow Lite ESP32") -- provides
  the TFLite Micro headers this firmware includes. If that exact
  library name isn't available in your IDE version, search for
  "TensorFlow Lite Micro" alternatives -- the API used here
  (`tflite::GetModel`, `MicroInterpreter`, `AllOpsResolver`) is the
  standard TFLite Micro C++ API and should match closely across ports.

No other libraries needed -- WiFi and I2S come from the ESP32 board
package itself.

## 4. Before you flash: replace the placeholder model

`model_data.h` / `model_data.cc` in this folder are dummy placeholders
so the project compiles as a template. Once you've run
`quantize_and_export.py` on your trained model:

1. Delete `model_data.h` and `model_data.cc` from this folder.
2. Copy `exported/kws_model_data.h` -> `model_data.h`
3. Copy `exported/kws_model_data.cc` -> `model_data.cc`

The symbol names (`g_kws_model_data`, `g_kws_model_data_len`) already
match, so no other code changes are needed.

## 5. Fill in your WiFi and server details

Edit `config.h`:
```
#define WIFI_SSID       "..."
#define WIFI_PASSWORD   "..."
#define ASR_HOST        "..."
#define ASR_PORT        443
```

## 6. Flashing

1. Open `kws_firmware.ino` in Arduino IDE (open this specific file --
   the whole `kws_firmware/` folder is the sketch, Arduino will pick up
   the other `.h`/`.cpp` files automatically).
2. Tools -> Board -> select your ESP32 board variant.
3. Tools -> Port -> select the correct serial port.
4. Upload.
5. Open Serial Monitor at 115200 baud -- you should see the boot log,
   the model's reported arena usage, and then a `confidence=...` line
   printed roughly every 200ms once the pre-roll buffer fills.

## 7. Validating feature parity (do this before trusting any results)

The single biggest risk in this whole deployment step is a mismatch
between the MFCC features computed in Python (`features.py`, training
time) and the MFCC features computed in C++ (`mfcc.h`, on-device). If
these drift even slightly, the model's confidence scores on real
hardware won't match what you saw during training/evaluation.

Recommended check, using your Serial Monitor debugging habit:
1. Temporarily add a `Serial.print` loop after `compute_mfcc()` in
   `run_inference()` that prints the raw float feature values for one
   clip.
2. Record the same clip's raw audio simultaneously (or use a fixed
   test tone).
3. Run that same audio through `features.py`'s `wav_to_features()` in
   Python and print the values.
4. Compare -- they won't be bit-identical (floating point differs
   slightly across platforms), but they should be close in shape and
   magnitude. Large discrepancies point to a bug in `mfcc.h` (window
   function, mel filterbank edges, or FFT scaling are the usual
   suspects).

## 8. Tuning once it's running

- If `AllocateTensors()` fails, increase `TENSOR_ARENA_SIZE` in
  `config.h` -- the printed `arena used` value on a successful boot
  tells you how much headroom you actually have.
- If you get frequent false triggers, raise `DETECTION_THRESHOLD` or
  `CONSECUTIVE_FRAMES_NEEDED` in `config.h` (see the threshold +
  smoothing design discussion for the tradeoffs).
- The streaming step currently sends raw 16-bit PCM for simplicity and
  correctness-first debugging. Once that's verified working end to
  end, swap in Opus or mu-law compression to cut bandwidth and improve
  the latency metric, as discussed earlier.
