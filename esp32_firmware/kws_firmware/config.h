// config.h
// All the tunable constants live here so you don't have to hunt through
// the rest of the firmware to adjust hardware pins or detection behavior.

#ifndef CONFIG_H
#define CONFIG_H

// ---- Microphone selection ----
// Change ONE line (MIC_TYPE below) to switch mic hardware -- both
// drivers expose the identical audio_provider_*() functions, so nothing
// else in the firmware needs to change either way.
#define MIC_TYPE_I2S      0   // digital I2S mic, e.g. INMP441
#define MIC_TYPE_ANALOG   1   // bare 2-pin electret capsule via ADC

#define MIC_TYPE   MIC_TYPE_ANALOG   // <- currently set for your electret
                                      //    capsule. Change to MIC_TYPE_I2S
                                      //    when you switch back -- that's
                                      //    the only line you need to touch.

// ---- I2S microphone pins (only used when MIC_TYPE == MIC_TYPE_I2S) ----
#define I2S_WS_PIN   15   // word select / LR clock
#define I2S_SD_PIN   32   // serial data in
#define I2S_SCK_PIN  14   // bit clock
#define I2S_PORT     I2S_NUM_0

// ---- Analog electret mic settings (only used when
// MIC_TYPE == MIC_TYPE_ANALOG) ----
#define ANALOG_MIC_PIN           34              // GPIO34 = ADC1_CHANNEL_6
#define ANALOG_MIC_ADC_CHANNEL   ADC1_CHANNEL_6  // must match the pin above
#define ANALOG_MIC_ADC_MIDPOINT  2048            // DEFAULT GUESS -- replace
                                                   // with your own measured
                                                   // silence baseline; see
                                                   // the calibration steps
                                                   // in README_ESP32.md

// ---- Audio settings -- MUST match features.py on the training side ----
#define SAMPLE_RATE     16000
#define CLIP_SECONDS    1.0f
#define CLIP_SAMPLES    (int)(SAMPLE_RATE * CLIP_SECONDS)
#define N_MELS          20     // must match features.py N_MEL_BINS
#define N_FFT           512
#define WIN_LENGTH      400    // 25ms
#define HOP_LENGTH      320    // 20ms -> yields 49 frames per 1s clip
#define N_FRAMES        49

// ---- Circular pre-roll buffer: how much audio we keep ready to send
// the instant a keyword is confirmed (covers the word + lead-in) ----
#define PREROLL_SECONDS   1.5f
#define PREROLL_SAMPLES   (int)(SAMPLE_RATE * PREROLL_SECONDS)

// ---- Inference cadence: how often we run the model on the last 1s
// of audio. Running every 200ms means worst-case detection delay is
// ~200ms plus the confirmation window below. ----
#define INFERENCE_INTERVAL_MS   200

// ---- Detection decision (see the threshold+smoothing design) ----
#define DETECTION_THRESHOLD        0.6f   // model confidence cutoff
#define CONSECUTIVE_FRAMES_NEEDED  3      // how many consecutive
                                           // inference passes must clear
                                           // the threshold before we trigger

// ---- Cooldown so one detection doesn't immediately re-trigger ----
#define COOLDOWN_MS   2000

// ---- Cloud ASR endpoint (fill in your own server) ----
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"
#define ASR_HOST        "your-asr-server.example.com"
#define ASR_PORT        443
#define ASR_PATH        "/stream"

// ---- Power management: "blink, don't stare" ----
// Deep sleep wipes RAM (would lose the circular buffer), so it's not an
// option here. Light sleep keeps RAM intact and, in tickless-idle mode,
// lets the I2S microphone's DMA keep capturing audio in the background
// while the CPU itself is powered down between useful work -- this is
// what actually lets the device "blink" instead of spinning at 100% CPU
// waiting for the next 200ms tick.
#define ENABLE_AUTOMATIC_LIGHT_SLEEP  1   // 0 disables, for debugging
#define LOOP_IDLE_DELAY_MS            5   // how long to yield the CPU
                                           // between poll/check cycles

// ---- TFLite Micro tensor arena size. Start here and grow if the
// interpreter reports AllocateTensors() failing -- print the actual
// bytes used (see main .ino) and right-size this once you know it. ----
#define TENSOR_ARENA_SIZE (40 * 1024)

// ---- TLS: encrypts everything in transit between the ESP32 and your
// ASR server, so no one snooping on the WiFi network or the path to
// your server can read the audio -- see the chat explanation for what
// this does and does NOT guarantee. ----
#define USE_TLS   1   // 0 = plain TCP (only for local testing on a
                       // trusted network -- never for anything real)

#endif  // CONFIG_H
