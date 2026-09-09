// audio_provider_analog.h
// MIC TYPE 2 of 2: bare 2-pin electret condenser capsule, read through
// the ESP32's built-in ADC.
//
// HONEST DIFFERENCE FROM audio_provider_i2s.h: the I2S driver offloads
// sample capture to dedicated DMA hardware, costing almost no CPU. This
// analog path has no equivalent free ride -- it uses a hardware timer
// firing at SAMPLE_RATE (16000 times/sec) to actively read the ADC each
// time, which costs real CPU cycles the I2S path doesn't. Expect higher
// idle CPU usage with this driver; if that pushes you over your <10%
// budget, that's the tradeoff for using the cheaper analog capsule --
// see the chat explanation of why for the full picture.
//
// REQUIRED EXTERNAL CIRCUIT (this header can't do this part for you):
//   3.3V --[bias resistor, ~4.7k]--+-- capsule pin 1 (signal)
//                                  |
//                                  +--[coupling cap, ~2.2uF]-- ADC pin
//   GND ----------------------------- capsule pin 2
// See README_ESP32.md for the full wiring diagram and why each part is
// there. Without this circuit, the capsule will not produce a usable
// signal at all -- this is not optional plug-and-play hardware.

#ifndef AUDIO_PROVIDER_ANALOG_H
#define AUDIO_PROVIDER_ANALOG_H

#include <driver/adc.h>
#include <esp_timer.h>
#include "config.h"

static int16_t g_circular_buffer[PREROLL_SAMPLES];
static volatile int g_write_pos = 0;
static volatile bool g_buffer_wrapped = false;
static portMUX_TYPE g_buffer_mux = portMUX_INITIALIZER_UNLOCKED;

static esp_timer_handle_t g_sample_timer = nullptr;

// Fires SAMPLE_RATE times per second. Runs in the esp_timer service
// task (not a true hardware ISR on Arduino-ESP32), but we still guard
// the shared write index with a critical section since it's genuinely
// concurrent with whatever loop() is doing.
static void IRAM_ATTR sample_timer_callback(void* arg) {
  int raw = adc1_get_raw(ANALOG_MIC_ADC_CHANNEL);  // 0-4095, 12-bit

  // Remove the DC bias the external circuit adds, then scale the
  // 12-bit ADC range up to the same 16-bit range the rest of the
  // firmware (mfcc.h, fill_model_input, etc.) already expects --
  // this is what lets both mic drivers share every downstream file.
  int centered = raw - ANALOG_MIC_ADC_MIDPOINT;
  int16_t sample = (int16_t)constrain(centered * 16, -32768, 32767);

  portENTER_CRITICAL_ISR(&g_buffer_mux);
  g_circular_buffer[g_write_pos] = sample;
  g_write_pos = (g_write_pos + 1) % PREROLL_SAMPLES;
  if (g_write_pos == 0) g_buffer_wrapped = true;
  portEXIT_CRITICAL_ISR(&g_buffer_mux);
}

inline void audio_provider_init() {
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ANALOG_MIC_ADC_CHANNEL, ADC_ATTEN_DB_11);  // full 0-3.3V range

  const esp_timer_create_args_t timer_args = {
    .callback = &sample_timer_callback,
    .name = "mic_sample_timer",
  };
  esp_timer_create(&timer_args, &g_sample_timer);
  esp_timer_start_periodic(g_sample_timer, 1000000 / SAMPLE_RATE);  // microseconds

  Serial.println("[audio] Analog electret mic initialized (ADC + timer)");
  Serial.println("[audio] NOTE: run the calibration step in README_ESP32.md "
                  "before trusting detections -- ANALOG_MIC_ADC_MIDPOINT in "
                  "config.h is a default guess, not measured on your circuit.");
}

// Sampling happens entirely in the timer callback above, in the
// background -- nothing to do here. Kept as a no-op (rather than
// removing the call from kws_firmware.ino) so both mic drivers expose
// the identical four-function interface and the rest of the firmware
// never needs to know which one is active.
inline void audio_provider_poll() {
  // intentionally empty
}

inline bool audio_provider_has_full_buffer() {
  return g_buffer_wrapped;
}

// Copies the most recent `num_samples` into `out`, oldest-first.
inline void audio_provider_get_recent(int16_t* out, int num_samples) {
  portENTER_CRITICAL(&g_buffer_mux);
  int write_pos_snapshot = g_write_pos;
  portEXIT_CRITICAL(&g_buffer_mux);

  for (int i = 0; i < num_samples; i++) {
    int idx = (write_pos_snapshot - num_samples + i + PREROLL_SAMPLES * 2) % PREROLL_SAMPLES;
    out[i] = g_circular_buffer[idx];
  }
}

#endif  // AUDIO_PROVIDER_ANALOG_H
