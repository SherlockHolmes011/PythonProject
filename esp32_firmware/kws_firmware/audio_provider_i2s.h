// audio_provider_i2s.h
// MIC TYPE 1 of 2: digital I2S microphone (e.g. INMP441).
// Reads audio continuously into a circular buffer, so at any moment we
// can grab either "the last 1 second" (for inference) or "the last
// PREROLL_SECONDS" (to send to the cloud the instant a keyword is
// confirmed). See config.h's MIC_TYPE to switch to the other driver
// (audio_provider_analog.h, for a 2-pin electret capsule) -- both files
// expose the identical four functions below, so nothing else in the
// firmware needs to change when you switch mics.

#ifndef AUDIO_PROVIDER_I2S_H
#define AUDIO_PROVIDER_I2S_H

#include <driver/i2s.h>
#include "config.h"

static int16_t g_circular_buffer[PREROLL_SAMPLES];
static volatile int g_write_pos = 0;
static volatile bool g_buffer_wrapped = false;

inline void audio_provider_init() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // INMP441 outputs 32-bit
                                                     // frames with 24 valid bits
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN,
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);

  Serial.println("[audio] I2S mic initialized");
}

// Call this frequently from loop() (or a dedicated task) -- reads
// whatever new samples are available and appends them to the circular
// buffer. INMP441 gives 32-bit words; we shift down to 16-bit PCM.
inline void audio_provider_poll() {
  // Shrunk from 256 -> 64: this is only the software chunk size we copy
  // out of the I2S driver per call, not the driver's own internal DMA
  // buffering (dma_buf_count=8 x dma_buf_len=256 below, unchanged --
  // that's allocated separately on the heap and keeps the same ~128ms
  // of safety margin either way). Saves 768 bytes of static RAM; poll()
  // just does slightly more, cheaper reads per full DMA buffer drained.
  static int32_t raw_buf[64];
  size_t bytes_read = 0;

  i2s_read(I2S_PORT, raw_buf, sizeof(raw_buf), &bytes_read, 0);
  int samples_read = bytes_read / sizeof(int32_t);

  for (int i = 0; i < samples_read; i++) {
    int16_t sample = (int16_t)(raw_buf[i] >> 14);  // scale 32-bit -> 16-bit
    g_circular_buffer[g_write_pos] = sample;
    g_write_pos = (g_write_pos + 1) % PREROLL_SAMPLES;
    if (g_write_pos == 0) g_buffer_wrapped = true;
  }
}

inline bool audio_provider_has_full_buffer() {
  return g_buffer_wrapped;
}

// Copies the most recent `num_samples` into `out`, oldest-first.
inline void audio_provider_get_recent(int16_t* out, int num_samples) {
  for (int i = 0; i < num_samples; i++) {
    int idx = (g_write_pos - num_samples + i + PREROLL_SAMPLES * 2) % PREROLL_SAMPLES;
    out[i] = g_circular_buffer[idx];
  }
}

#endif  // AUDIO_PROVIDER_I2S_H
