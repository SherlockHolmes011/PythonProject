// mfcc.h
// Computes the same (49, N_MEL_BINS) log-mel spectrogram on-device that
// features.py computes in Python during training -- same sample rate,
// window, hop, FFT size, and mel filter count (see config.h). No DCT
// step: the model was trained on log-mel energies directly, matching
// the merged architecture's input. This parity is the single most
// important thing to verify before trusting the deployed model; see
// the "Validating feature parity" section of README_ESP32.md.

#ifndef MFCC_H
#define MFCC_H

#include <math.h>
#include "config.h"
#include "fft.h"

// Mel filterbank weights, built once at startup.
// Shape: [N_MELS][N_FFT/2 + 1]
// Stored as fixed-point (uint16_t, scaled by MEL_FILTERBANK_SCALE) instead
// of float -- these weights are always in [0.0, 1.0], so this halves the
// array's static RAM footprint (20,560 -> 10,280 bytes for N_MELS=20,
// N_FFT=512) with negligible precision loss, well below what the int8
// model quantization already introduces downstream.
#define MEL_FILTERBANK_SCALE 65535.0f
static uint16_t g_mel_filterbank[N_MELS][N_FFT / 2 + 1];

inline float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}
inline float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Builds a triangular mel filterbank spanning 0Hz to SAMPLE_RATE/2,
// matching librosa's default filterbank shape used in features.py.
inline void mfcc_init_filterbank() {
  const int n_bins = N_FFT / 2 + 1;
  float mel_min = hz_to_mel(0.0f);
  float mel_max = hz_to_mel(SAMPLE_RATE / 2.0f);

  float mel_points[N_MELS + 2];
  for (int i = 0; i < N_MELS + 2; i++) {
    mel_points[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);
  }

  int bin_points[N_MELS + 2];
  for (int i = 0; i < N_MELS + 2; i++) {
    float hz = mel_to_hz(mel_points[i]);
    bin_points[i] = (int)floorf((N_FFT + 1) * hz / SAMPLE_RATE);
  }

  for (int m = 0; m < N_MELS; m++) {
    int left = bin_points[m];
    int center = bin_points[m + 1];
    int right = bin_points[m + 2];
    for (int b = 0; b < n_bins; b++) {
      g_mel_filterbank[m][b] = 0;
    }
    for (int b = left; b < center; b++) {
      if (b >= 0 && b < n_bins && center > left) {
        float w = (float)(b - left) / (float)(center - left);
        g_mel_filterbank[m][b] = (uint16_t)roundf(w * MEL_FILTERBANK_SCALE);
      }
    }
    for (int b = center; b < right; b++) {
      if (b >= 0 && b < n_bins && right > center) {
        float w = (float)(right - b) / (float)(right - center);
        g_mel_filterbank[m][b] = (uint16_t)roundf(w * MEL_FILTERBANK_SCALE);
      }
    }
  }
}

// Computes DCT-II of a length-N_MELS input, keeping only the first
// N_MFCC coefficients. NOT used by compute_mfcc() below (the current
// model consumes log-mel energies directly, no DCT) -- left in place in
// case you retrain with MFCC features instead and want it back.
inline void dct_ii(const float* in, int n_in, float* out, int n_out) {
  for (int k = 0; k < n_out; k++) {
    float sum = 0.0f;
    for (int i = 0; i < n_in; i++) {
      sum += in[i] * cosf(M_PI / n_in * (i + 0.5f) * k);
    }
    out[k] = sum;
  }
}

// audio: CLIP_SAMPLES int16 samples (one second of audio, already at
//        SAMPLE_RATE), scaled to [-1, 1] internally.
// out_features: [N_FRAMES][N_MELS] float32 log-mel output.
inline void compute_mfcc(const int16_t* audio, float out_features[N_FRAMES][N_MELS]) {
  static float real[N_FFT];
  static float imag[N_FFT];
  // Same fixed-point trick as g_mel_filterbank: Hamming window values are
  // always in [0.08, 1.0], so uint16_t (scaled by MEL_FILTERBANK_SCALE)
  // costs half the RAM (800 bytes vs 1,600) with negligible precision loss.
  static uint16_t window[WIN_LENGTH];
  static bool window_init = false;

  if (!window_init) {
    // Hamming window, matching librosa's default.
    for (int i = 0; i < WIN_LENGTH; i++) {
      float w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (WIN_LENGTH - 1));
      window[i] = (uint16_t)roundf(w * MEL_FILTERBANK_SCALE);
    }
    window_init = true;
  }

  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LENGTH;

    for (int i = 0; i < N_FFT; i++) {
      if (i < WIN_LENGTH && (start + i) < CLIP_SAMPLES) {
        real[i] = (audio[start + i] / 32768.0f) * ((float)window[i] / MEL_FILTERBANK_SCALE);
      } else {
        real[i] = 0.0f;  // zero-padded to N_FFT
      }
      imag[i] = 0.0f;
    }

    fft_inplace(real, imag, N_FFT);

    // Power spectrum for the first N_FFT/2 + 1 bins (real-signal symmetry).
    float power[N_FFT / 2 + 1];
    for (int b = 0; b < N_FFT / 2 + 1; b++) {
      power[b] = real[b] * real[b] + imag[b] * imag[b];
    }

    // Apply mel filterbank, then log -- this IS the feature (no DCT).
    float mel_energies[N_MELS];
    for (int m = 0; m < N_MELS; m++) {
      float sum = 0.0f;
      for (int b = 0; b < N_FFT / 2 + 1; b++) {
        sum += power[b] * (float)g_mel_filterbank[m][b];
      }
      sum /= MEL_FILTERBANK_SCALE;  // undo the fixed-point scale once per
                                     // band, instead of once per multiply
      mel_energies[m] = logf(sum + 1e-6f);  // matches features.py's 1e-6 epsilon
    }

    for (int c = 0; c < N_MELS; c++) {
      out_features[frame][c] = mel_energies[c];
    }
  }
}

#endif  // MFCC_H