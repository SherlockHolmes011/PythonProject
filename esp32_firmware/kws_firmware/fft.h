// fft.h
// A small self-contained iterative radix-2 Cooley-Tukey FFT.
//
// Written from scratch (not the ESP-DSP library) so this firmware has no
// extra library dependency beyond TensorFlowLite_ESP32. If you'd rather
// use Espressif's ESP-DSP library for a faster, hardware-optimized FFT,
// swap the body of fft_inplace() for a call to dsps_fft2r_fc32() --
// the rest of mfcc.cpp doesn't need to change.

#ifndef FFT_H
#define FFT_H

#include <math.h>
#include <stdint.h>

// In-place FFT. n must be a power of two. real/imag are both length n.
inline void fft_inplace(float* real, float* imag, int n) {
  // Bit-reversal permutation
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      float tr = real[i]; real[i] = real[j]; real[j] = tr;
      float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
  }

  // Iterative Cooley-Tukey butterflies
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / (float)len;
    float wr = cosf(ang);
    float wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cwr = 1.0f, cwi = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        float ur = real[i + k];
        float ui = imag[i + k];
        float vr = real[i + k + len / 2] * cwr - imag[i + k + len / 2] * cwi;
        float vi = real[i + k + len / 2] * cwi + imag[i + k + len / 2] * cwr;

        real[i + k] = ur + vr;
        imag[i + k] = ui + vi;
        real[i + k + len / 2] = ur - vr;
        imag[i + k + len / 2] = ui - vi;

        float next_cwr = cwr * wr - cwi * wi;
        float next_cwi = cwr * wi + cwi * wr;
        cwr = next_cwr;
        cwi = next_cwi;
      }
    }
  }
}

#endif  // FFT_H
