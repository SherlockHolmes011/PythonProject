// kws_firmware.ino
//
// Ties together every piece discussed in the design:
//   - continuously capture audio (audio_provider.h)
//   - periodically extract MFCC features (mfcc.h) and run the quantized
//     KWS model (TFLite Micro)
//   - apply the threshold + N-consecutive-frames smoothing rule before
//     trusting a detection
//   - on a confirmed detection, grab the pre-roll buffer and stream it
//     to the cloud ASR server
//
// Libraries needed (Arduino IDE -> Library Manager):
//   - "TensorFlowLite_ESP32" (or "Chirale_TensorFlowLite" -- either
//     Arduino-friendly TFLite Micro port works; adjust the include below
//     to match whichever you install)
//   - Uses the ESP32's built-in <driver/i2s.h> and <WiFi.h>, no install
//     needed for those.
//
// IMPORTANT: this compiles as a template even with the placeholder
// model, but will not detect anything real until you replace
// model_data.h/.cc with your actual trained+quantized model (see the
// comment in model_data.h) and validate feature parity (see
// README_ESP32.md).

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_pm.h>
#include <esp_sleep.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "config.h"
#include "model_data.h"
#if MIC_TYPE == MIC_TYPE_I2S
  #include "audio_provider_i2s.h"
#else
  #include "audio_provider_analog.h"
#endif
#include "mfcc.h"
#include "certs.h"

// ---- TFLite Micro globals ----
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// ---- Detection state ----
enum DeviceState { LISTENING, STREAMING, COOLDOWN };
DeviceState g_state = LISTENING;
int g_consecutive_hits = 0;
unsigned long g_cooldown_until_ms = 0;

// Detection class indices must match LABELS in prepare_dataset.py: 0=keyword,
// 1=unknown, 2=silence.
#define CLASS_KEYWORD 0

// Shared audio scratch buffer, sized for the LARGER of the two things
// that ever get copied out of the circular buffer (the 1.5s pre-roll).
// run_inference() and stream_to_cloud_asr() are never active at the same
// time (LISTENING and STREAMING are mutually exclusive states), so they
// can safely share one buffer instead of each holding their own --
// this alone removes ~32KB of duplicated static RAM versus declaring
// separate buffers in each function.
static int16_t g_audio_scratch[PREROLL_SAMPLES];

void print_state_change(DeviceState new_state) {
  // Explicit Serial print at every state transition -- makes debugging
  // false triggers or streaming stalls on real hardware far easier than
  // trying to infer what happened after the fact.
  Serial.print("[state] -> ");
  switch (new_state) {
    case LISTENING: Serial.println("LISTENING"); break;
    case STREAMING: Serial.println("STREAMING"); break;
    case COOLDOWN:  Serial.println("COOLDOWN");  break;
  }
}

void setup_model() {
  model = tflite::GetModel(g_kws_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[model] ERROR: schema version mismatch");
    while (true) delay(1000);
  }

  static tflite::AllOpsResolver resolver;  // simplest to get compiling;
  // once it works, swap for a MicroMutableOpResolver with only the ops
  // your model actually uses (Conv2D, DepthwiseConv2D, MaxPool2D, Mean,
  // FullyConnected, Softmax, Reshape/Quantize) to save flash.

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("[model] ERROR: AllocateTensors() failed -- "
                    "increase TENSOR_ARENA_SIZE in config.h");
    while (true) delay(1000);
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.printf("[model] arena used: %d bytes\n", interpreter->arena_used_bytes());
  Serial.printf("[model] input scale=%.6f zero_point=%d\n",
                model_input->params.scale, model_input->params.zero_point);
}

void connect_wifi() {
  Serial.print("[wifi] connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println(" connected");
}

// "Blink, don't stare": configures ESP-IDF's automatic tickless-idle
// light sleep. Unlike a manual esp_light_sleep_start() call (which would
// require us to explicitly stop and restart I2S ourselves), this mode
// lets FreeRTOS automatically drop the CPU into light sleep during any
// idle gap it finds -- including the delay() calls already in loop() --
// while peripherals clocked to keep running (the I2S mic's DMA) continue
// capturing audio in the background. RAM is preserved, so the circular
// buffer survives every sleep cycle intact.
//
// HONESTY NOTE: exact peripheral behavior during automatic light sleep
// varies by ESP-IDF version and board. Verify on your real hardware with
// the Serial free-heap logging from earlier, and watch for audio gaps in
// your feature-parity test (see README_ESP32.md) after enabling this --
// if you see dropped/garbled audio, disable it (config.h) and fall back
// to just the loop-delay fix, which is safe unconditionally.
void setup_power_management() {
#if ENABLE_AUTOMATIC_LIGHT_SLEEP
  esp_pm_config_t pm_config = {
    .max_freq_mhz = 240,
    .min_freq_mhz = 80,   // CPU scales down instead of sleeping fully
                          // whenever a full sleep isn't safe to enter
    .light_sleep_enable = true,
  };
  esp_err_t err = esp_pm_configure(&pm_config);
  if (err == ESP_OK) {
    Serial.println("[power] automatic light sleep enabled");
  } else {
    Serial.printf("[power] WARNING: esp_pm_configure failed (%d) -- "
                   "check that CONFIG_PM_ENABLE / CONFIG_FREERTOS_USE_TICKLESS_IDLE "
                   "are on for your board in menuconfig/sdkconfig\n", err);
  }
#else
  Serial.println("[power] automatic light sleep disabled (config.h)");
#endif
}

// Quantizes float MFCC features into the model's int8 input tensor using
// the model's OWN scale/zero_point (read at runtime), rather than a
// hardcoded constant -- this keeps it correct even if you retrain with
// different quantization ranges.
void fill_model_input(float features[N_FRAMES][N_MELS]) {
  float scale = model_input->params.scale;
  int zero_point = model_input->params.zero_point;
  int8_t* input_data = model_input->data.int8;

  int idx = 0;
  for (int f = 0; f < N_FRAMES; f++) {
    for (int c = 0; c < N_MELS; c++) {
      int quantized = (int)roundf(features[f][c] / scale) + zero_point;
      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;
      input_data[idx++] = (int8_t)quantized;
    }
  }
}

// Runs one inference pass and returns the KEYWORD class confidence,
// dequantized back to a 0.0-1.0 float using the output tensor's own
// scale/zero_point.
float run_inference() {
  static float features[N_FRAMES][N_MELS];

  audio_provider_get_recent(g_audio_scratch, CLIP_SAMPLES);
  compute_mfcc(g_audio_scratch, features);
  fill_model_input(features);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[model] ERROR: Invoke() failed");
    return 0.0f;
  }

  float out_scale = model_output->params.scale;
  int out_zero_point = model_output->params.zero_point;
  int8_t raw = model_output->data.int8[CLASS_KEYWORD];
  return (raw - out_zero_point) * out_scale;
}

// Streams the pre-roll buffer + a short tail of live audio to the cloud
// ASR server as raw 16-bit PCM, over TLS (see the chat explanation for
// exactly what this does and doesn't protect against). (Swap raw PCM for
// Opus/mu-law encoding once the basic path works -- see the design
// discussion for why that helps both latency and bandwidth.)
void stream_to_cloud_asr() {
#if USE_TLS
  WiFiClientSecure client;
  #if defined(ASR_SERVER_CA_CERT)
    client.setCACert(ASR_SERVER_CA_CERT);  // verifies the server's identity
  #else
    client.setInsecure();  // TESTING ONLY -- see certs.h. Encrypts the
                            // connection but does NOT verify who's on the
                            // other end, so it's vulnerable to
                            // man-in-the-middle on an untrusted network.
                            // Replace with setCACert() before this leaves
                            // your bench.
  #endif
#else
  WiFiClient client;  // plain, unencrypted -- local testing only
#endif

  Serial.printf("[stream] connecting to %s:%d (TLS=%d)\n", ASR_HOST, ASR_PORT, USE_TLS);

  if (!client.connect(ASR_HOST, ASR_PORT)) {
    Serial.println("[stream] ERROR: connection failed");
    return;
  }

  audio_provider_get_recent(g_audio_scratch, PREROLL_SAMPLES);

  String header = String("POST ") + ASR_PATH + " HTTP/1.1\r\n" +
                  "Host: " + ASR_HOST + "\r\n" +
                  "Content-Type: audio/l16;rate=16000\r\n" +
                  "Content-Length: " + String(PREROLL_SAMPLES * 2) + "\r\n" +
                  "Connection: close\r\n\r\n";
  client.print(header);
  client.write((const uint8_t*)g_audio_scratch, PREROLL_SAMPLES * 2);

  Serial.println("[stream] pre-roll sent, streaming live audio...");

  unsigned long stream_start = millis();
  const unsigned long STREAM_DURATION_MS = 3000;  // simple fixed window;
  // swap for real VAD-based end-of-speech detection once basic streaming
  // is verified working.

  while (millis() - stream_start < STREAM_DURATION_MS) {
    audio_provider_poll();
    static int16_t chunk[512];
    audio_provider_get_recent(chunk, 512);
    client.write((const uint8_t*)chunk, sizeof(chunk));
    delay(20);
  }

  client.stop();
  Serial.println("[stream] done");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] KWS firmware starting");

  setup_power_management();
  mfcc_init_filterbank();
  audio_provider_init();
  setup_model();

  print_state_change(g_state);
}

void loop() {
  audio_provider_poll();

  static unsigned long last_inference_ms = 0;
  unsigned long now = millis();

  switch (g_state) {
    case LISTENING: {
      if (!audio_provider_has_full_buffer()) {
        // Buffer still filling on first boot -- nothing useful to do yet.
        vTaskDelay(pdMS_TO_TICKS(LOOP_IDLE_DELAY_MS));
        return;
      }
      if (now - last_inference_ms < INFERENCE_INTERVAL_MS) {
        // THE ACTUAL FIX: this used to be a bare `return`, which meant
        // loop() spun as fast as the CPU physically could, re-checking
        // "has 200ms passed yet?" thousands of times a second -- that
        // busy-wait was very likely costing more CPU than the model
        // inference itself. vTaskDelay() yields the CPU for real, which
        // is also what gives automatic light sleep (above) an idle
        // window to actually use.
        vTaskDelay(pdMS_TO_TICKS(LOOP_IDLE_DELAY_MS));
        return;
      }
      last_inference_ms = now;

      float confidence = run_inference();
      Serial.printf("[infer] keyword confidence=%.3f\n", confidence);

      if (confidence >= DETECTION_THRESHOLD) {
        g_consecutive_hits++;
      } else {
        g_consecutive_hits = 0;
      }

      if (g_consecutive_hits >= CONSECUTIVE_FRAMES_NEEDED) {
        Serial.println("[detect] KEYWORD CONFIRMED");
        g_consecutive_hits = 0;
        g_state = STREAMING;
        print_state_change(g_state);
      }
      break;
    }

    case STREAMING: {
      if (WiFi.status() != WL_CONNECTED) connect_wifi();
      stream_to_cloud_asr();
      g_cooldown_until_ms = millis() + COOLDOWN_MS;
      g_state = COOLDOWN;
      print_state_change(g_state);
      break;
    }

    case COOLDOWN: {
      if (millis() >= g_cooldown_until_ms) {
        g_state = LISTENING;
        print_state_change(g_state);
      } else {
        vTaskDelay(pdMS_TO_TICKS(LOOP_IDLE_DELAY_MS));
      }
      break;
    }
  }
}
