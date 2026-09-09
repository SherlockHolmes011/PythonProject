// model_data.h
//
// PLACEHOLDER. This is NOT a trained model -- it's here so the firmware
// compiles as a template before you've trained anything.
//
// Once you've run quantize_and_export.py, DELETE this file and
// model_data.cc, and copy in the real ones instead:
//   exported/kws_model_data.h  -> esp32_firmware/kws_firmware/model_data.h
//   exported/kws_model_data.cc -> esp32_firmware/kws_firmware/model_data.cc
//
// Those files already define the same symbol names used below
// (g_kws_model_data / g_kws_model_data_len), so no other code needs to
// change -- just swap these two files out.

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

extern const unsigned char g_kws_model_data[];
extern const int g_kws_model_data_len;

#endif  // MODEL_DATA_H
