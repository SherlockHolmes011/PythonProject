// model_data.cc
// PLACEHOLDER -- see the comment in model_data.h. This dummy array will
// NOT run correctly; it exists only so the firmware compiles before you
// have a trained model. Replace with exported/kws_model_data.cc.

#include "model_data.h"

alignas(8) const unsigned char g_kws_model_data[] = {
  0x00, 0x00, 0x00, 0x00,
};
const int g_kws_model_data_len = sizeof(g_kws_model_data);
