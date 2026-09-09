// certs.h
//
// Holds the root CA certificate used to verify your ASR server's
// identity over TLS. Without this, stream_to_cloud_asr() falls back to
// client.setInsecure() -- the connection is still encrypted, but the
// ESP32 has no way to confirm it's actually talking to YOUR server and
// not an impostor on the same network. Filling this in closes that gap.
//
// How to get the right value, once you have a real ASR server running
// (see asr_server/README.md):
//
//   openssl s_client -connect your-asr-server.example.com:443 -showcerts
//
// ...and copy the certificate block (including the BEGIN/END lines) in
// below, OR (for the self-signed cert the local dev server generates)
// copy directly from asr_server/certs/server.crt.
//
// Leave ASR_SERVER_CA_CERT undefined (comment out the #define) to use
// setInsecure() for quick local testing -- just don't ship it that way.

#ifndef CERTS_H
#define CERTS_H

// #define ASR_SERVER_CA_CERT asr_root_ca_pem

// static const char asr_root_ca_pem[] = R"EOF(
// -----BEGIN CERTIFICATE-----
// PASTE YOUR SERVER'S (OR SELF-SIGNED) CERTIFICATE HERE
// -----END CERTIFICATE-----
// )EOF";

#endif  // CERTS_H
