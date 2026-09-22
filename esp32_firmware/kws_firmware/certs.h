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

#define ASR_SERVER_CA_CERT asr_root_ca_pem

static const char asr_root_ca_pem[] = R"EOF(

-----BEGIN CERTIFICATE-----
MIIC3TCCAcWgAwIBAgIUFny6dA7cY/L4lTJDg1RMS0SCY5MwDQYJKoZIhvcNAQEL
BQAwGTEXMBUGA1UEAwwOWU9VUl9TRVJWRVJfSVAwHhcNMjYwOTEwMTcyNTI3WhcN
MjcwOTEwMTcyNTI3WjAZMRcwFQYDVQQDDA5ZT1VSX1NFUlZFUl9JUDCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBANw3xslQRkPf8hZYXbNGY6y4A2pkXwON
Clvqj/puDXIwpofQBi1IuU5dcMyvTqu7Xx/wE+jo+sja6D6oSkBSKWudRNFH97Be
e9gVTMCSRVYz4ixOfTAoPC2l4/1htjyHAY4tKQt8CIKkLORnbWyLDnBffZCEoWUp
r0ah3UXpxVFc5mPUXUmrp3u/S3HdYJbD7sO1OW2rbrMlGLZlwO6q8cyArfykMVZD
/oygkYtr6mYEbAH7E/LkAe0qD0mgic1yknTOYQVe9yhpNXTvKWWNLBTOHX7OSxWf
wfZKIvz1HHDxXlhHbEYKq7hwdI/tTHIwBD6HciW0fldoUN28Ik53z+kCAwEAAaMd
MBswGQYDVR0RBBIwEIIOWU9VUl9TRVJWRVJfSVAwDQYJKoZIhvcNAQELBQADggEB
ANLkU3PrSWVHFFdGjC2ZzTOqJXiXTDA2ZWf+LnH96PdZ9osncVr1F2fFGgKfHWq+
po/oudbhMWCzi6Rxs01VCkD0c29j2JOgz4WMYnSdmn/ZudI/p7irf8GkMLp58S82
75RHjuMUkOJbm2QZ9YN4M3Xobf+Yi/DIUyt2+Ur4CSeXZH1EJdm15yIp+6d4c2d+
E53zkQeYQaNZ4uGm0KyeNpjRK/agK4gg40NLLTmBnAhzVUxPANZKPJcrMR+0Mgvr
UpMF2opus7enefm8h0LtxaLOOpbOzBh9HO7uGPSsTbWBE5KWmfXn5SPJ1kpwf4Mv
O3nrm7JrptqSpIjhaTJFhQQ=
-----END CERTIFICATE-----

 )EOF";

#endif  // CERTS_H
