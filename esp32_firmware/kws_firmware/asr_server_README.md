# ASR Server

The cloud half of the "edge wake-word, cloud ASR" architecture. An
open-source Whisper-based transcription server that speaks the exact
HTTP protocol `kws_firmware.ino`'s `stream_to_cloud_asr()` already
sends, over TLS.

## What was actually verified vs. what wasn't

Being direct about this: in the environment these files were written
in, network access to Hugging Face (where the Whisper model weights
are hosted) was blocked, so **the actual speech-to-text step could not
be run here.** What WAS verified, for real, end to end:
- The WAV-wrapping logic that converts the ESP32's raw PCM into
  something faster-whisper can read.
- A real TLS handshake and a real HTTPS server accepting a connection.
- The exact HTTP request format (`POST /stream`, headers, raw body)
  matching what the firmware sends, parsed correctly, with a stubbed
  model returning a transcript through the full response path.

So the "plumbing" is confirmed working. The one piece riding on trust
in a well-established open-source library rather than my own testing
is faster-whisper's transcription itself -- run `python asr_server.py`
on your machine (which presumably has normal internet access) and
that part will resolve itself on first run.

## Setup

```bash
pip install -r requirements.txt
python generate_self_signed_cert.py --hostname YOUR_SERVER_IP
python asr_server.py
```

First run downloads the "tiny" Whisper model automatically (a few
hundred MB). The server then listens on `https://<your-ip>:8443/stream`.

## Connecting the ESP32 to it

In `esp32_firmware/kws_firmware/config.h`:
```cpp
#define ASR_HOST   "YOUR_SERVER_IP"   // the machine running asr_server.py
#define ASR_PORT   8443
```

For the ESP32 to actually verify it's talking to your real server (not
just encrypt-and-hope), copy the contents of `certs/server.crt` into
`esp32_firmware/kws_firmware/certs.h` as `ASR_SERVER_CA_CERT` -- the
comments in that file show exactly where. Skipping this falls back to
`setInsecure()`, which still encrypts the connection but can't confirm
identity -- fine for a bench test on your own network, not for
anything beyond that.

## On the encryption/privacy question specifically

This setup gives you real, working TLS: everything between the ESP32
and this server is encrypted, and (once you've added the cert to
`certs.h`) the ESP32 verifies it's actually talking to your server, not
an impostor. That protects against exactly the kind of network-level
exposure your original problem statement raised as a privacy concern.

What it does NOT do, and can't: hide the audio content from whoever
runs this server -- transcription requires reading the audio, that's
not something encryption can route around. If "only the user can
decrypt it" needs to mean "not even my own server sees it," that's a
fundamentally different (and, for a working ASR pipeline, currently
impractical) problem -- not something achievable by configuring TLS
differently. Running this server yourself, rather than sending audio
to a third party, is the practical version of "the user controls who
sees it" that's actually achievable here.

## Model size tradeoff

`asr_server.py` uses Whisper "tiny" by default -- fast, low resource
use, appropriate for a class project demo. Accuracy is noticeably
better with "base" or "small" (edit the `WhisperModel(...)` line in
`asr_server.py`), at the cost of slower responses and more RAM/CPU on
whatever machine runs this server (not the ESP32 -- this part runs on
your PC/cloud instance, unrelated to the 256KB constraint).
