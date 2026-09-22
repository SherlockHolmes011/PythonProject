"""
asr_server.py

A minimal, open-source cloud ASR server matching the exact HTTP protocol
kws_firmware.ino's stream_to_cloud_asr() already sends:

    POST /stream HTTP/1.1
    Host: ...
    Content-Type: audio/l16;rate=16000
    Content-Length: ...

    <raw 16-bit PCM bytes>

Uses faster-whisper (an open-source, efficient reimplementation of
OpenAI's open-source Whisper model) for transcription -- keeping with
your project's "open-source only" constraint. Served over HTTPS so the
ESP32's TLS connection actually terminates somewhere real.

Setup:
    pip install -r requirements.txt
    python generate_self_signed_cert.py   # for local testing
    python asr_server.py

Then point config.h's ASR_HOST at this machine's IP/hostname and
ASR_PORT at 8443 (or whatever you configure below), and either paste
certs/server.crt into the ESP32's certs.h, or leave client.setInsecure()
in place for a quick bench test (see certs.h for the tradeoff).
"""

import http.server
import ssl
import io
import wave
import numpy as np

from faster_whisper import WhisperModel

HOST = "0.0.0.0"
PORT = 8443
CERT_FILE = "asr/certs/server.crt"
KEY_FILE = "asr/certs/server.key"

SAMPLE_RATE = 16000

# "tiny" or "base" keep this usable on modest server hardware; move up to
# "small"/"medium" if you have a GPU or don't mind slower responses.
# Loaded lazily (inside main(), not at import time) so this module can be
# imported/tested without needing network access or the model weights.
model = None


def load_model():
    global model
    print("Loading Whisper model (first run downloads it)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Model loaded.")


def pcm16_bytes_to_wav_buffer(pcm_bytes, sample_rate=SAMPLE_RATE):
    """faster-whisper reads from a file-like object or path; wrapping
    the raw PCM the ESP32 sends into an in-memory WAV avoids ever
    writing anything to disk."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


class ASRHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_response(400)
            self.end_headers()
            return

        pcm_bytes = self.rfile.read(content_length)
        print(f"[asr] received {len(pcm_bytes)} bytes "
              f"(~{len(pcm_bytes) / 2 / SAMPLE_RATE:.2f}s of audio)")

        wav_buf = pcm16_bytes_to_wav_buffer(pcm_bytes)

        segments, info = model.transcribe(wav_buf, language="en")
        transcript = " ".join(seg.text.strip() for seg in segments)

        print(f"[asr] transcript: {transcript!r}")

        response_body = transcript.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Keep the default noisy per-request logging out of the way;
        # we print our own [asr] lines above instead.
        pass


def main():
    load_model()

    httpd = http.server.HTTPServer((HOST, PORT), ASRHandler)

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ssl_context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    except FileNotFoundError:
        print(f"ERROR: {CERT_FILE} / {KEY_FILE} not found.")
        print("Run generate_self_signed_cert.py first (see README.md).")
        return

    httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)

    print(f"ASR server listening on https://{HOST}:{PORT}/stream")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
