"""
features.py

Shared audio -> feature-map conversion used by prepare_dataset.py,
train_kws_dscnn.py, and quantize_and_export.py.

Every script that touches audio imports from here, so the *exact* same
windowing/hop/MFCC settings are used at training time and (later) on the
microcontroller. If these numbers drift between training and the on-device
feature extractor, accuracy silently degrades after deployment -- this file
exists specifically to prevent that.
"""

import numpy as np
import librosa

# ---- Audio settings ---------------------------------------------------
SAMPLE_RATE = 16000          # 16kHz mono, standard for speech
CLIP_DURATION_S = 1.0        # each training example is a 1-second clip
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)

# ---- Log-mel spectrogram settings ---------------------------------------
# win_length=400 (25ms) / hop_length=320 (20ms) on a 1s clip yields exactly
# 49 time frames. We use a log-mel spectrogram (no DCT step) rather than
# MFCC: a CNN can learn useful spatial patterns directly from the mel
# energies, and skipping the DCT keeps the ESP32-side feature extractor
# simpler (one less step to keep in exact sync with Python).
N_MEL_BINS = 20               # kept LOW deliberately: the ESP32 firmware
                               # rebuilds this same mel filterbank in C++,
                               # and storing the filterbank matrix costs
                               # RAM, so both sides must agree on this
                               # number explicitly.
N_FFT = 512
WIN_LENGTH = 400              # 25 ms
HOP_LENGTH = 320              # 20 ms

NUM_FRAMES = 49
INPUT_SHAPE = (NUM_FRAMES, N_MEL_BINS, 1)  # (time, mel_bins, channel)


def load_and_fix_length(path, sample_rate=SAMPLE_RATE, clip_samples=CLIP_SAMPLES):
    """Load a wav file, resample to SAMPLE_RATE, and force it to exactly
    clip_samples long (center-crop if too long, zero-pad if too short)."""
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if len(audio) > clip_samples:
        start = (len(audio) - clip_samples) // 2
        audio = audio[start:start + clip_samples]
    elif len(audio) < clip_samples:
        pad_total = clip_samples - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right))
    return audio.astype(np.float32)


def extract_log_mel(audio, sample_rate=SAMPLE_RATE):
    """Turn a 1-second waveform into a (49, N_MEL_BINS, 1) log-mel
    spectrogram -- the raw mel energies (log-compressed), with no DCT
    step, so the CNN learns spatial patterns directly from them."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MEL_BINS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )  # shape: (n_mels, n_frames)

    log_mel = np.log(mel_spec + 1e-6)
    log_mel = log_mel.T  # -> (n_frames, n_mels)

    # Force exactly NUM_FRAMES frames (guards against off-by-one rounding).
    target_frames = INPUT_SHAPE[0]
    if log_mel.shape[0] > target_frames:
        log_mel = log_mel[:target_frames, :]
    elif log_mel.shape[0] < target_frames:
        pad = target_frames - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)))

    log_mel = log_mel[..., np.newaxis]  # -> (49, N_MEL_BINS, 1)
    return log_mel.astype(np.float32)


def wav_to_features(path):
    """Convenience wrapper: wav file path -> (49, N_MEL_BINS, 1) float32
    log-mel feature map."""
    audio = load_and_fix_length(path)
    return extract_log_mel(audio)


def add_background_noise(audio, noise_clips, snr_db_range=(0.0, 15.0), prob=0.8):
    """Randomly mix in a snippet of background _background_noise_ at a random SNR.
    noise_clips: list of 1D float32 numpy arrays (raw _background_noise_ waveforms)."""
    if not noise_clips or np.random.rand() > prob:
        return audio

    noise = noise_clips[np.random.randint(len(noise_clips))]
    if len(noise) < len(audio):
        reps = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, reps)
    start = np.random.randint(0, len(noise) - len(audio) + 1)
    noise_seg = noise[start:start + len(audio)]

    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise_seg ** 2) + 1e-10
    snr_db = np.random.uniform(*snr_db_range)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    mixed = audio + noise_seg * scale
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def time_shift(audio, max_shift_ms=100, sample_rate=SAMPLE_RATE):
    """Randomly shift audio left/right, wrapping around (cheap augmentation
    that helps the model tolerate imperfect keyword alignment)."""
    max_shift = int(sample_rate * max_shift_ms / 1000)
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(audio, shift)


def volume_jitter(audio, gain_range=(0.7, 1.3)):
    gain = np.random.uniform(*gain_range)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
