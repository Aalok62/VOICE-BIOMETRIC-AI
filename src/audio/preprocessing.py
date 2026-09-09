"""
src/audio/preprocessing.py
==========================
Audio loading, resampling, noise reduction, normalization, and validation.
"""

import io
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    TARGET_SAMPLE_RATE,
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
    SILENCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_audio(source, target_sr: int = TARGET_SAMPLE_RATE):
    """
    Load audio from a file path, bytes, or BytesIO object with fallback decoders.

    Returns
    -------
    y  : np.ndarray  – mono waveform, float32, shape (n_samples,)
    sr : int         – sample rate (= target_sr after resampling)
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    y = None
    sr = None
    last_err = None

    # Primary attempt: librosa load
    try:
        y, sr = librosa.load(source, sr=None, mono=True)
    except Exception as e1:
        last_err = e1
        # Fallback 1: torchaudio
        try:
            import torchaudio
            if isinstance(source, io.BytesIO):
                source.seek(0)
            waveform, sr = torchaudio.load(source)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            y = waveform.squeeze().numpy()
        except Exception as e2:
            last_err = e2
            # Fallback 2: soundfile directly
            try:
                if isinstance(source, io.BytesIO):
                    source.seek(0)
                data, sr = sf.read(source)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                y = data.astype(np.float32)
            except Exception as e3:
                last_err = e3

    if y is None or sr is None:
        raise ValueError(f"Could not load audio: {last_err}")

    # Resample to target sample rate if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = y.astype(np.float32)
    return y, sr


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_audio(y: np.ndarray, sr: int) -> dict:
    """
    Run sanity checks on a loaded waveform.
    """
    errors   = []
    warnings = []
    duration = len(y) / sr

    if duration < MIN_AUDIO_DURATION:
        errors.append(
            f"Audio too short ({duration:.2f}s). Minimum is {MIN_AUDIO_DURATION}s."
        )

    if duration > MAX_AUDIO_DURATION:
        warnings.append(
            f"Audio trimmed from {duration:.2f}s to {MAX_AUDIO_DURATION}s."
        )

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < SILENCE_THRESHOLD:
        errors.append(
            f"Audio appears to be silent (RMS={rms:.4f}). "
            "Please check microphone or file."
        )

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "duration": duration,
        "rms":      rms,
    }


def trim_audio(y: np.ndarray, sr: int, max_duration: float = MAX_AUDIO_DURATION):
    """Trim audio to max_duration seconds."""
    max_samples = int(max_duration * sr)
    return y[:max_samples]


def trim_silence(y: np.ndarray, sr: int, top_db: int = 30):
    """Strip leading/trailing silence using librosa."""
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed
    except Exception:
        return y


def normalize_audio(y: np.ndarray, method: str = "peak") -> np.ndarray:
    """Normalize audio amplitude."""
    if method == "peak":
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 0:
            y = y * (0.1 / rms)
    return y.astype(np.float32)


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply spectral noise reduction."""
    try:
        noise_sample_len = int(0.5 * sr)
        if len(y) > noise_sample_len * 2:
            noise_clip = y[:noise_sample_len]
            y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, stationary=False)
        else:
            y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=True)
    except Exception:
        y_denoised = y
    return y_denoised.astype(np.float32)


def preprocess(
    source,
    apply_noise_reduction: bool = True,
    normalize: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline:
      load -> validate -> trim -> noise reduce -> normalize
    """
    y, sr = load_audio(source)
    validation = validate_audio(y, sr)

    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))

    y = trim_audio(y, sr)

    if len(y) > int(0.5 * sr):
        y = trim_silence(y, sr)

    if apply_noise_reduction:
        y = reduce_noise(y, sr)

    if normalize:
        y = normalize_audio(y)

    return y, sr, validation
