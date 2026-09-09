"""
src/audio/features.py
=====================
Feature extraction: Mel-spectrograms, MFCCs, and audio metadata.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    TARGET_SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, FMIN, FMAX
)


# ---------------------------------------------------------------------------
# Mel-Spectrogram
# ---------------------------------------------------------------------------

def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX,
    as_db: bool = True,
) -> np.ndarray:
    """
    Extract Mel-spectrogram from waveform.

    Parameters
    ----------
    as_db : if True, convert to dB scale (log-mel spectrogram)

    Returns
    -------
    S : np.ndarray shape (n_mels, time_frames)
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    if as_db:
        S = librosa.power_to_db(S, ref=np.max)
    return S.astype(np.float32)


def mel_to_fixed_size(mel: np.ndarray, target_frames: int = 128) -> np.ndarray:
    """
    Resize mel-spectrogram time axis to a fixed width for CNN input.
    Pads with minimum value or crops as needed.
    """
    n_mels, n_frames = mel.shape
    if n_frames < target_frames:
        # Pad with minimum value (silence in dB)
        pad_width = target_frames - n_frames
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant",
                     constant_values=mel.min())
    elif n_frames > target_frames:
        mel = mel[:, :target_frames]
    return mel


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

def extract_mfcc(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = 40,
) -> np.ndarray:
    """
    Extract MFCC features.

    Returns
    -------
    mfcc : np.ndarray shape (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio Info
# ---------------------------------------------------------------------------

def get_audio_info(y: np.ndarray, sr: int, filename: str = "audio") -> dict:
    """Return a dictionary of audio metadata."""
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    return {
        "filename":    filename,
        "duration_s":  round(duration, 2),
        "sample_rate": sr,
        "channels":    1,  # always mono after preprocessing
        "n_samples":   len(y),
        "rms":         round(rms, 4),
        "peak":        round(peak, 4),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_waveform(y: np.ndarray, sr: int, title: str = "Audio Waveform") -> plt.Figure:
    """Return a matplotlib Figure showing the waveform."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color="#2196F3", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444")
    ax.grid(True, alpha=0.2, color="#555")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    title: str = "Mel-Spectrogram",
) -> plt.Figure:
    """Return a matplotlib Figure showing the Mel-spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX,
        ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#444")
    fig.tight_layout()
    return fig
