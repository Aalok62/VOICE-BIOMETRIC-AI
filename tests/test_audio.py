"""
tests/test_audio.py
===================
Unit tests for audio preprocessing and feature extraction.
Run with: pytest tests/test_audio.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.audio.preprocessing import (
    normalize_audio, reduce_noise, validate_audio,
    trim_audio, trim_silence
)
from src.audio.features import (
    extract_mel_spectrogram, extract_mfcc,
    mel_to_fixed_size, get_audio_info,
    plot_waveform, plot_mel_spectrogram
)
from config import TARGET_SAMPLE_RATE, N_MELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_wave():
    """Generate a clean 2-second 440 Hz sine wave at 16 kHz."""
    sr = TARGET_SAMPLE_RATE
    t  = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    y  = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


@pytest.fixture
def silent_audio():
    """Generate 2 seconds of silence."""
    sr = TARGET_SAMPLE_RATE
    y  = np.zeros(int(2.0 * sr), dtype=np.float32)
    return y, sr


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_peak_normalization_range(self, sine_wave):
        y, sr = sine_wave
        y_norm = normalize_audio(y, method="peak")
        assert np.max(np.abs(y_norm)) <= 1.0 + 1e-6

    def test_peak_normalization_max_is_one(self, sine_wave):
        y, sr = sine_wave
        y_norm = normalize_audio(y, method="peak")
        assert abs(np.max(np.abs(y_norm)) - 1.0) < 1e-5

    def test_rms_normalization(self, sine_wave):
        y, sr = sine_wave
        y_norm = normalize_audio(y, method="rms")
        rms = float(np.sqrt(np.mean(y_norm ** 2)))
        assert abs(rms - 0.1) < 0.01

    def test_dtype_preserved(self, sine_wave):
        y, sr = sine_wave
        y_norm = normalize_audio(y)
        assert y_norm.dtype == np.float32


class TestValidation:
    def test_valid_audio_passes(self, sine_wave):
        y, sr = sine_wave
        result = validate_audio(y, sr)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_silent_audio_fails(self, silent_audio):
        y, sr = silent_audio
        result = validate_audio(y, sr)
        assert result["valid"] is False
        assert any("silent" in e.lower() for e in result["errors"])

    def test_too_short_fails(self):
        sr = TARGET_SAMPLE_RATE
        y  = np.ones(int(0.1 * sr), dtype=np.float32) * 0.5  # 0.1 s
        result = validate_audio(y, sr)
        assert result["valid"] is False
        assert any("short" in e.lower() for e in result["errors"])

    def test_duration_computed(self, sine_wave):
        y, sr = sine_wave
        result = validate_audio(y, sr)
        assert abs(result["duration"] - 2.0) < 0.1


class TestTrimming:
    def test_trim_audio_length(self, sine_wave):
        y, sr = sine_wave
        y_trimmed = trim_audio(y, sr, max_duration=1.0)
        assert len(y_trimmed) == sr  # 1 second

    def test_short_audio_unchanged(self, sine_wave):
        y, sr = sine_wave
        y_trimmed = trim_audio(y, sr, max_duration=5.0)
        assert len(y_trimmed) == len(y)


class TestNoiseReduction:
    def test_noise_reduction_shape_preserved(self, sine_wave):
        y, sr = sine_wave
        y_denoised = reduce_noise(y, sr)
        assert y_denoised.shape == y.shape

    def test_noise_reduction_dtype(self, sine_wave):
        y, sr = sine_wave
        y_denoised = reduce_noise(y, sr)
        assert y_denoised.dtype == np.float32


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestMelSpectrogram:
    def test_mel_shape(self, sine_wave):
        y, sr = sine_wave
        mel = extract_mel_spectrogram(y, sr)
        assert mel.shape[0] == N_MELS

    def test_mel_dtype(self, sine_wave):
        y, sr = sine_wave
        mel = extract_mel_spectrogram(y, sr)
        assert mel.dtype == np.float32

    def test_mel_to_fixed_size_pads(self, sine_wave):
        y, sr = sine_wave
        mel = extract_mel_spectrogram(y, sr)
        mel_fixed = mel_to_fixed_size(mel, target_frames=128)
        assert mel_fixed.shape == (N_MELS, 128)

    def test_mel_to_fixed_size_crops(self, sine_wave):
        y, sr = sine_wave
        big_mel = np.zeros((N_MELS, 300), dtype=np.float32)
        mel_fixed = mel_to_fixed_size(big_mel, target_frames=128)
        assert mel_fixed.shape == (N_MELS, 128)


class TestMFCC:
    def test_mfcc_shape(self, sine_wave):
        y, sr = sine_wave
        mfcc = extract_mfcc(y, sr, n_mfcc=40)
        assert mfcc.shape[0] == 40

    def test_mfcc_dtype(self, sine_wave):
        y, sr = sine_wave
        mfcc = extract_mfcc(y, sr)
        assert mfcc.dtype == np.float32


class TestAudioInfo:
    def test_info_keys(self, sine_wave):
        y, sr = sine_wave
        info = get_audio_info(y, sr)
        required_keys = {"duration_s", "sample_rate", "channels", "n_samples", "rms", "peak"}
        assert required_keys.issubset(info.keys())

    def test_duration_correct(self, sine_wave):
        y, sr = sine_wave
        info = get_audio_info(y, sr)
        assert abs(info["duration_s"] - 2.0) < 0.1


class TestPlots:
    def test_waveform_plot_returns_figure(self, sine_wave):
        import matplotlib.pyplot as plt
        y, sr = sine_wave
        fig = plot_waveform(y, sr)
        assert fig is not None
        plt.close(fig)

    def test_mel_plot_returns_figure(self, sine_wave):
        import matplotlib.pyplot as plt
        y, sr = sine_wave
        fig = plot_mel_spectrogram(y, sr)
        assert fig is not None
        plt.close(fig)
