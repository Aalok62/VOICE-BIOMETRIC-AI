"""
src/audio/inference.py
======================
High-level inference orchestrator.
Ties together preprocessing, feature extraction, and the classifier.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.audio.preprocessing import preprocess
from src.audio.features import (
    extract_mel_spectrogram, get_audio_info,
    plot_waveform, plot_mel_spectrogram
)
from src.prediction.classifier import BioSoundClassifier


def run_pipeline(source, classifier: BioSoundClassifier = None):
    """
    Full inference pipeline: load -> preprocess -> features -> predict.

    Parameters
    ----------
    source     : file path (str), bytes, or BytesIO
    classifier : optional pre-instantiated BioSoundClassifier
                 (creates a new one with default config if None)

    Returns
    -------
    result : dict with keys:
        - predicted_class  (str)
        - confidence       (float 0-1)
        - all_probabilities (dict class_name -> float)
        - top_k            (list of (class_name, prob) tuples)
        - audio_info       (dict)
        - validation       (dict)
        - waveform_fig     (matplotlib Figure)
        - mel_fig          (matplotlib Figure)
        - y                (np.ndarray) raw waveform
        - sr               (int) sample rate
    """
    if classifier is None:
        classifier = BioSoundClassifier()

    # --- Step 1: Preprocess ---
    y, sr, validation = preprocess(source)

    # --- Step 2: Audio metadata ---
    filename = source if isinstance(source, str) else "recorded_audio"
    audio_info = get_audio_info(y, sr, filename=os.path.basename(str(filename)))

    # --- Step 3: Predict ---
    prediction = classifier.predict(y, sr)

    # --- Step 4: Visualizations ---
    waveform_fig = plot_waveform(y, sr, title="Input Audio Waveform")
    mel_fig = plot_mel_spectrogram(y, sr, title="Mel-Spectrogram")

    return {
        "predicted_class":   prediction["predicted_class"],
        "confidence":        prediction["confidence"],
        "all_probabilities": prediction["all_probabilities"],
        "top_k":             prediction["top_k"],
        "audio_info":        audio_info,
        "validation":        validation,
        "waveform_fig":      waveform_fig,
        "mel_fig":           mel_fig,
        "y":                 y,
        "sr":                sr,
    }
