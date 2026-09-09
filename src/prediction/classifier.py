"""
src/prediction/classifier.py
=============================
BioSoundClassifier – abstraction layer that routes predictions to either:
  - Audio Spectrogram Transformer (AST) via PyTorch / Hugging Face (default, zero-setup)
  - Custom BioSoundCNN (requires prior training)

Prediction output is always normalised to our SUPPORTED_CLASSES.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SUPPORTED_CLASSES, AUDIOSET_CLASS_MAP, AST_MODEL_ID,
    CUSTOM_MODEL_PATH, LABEL_MAP_PATH,
    DEFAULT_MODEL, TARGET_SAMPLE_RATE,
    TOP_K_PREDICTIONS, N_MELS,
)


class BioSoundClassifier:
    """
    Unified classifier interface.

    Parameters
    ----------
    mode : "ast" | "custom" | "yamnet"
        Which backend to use. Defaults to config.DEFAULT_MODEL ("ast").
    load_immediately : bool
        If True, loads model weights immediately upon instantiation.
    """

    def __init__(self, mode: str = DEFAULT_MODEL, load_immediately: bool = True):
        self.mode       = mode
        self._model     = None
        self._extractor = None
        self.classes    = SUPPORTED_CLASSES
        self._id2label  = None
        self._label_map = None

        if load_immediately:
            self.load_model()

    def load_model(self):
        """Pre-load model weights into memory."""
        if self.mode in ("ast", "default", "yamnet"):
            self._load_ast()
        elif self.mode == "custom":
            self._load_custom()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, y: np.ndarray, sr: int) -> dict:
        """
        Classify audio waveform.

        Parameters
        ----------
        y  : np.ndarray  mono waveform float32
        sr : int         sample rate (should be TARGET_SAMPLE_RATE)

        Returns
        -------
        {
            "predicted_class"   : str,
            "confidence"        : float,
            "all_probabilities" : {class_name: float},
            "top_k"             : [(class_name, float), ...],
        }
        """
        if self.mode in ("ast", "default", "yamnet"):
            return self._predict_ast(y, sr)
        elif self.mode == "custom":
            return self._predict_custom(y, sr)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'ast' or 'custom'.")

    # ------------------------------------------------------------------ #
    # AST Backend (PyTorch + Hugging Face)
    # ------------------------------------------------------------------ #

    def _load_ast(self):
        """Lazy-load AST model and feature extractor from Hugging Face."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            print(f"[BioSoundClassifier] Loading AST model ({AST_MODEL_ID})...")
            self._extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
            self._model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID)
            self._model.eval()
            self._id2label = self._model.config.id2label
            print("[BioSoundClassifier] AST model loaded successfully.")
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for AST mode. "
                "Install with: pip install transformers torch"
            ) from e

    def _predict_ast(self, y: np.ndarray, sr: int) -> dict:
        """Run inference with AST and map AudioSet classes to our living-being classes."""
        import torch

        self._load_ast()

        inputs = self._extractor(y, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        our_scores = {cls: 0.0 for cls in SUPPORTED_CLASSES}

        for idx, prob in enumerate(probs):
            audioset_label = self._id2label.get(idx, "")
            our_cls = AUDIOSET_CLASS_MAP.get(audioset_label, None)
            if our_cls and our_cls in our_scores:
                our_scores[our_cls] += float(prob)

        total = sum(our_scores.values())
        if total > 0:
            our_scores = {k: v / total for k, v in our_scores.items()}
        else:
            our_scores = {k: 1.0 / len(SUPPORTED_CLASSES) for k in SUPPORTED_CLASSES}

        predicted_class = max(our_scores, key=our_scores.get)
        confidence      = our_scores[predicted_class]

        top_k = sorted(our_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_PREDICTIONS]

        return {
            "predicted_class":   predicted_class,
            "confidence":        round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in our_scores.items()},
            "top_k":             [(k, round(v, 4)) for k, v in top_k],
        }

    # ------------------------------------------------------------------ #
    # Custom CNN Backend
    # ------------------------------------------------------------------ #

    def _load_custom(self):
        """Lazy-load the trained BioSoundCNN checkpoint."""
        if self._model is not None:
            return
        try:
            import torch
            from src.model.architecture import BioSoundCNN
        except ImportError as e:
            raise ImportError("PyTorch is required for custom model. pip install torch") from e

        if not os.path.exists(CUSTOM_MODEL_PATH):
            raise FileNotFoundError(
                f"Custom model not found at {CUSTOM_MODEL_PATH}. "
                "Train first with: python src/model/train.py"
            )

        checkpoint = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")
        self.classes = checkpoint["classes"]
        n_mels   = checkpoint.get("n_mels", N_MELS)
        n_frames = checkpoint.get("n_frames", 128)

        self._model = BioSoundCNN(n_classes=len(self.classes), n_mels=n_mels, n_frames=n_frames)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        print(f"[BioSoundClassifier] Custom CNN loaded (val_acc={checkpoint.get('val_acc', 'N/A'):.4f})")

    def _predict_custom(self, y: np.ndarray, sr: int) -> dict:
        """Run inference with the custom BioSoundCNN."""
        import torch
        from src.audio.features import extract_mel_spectrogram, mel_to_fixed_size

        self._load_custom()

        mel = extract_mel_spectrogram(y, sr)
        mel = mel_to_fixed_size(mel, target_frames=128)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            probs = self._model.predict_proba(tensor)[0].numpy()

        our_scores = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

        for cls in SUPPORTED_CLASSES:
            if cls not in our_scores:
                our_scores[cls] = 0.0

        predicted_class = max(our_scores, key=our_scores.get)
        confidence      = our_scores[predicted_class]
        top_k = sorted(our_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_PREDICTIONS]

        return {
            "predicted_class":   predicted_class,
            "confidence":        round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in our_scores.items()},
            "top_k":             [(k, round(v, 4)) for k, v in top_k],
        }
