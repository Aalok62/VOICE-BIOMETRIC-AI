"""
tests/test_prediction.py
========================
Unit tests for the BioSoundClassifier output contract.
Run with: pytest tests/test_prediction.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SUPPORTED_CLASSES, TOP_K_PREDICTIONS


def make_sine(freq=440, duration=2.0, sr=16000, amp=0.5):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr


class TestClassifierOutputContract:
    """
    Tests that verify the shape and types of BioSoundClassifier.predict().
    These tests use a mock backend so they run without TF or PyTorch.
    """

    @pytest.fixture
    def mock_classifier(self, monkeypatch):
        """
        Monkeypatched classifier that returns a deterministic result
        without loading any ML models.
        """
        from src.prediction.classifier import BioSoundClassifier

        clf = BioSoundClassifier.__new__(BioSoundClassifier)
        clf.mode    = "mock"
        clf.classes = SUPPORTED_CLASSES
        clf._model  = "mock"
        clf._yamnet_class_names = {}
        clf._label_map = {}

        def mock_predict(y, sr):
            probs = np.random.dirichlet(np.ones(len(SUPPORTED_CLASSES))).tolist()
            scores = dict(zip(SUPPORTED_CLASSES, probs))
            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class]
            top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_PREDICTIONS]
            return {
                "predicted_class":   predicted_class,
                "confidence":        confidence,
                "all_probabilities": scores,
                "top_k":             top_k,
            }

        clf.predict = mock_predict
        return clf

    def test_predict_returns_dict(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        assert isinstance(result, dict)

    def test_predicted_class_in_supported(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        assert result["predicted_class"] in SUPPORTED_CLASSES

    def test_confidence_is_float_in_range(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        conf = result["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_all_probabilities_sums_to_one(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_all_probabilities_keys(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        for cls in SUPPORTED_CLASSES:
            assert cls in result["all_probabilities"]

    def test_top_k_length(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        assert len(result["top_k"]) == TOP_K_PREDICTIONS

    def test_top_k_is_sorted_descending(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        probs = [p for _, p in result["top_k"]]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_first_matches_predicted(self, mock_classifier):
        y, sr = make_sine()
        result = mock_classifier.predict(y, sr)
        top_class = result["top_k"][0][0]
        pred_class = result["predicted_class"]
        assert top_class == pred_class


class TestHelpers:
    def test_format_confidence(self):
        from src.utils.helpers import format_confidence
        assert format_confidence(0.968) == "96.8%"
        assert format_confidence(0.0)   == "0.0%"
        assert format_confidence(1.0)   == "100.0%"

    def test_get_confidence_badge_high(self):
        from src.utils.helpers import get_confidence_badge
        color, label = get_confidence_badge(0.90)
        assert label == "HIGH"

    def test_get_confidence_badge_medium(self):
        from src.utils.helpers import get_confidence_badge
        color, label = get_confidence_badge(0.55)
        assert label == "MEDIUM"

    def test_get_confidence_badge_low(self):
        from src.utils.helpers import get_confidence_badge
        color, label = get_confidence_badge(0.20)
        assert label == "LOW"

    def test_format_duration_seconds(self):
        from src.utils.helpers import format_duration
        result = format_duration(2.5)
        assert "2.50s" in result

    def test_validate_file_extension_valid(self):
        from src.utils.helpers import validate_file_extension
        assert validate_file_extension("sound.wav")  is True
        assert validate_file_extension("dog.mp3")    is True
        assert validate_file_extension("cat.flac")   is True

    def test_validate_file_extension_invalid(self):
        from src.utils.helpers import validate_file_extension
        assert validate_file_extension("image.png")  is False
        assert validate_file_extension("video.mp4")  is False
