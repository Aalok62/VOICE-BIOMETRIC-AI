"""
tests/test_model.py
===================
Unit tests for BioSoundCNN architecture.
Run with: pytest tests/test_model.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBioSoundCNN:
    """
    Architecture-level tests that verify tensor shapes.
    Runs without GPU.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_forward_pass_shape(self):
        import torch
        from src.model.architecture import BioSoundCNN

        model = BioSoundCNN(n_classes=8)
        x = torch.zeros(4, 1, 128, 128)  # batch=4, C=1, H=128, W=128
        out = model(x)
        assert out.shape == (4, 8)

    def test_single_sample_shape(self):
        import torch
        from src.model.architecture import BioSoundCNN

        model = BioSoundCNN(n_classes=8)
        x = torch.zeros(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 8)

    def test_different_n_classes(self):
        import torch
        from src.model.architecture import BioSoundCNN

        for n in [2, 4, 8, 16]:
            model = BioSoundCNN(n_classes=n)
            x = torch.zeros(2, 1, 128, 128)
            out = model(x)
            assert out.shape == (2, n)

    def test_log_softmax_output(self):
        import torch
        from src.model.architecture import BioSoundCNN

        model = BioSoundCNN(n_classes=8)
        x = torch.zeros(2, 1, 128, 128)
        out = model(x)
        # log-softmax outputs should be <= 0
        assert (out <= 0).all()

    def test_predict_proba_sums_to_one(self):
        import torch
        from src.model.architecture import BioSoundCNN

        model = BioSoundCNN(n_classes=8)
        x = torch.zeros(3, 1, 128, 128)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-4)

    def test_parameter_count_positive(self):
        from src.model.architecture import BioSoundCNN, count_parameters
        model = BioSoundCNN(n_classes=8)
        count = count_parameters(model)
        assert count > 0
        print(f"\nTotal parameters: {count:,}")

    def test_build_model_helper(self):
        import torch
        from src.model.architecture import build_model
        model = build_model(n_classes=8, device="cpu")
        x = torch.zeros(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 8)
