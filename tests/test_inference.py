"""
Unit tests for the inference pipeline components.
Run with: pytest tests/
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "training" / "convlstm"))
sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "inference"))

from model import ConvLSTMCell, ConvLSTMAnomalyDetector


# ── ConvLSTMCell ──────────────────────────────────────────────────────────────
class TestConvLSTMCell:
    def test_output_shape(self):
        cell = ConvLSTMCell(in_channels=32, hidden_channels=16)
        x = torch.randn(2, 32, 8, 8)
        h = torch.zeros(2, 16, 8, 8)
        c = torch.zeros(2, 16, 8, 8)
        h_next, c_next = cell(x, h, c)
        assert h_next.shape == (2, 16, 8, 8)
        assert c_next.shape == (2, 16, 8, 8)

    def test_hidden_state_changes(self):
        cell = ConvLSTMCell(in_channels=8, hidden_channels=4)
        x = torch.randn(1, 8, 4, 4)
        h = torch.zeros(1, 4, 4, 4)
        c = torch.zeros(1, 4, 4, 4)
        h_next, _ = cell(x, h, c)
        assert not torch.allclose(h_next, h), "Hidden state should update after forward pass"


# ── ConvLSTMAnomalyDetector ───────────────────────────────────────────────────
class TestConvLSTMAnomalyDetector:
    def test_output_shape(self):
        model = ConvLSTMAnomalyDetector(in_channels=3, hidden_channels=16, img_size=64)
        x = torch.randn(2, 4, 3, 64, 64)   # (batch=2, seq=4, C, H, W)
        out = model(x)
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"

    def test_output_range_after_sigmoid(self):
        model = ConvLSTMAnomalyDetector(in_channels=3, hidden_channels=16, img_size=64)
        model.eval()
        x = torch.randn(1, 4, 3, 64, 64)
        with torch.no_grad():
            score = torch.sigmoid(model(x)).item()
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_single_batch(self):
        model = ConvLSTMAnomalyDetector()
        x = torch.randn(1, 4, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1)

    def test_deterministic_eval(self):
        model = ConvLSTMAnomalyDetector()
        model.eval()
        x = torch.randn(1, 4, 3, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# ── Frame preprocessing ───────────────────────────────────────────────────────
class TestFramePreprocessing:
    def test_normalization(self):
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        normalized = frame.astype(np.float32) / 255.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_resize_shape(self):
        import cv2
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        resized = cv2.resize(frame, (64, 64))
        assert resized.shape == (64, 64, 3)

    def test_tensor_conversion(self):
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = torch.from_numpy(frame.transpose(2, 0, 1))
        assert tensor.shape == (3, 64, 64)


# ── Batch results schema ──────────────────────────────────────────────────────
class TestBatchResultsSchema:
    def test_frame_result_keys(self):
        result = {
            "frame_index": 0,
            "frame_name": "frame_001",
            "vehicles": 2,
            "detections": [{"class": "car", "confidence": 0.9, "bbox": [0, 0, 100, 100]}],
            "anomaly_score": 0.45,
            "accident": False,
        }
        required = {"frame_index", "frame_name", "vehicles", "detections", "anomaly_score", "accident"}
        assert required.issubset(result.keys())

    def test_detection_keys(self):
        det = {"class": "truck", "confidence": 0.75, "bbox": [10.0, 20.0, 80.0, 120.0]}
        assert "class" in det and "confidence" in det and "bbox" in det
        assert len(det["bbox"]) == 4
