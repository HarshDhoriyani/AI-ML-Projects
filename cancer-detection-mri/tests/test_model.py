"""
test_model.py
Basic sanity checks for model architectures (no dataset required).

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_cnn, build_transfer, compile_model  # noqa: E402


def test_cnn_output_shape():
    model = build_cnn(input_shape=(224, 224, 3), num_classes=4)
    dummy_input = np.random.rand(2, 224, 224, 3).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 4)
    # softmax outputs should sum to ~1
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_cnn_compiles_and_trains_one_step():
    model = build_cnn(input_shape=(64, 64, 3), num_classes=4)
    model = compile_model(model, learning_rate=1e-3)
    x = np.random.rand(4, 64, 64, 3).astype("float32")
    y = np.eye(4)[np.random.randint(0, 4, size=4)]
    history = model.fit(x, y, epochs=1, verbose=0)
    assert "loss" in history.history


def test_transfer_output_shape():
    # weights=None avoids downloading ImageNet weights, so this test runs offline.
    model = build_transfer(input_shape=(224, 224, 3), num_classes=4, fine_tune_at=None, weights=None)
    dummy_input = np.random.rand(1, 224, 224, 3).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 4)
