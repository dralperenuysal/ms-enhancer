"""Unit tests for scripts/attribution_scan.py.

Enformer itself is never loaded: a stub model returns a differentiable function
of the input, which is enough to check that the attribution is taken with
respect to MSSI, restricted to the insert, and one value per base.
"""

import numpy as np
import pytest
import torch

from scripts.attribution_scan import attribute


class _StubOracle:
    """Minimal stand-in exposing the attributes ``attribute`` uses."""

    device = "cpu"
    background_tracks = [1]

    def __init__(self, weights):
        self.weights = weights

    def _load_model(self):
        return None

    def _one_hot_encode(self, sequence):
        index = {"A": 0, "C": 1, "G": 2, "T": 3}
        one_hot = torch.zeros(len(sequence), 4)
        for position, base in enumerate(sequence):
            one_hot[position, index[base]] = 1.0
        return one_hot

    def model(self, one_hot):
        # (1, bins, tracks): each track is a weighted sum over the input, so the
        # gradient is non-zero exactly where the weights are.
        pooled = (one_hot.squeeze(0) * self.weights).sum()
        return torch.stack([pooled, torch.zeros_like(pooled)]).reshape(1, 1, 2)


@pytest.fixture
def sequence():
    return "ACGT" * 50  # 200 bp, of which the middle 100 is the "insert"


def test_attribution_has_one_value_per_insert_base(sequence):
    oracle = _StubOracle(torch.ones(len(sequence), 4))
    values = attribute(oracle, sequence, target_tracks=[0], insert_start=50, insert_length=100)
    assert values.shape == (100,)


def test_attribution_is_restricted_to_the_insert(sequence):
    # Weight only bases 0-49, which lie outside the requested insert window.
    weights = torch.zeros(len(sequence), 4)
    weights[:50] = 1.0
    oracle = _StubOracle(weights)
    values = attribute(oracle, sequence, target_tracks=[0], insert_start=50, insert_length=100)
    assert np.allclose(values, 0.0)


def test_attribution_follows_the_weighted_positions(sequence):
    weights = torch.zeros(len(sequence), 4)
    weights[70] = 3.0
    oracle = _StubOracle(weights)
    values = attribute(oracle, sequence, target_tracks=[0], insert_start=50, insert_length=100)
    assert int(np.argmax(np.abs(values))) == 20  # 70 - 50
    assert values[20] == pytest.approx(3.0)


def test_background_tracks_are_subtracted(sequence):
    # Target and background are the same track, so MSSI is identically zero and
    # every attribution must vanish.
    oracle = _StubOracle(torch.ones(len(sequence), 4))
    oracle.background_tracks = [0]
    values = attribute(oracle, sequence, target_tracks=[0], insert_start=50, insert_length=100)
    assert np.allclose(values, 0.0)
