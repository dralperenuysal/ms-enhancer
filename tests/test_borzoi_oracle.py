"""Unit tests for src/evaluation/borzoi_oracle.py.

The tests that matter here are the axis-order ones. Borzoi takes (batch, 4, L)
and returns (batch, tracks, bins), both the transpose of what enformer-pytorch
does, and getting either wrong yields a plausible number rather than an error.
"""

import numpy as np
import pytest
import torch

from src.evaluation.borzoi_oracle import BorzoiOracle
from src.evaluation.enformer_oracle import EnformerOracle


def test_borzoi_reads_its_own_config_section():
    """Each oracle takes its tracks from its own section, not a shared one."""
    borzoi, enformer = BorzoiOracle(), EnformerOracle()

    assert borzoi.context_length == 524288
    assert enformer.context_length == 196608
    assert borzoi.n_tracks == 7611
    assert borzoi.target_tracks_by_cell_type != enformer.target_tracks_by_cell_type


def test_borzoi_feeds_the_model_channels_first():
    """A (length, 4) tensor would be read as a 4 bp sequence with 524288 channels."""
    oracle = BorzoiOracle()
    seen = {}

    class FakeBorzoi:
        def __call__(self, x):
            seen["shape"] = tuple(x.shape)
            return torch.zeros(1, oracle.n_tracks, 16)

    oracle.model = FakeBorzoi()
    oracle._predict_per_bin("ACGT" * 25)

    assert seen["shape"] == (1, 4, 100)


def test_borzoi_returns_bins_first_not_tracks_first():
    """Borzoi emits (tracks, bins); the base class needs (bins, tracks) from both oracles."""
    oracle = BorzoiOracle()
    bins = 16

    class FakeBorzoi:
        def __call__(self, x):
            # Track t is constant at value t, so a correct transpose puts
            # arange(n_tracks) along the second axis, not the first.
            return torch.arange(oracle.n_tracks, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).repeat(1, 1, bins)

    oracle.model = FakeBorzoi()
    per_bin = oracle._predict_per_bin("ACGT" * 25)

    assert per_bin.shape == (bins, oracle.n_tracks)
    assert np.allclose(per_bin.mean(axis=0), np.arange(oracle.n_tracks))


def test_borzoi_mssi_uses_the_configured_tracks(monkeypatch):
    """MSSI is mean(target) - mean(background) on this oracle's own indices."""
    oracle = BorzoiOracle()
    cell_type = next(iter(oracle.target_tracks_by_cell_type))
    targets = oracle.target_tracks_by_cell_type[cell_type]

    signal = np.zeros(oracle.n_tracks, dtype=np.float32)
    signal[targets] = 4.0
    signal[oracle.background_tracks] = 1.0

    per_bin = np.repeat(signal[None, :], oracle.n_bins, axis=0)
    monkeypatch.setattr(oracle, "_predict_per_bin", lambda seq: per_bin)
    monkeypatch.setattr(oracle, "construct_context_window", lambda *a, **k: "A" * 10)

    result = oracle.predict_cell_specificity("ACGT" * 250, cell_type=cell_type, flanking_seq="A" * 10)

    assert result["mssi_score"] == pytest.approx(3.0)
    # Uniform across bins, so the insert-localised reduction must agree.
    assert result["mssi_local"] == pytest.approx(3.0)


def test_borzoi_rejects_unknown_cell_type():
    oracle = BorzoiOracle()

    with pytest.raises(KeyError, match="No Enformer target tracks|target tracks configured"):
        oracle.predict_cell_specificity("ACGT", cell_type="hepatocyte", flanking_seq="A")


def test_base_class_refuses_to_score_without_a_model():
    """TrackOracle itself has no forward pass and must say so rather than guess."""
    from src.evaluation.track_oracle import TrackOracle

    class Bare(TrackOracle):
        CONFIG_SECTION = "borzoi"

    with pytest.raises(NotImplementedError):
        Bare()._predict_per_bin("ACGT")


def test_insert_bin_slice_centres_on_the_insert():
    """The localised reduction is worthless if it reads the wrong bins."""
    enformer = EnformerOracle()
    borzoi = BorzoiOracle()

    for oracle in (enformer, borzoi):
        lo, hi = oracle.insert_bin_slice(1000)
        # The slice must straddle the centre bin, since the insert is centred.
        assert lo <= oracle.n_bins // 2 < hi
        # 1000 bp cannot need more than one extra bin at each edge.
        expected = -(-1000 // oracle.bin_size)
        assert expected <= hi - lo <= expected + 2
        # And it must be a small fraction of the whole window, which is the
        # entire point of computing it separately.
        assert (hi - lo) < oracle.n_bins // 10

    # Degenerate lengths stay inside the bin axis and never go empty.
    for length in (1, 64, 128, 100000, enformer.context_length):
        lo, hi = enformer.insert_bin_slice(length)
        assert 0 <= lo < hi <= enformer.n_bins
