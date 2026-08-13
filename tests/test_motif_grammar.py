"""Unit tests for src/evaluation/motif_grammar.py."""

import numpy as np
import pytest

from src.evaluation.motif_grammar import MotifGrammar


def hit(tf, start, width=10):
    return {"tf": tf, "start": start, "end": start + width}


def test_rejects_non_positive_distances():
    with pytest.raises(ValueError):
        MotifGrammar(cluster_window=0)
    with pytest.raises(ValueError):
        MotifGrammar(pair_distance=0)


def test_max_homotypic_cluster_counts_same_factor_only():
    """Three SPI1 sites packed together beat two, and other factors do not count."""
    grammar = MotifGrammar(cluster_window=100)
    hits = [hit("SPI1", 0), hit("SPI1", 40), hit("SPI1", 80), hit("PAX5", 20), hit("PAX5", 60)]

    assert grammar.max_homotypic_cluster(hits) == 3


def test_max_homotypic_cluster_respects_the_window():
    """Sites spread wider than the window are not a cluster."""
    grammar = MotifGrammar(cluster_window=100)
    spread = [hit("SPI1", 0), hit("SPI1", 500), hit("SPI1", 900)]

    assert grammar.max_homotypic_cluster(spread) == 1


def test_max_homotypic_cluster_is_zero_without_hits():
    assert MotifGrammar().max_homotypic_cluster([]) == 0


def test_heterotypic_pairs_ignores_same_factor_pairs():
    """Only sites for different factors within pair_distance count."""
    grammar = MotifGrammar(pair_distance=50)
    hits = [hit("SPI1", 0), hit("SPI1", 20), hit("PAX5", 30), hit("EBF1", 600)]

    # SPI1@0-PAX5@30 and SPI1@20-PAX5@30 qualify; the SPI1 pair and the distant
    # EBF1 do not.
    assert grammar.heterotypic_pairs(hits) == 2


def test_distinct_factors():
    assert MotifGrammar.distinct_factors([hit("A", 0), hit("A", 5), hit("B", 9)]) == 2


def test_mean_distance_from_centre_detects_central_concentration():
    """Sites at the middle score near zero; sites at the edges score near L/2."""
    central = [hit("A", 495, 10)]
    edges = [hit("A", 0, 10), hit("A", 990, 10)]

    assert MotifGrammar.mean_distance_from_centre(central, 1000) < 5
    assert MotifGrammar.mean_distance_from_centre(edges, 1000) > 480


def test_mean_distance_from_centre_is_nan_without_hits():
    assert np.isnan(MotifGrammar.mean_distance_from_centre([], 1000))


def test_profile_returns_one_value_per_sequence():
    grammar = MotifGrammar()
    per_sequence = [[hit("A", 0), hit("B", 20)], [], [hit("A", 100)]]

    profile = grammar.profile(per_sequence, sequence_length=1000)

    assert set(profile) == {
        "max_homotypic_cluster", "distinct_factors", "heterotypic_pairs",
        "mean_distance_from_centre", "total_hits",
    }
    assert all(len(v) == 3 for v in profile.values())
    assert list(profile["total_hits"]) == [2.0, 0.0, 1.0]


def test_profile_rejects_empty_input():
    with pytest.raises(ValueError):
        MotifGrammar().profile([], sequence_length=1000)


def test_permutation_p_separates_clearly_different_sets():
    """A large real difference is significant; identical sets are not."""
    a = np.arange(50, dtype=float) + 100
    b = np.arange(50, dtype=float)

    assert MotifGrammar.permutation_p(a, b, n_permutations=1000) < 0.01
    assert MotifGrammar.permutation_p(a, a.copy(), n_permutations=1000) == pytest.approx(1.0)
