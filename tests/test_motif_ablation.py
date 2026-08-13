"""Unit tests for scripts/motif_ablation.py."""

import numpy as np
import pytest

from scripts.motif_ablation import draw_dose, matched_control_spans, shuffle_spans


def test_shuffle_keeps_length_and_composition():
    sequence = "AAAACCCCGGGGTTTT"
    shuffled = shuffle_spans(sequence, [(0, 8)], seed=1)
    assert len(shuffled) == len(sequence)
    assert sorted(shuffled) == sorted(sequence)


def test_shuffle_touches_only_the_named_spans():
    sequence = "AAAACCCCGGGGTTTT"
    shuffled = shuffle_spans(sequence, [(4, 8)], seed=1)
    assert shuffled[:4] == "AAAA" and shuffled[8:] == "GGGGTTTT"


def test_shuffle_is_reproducible_under_a_seed():
    sequence = "ACGTACGTACGTACGT"
    assert shuffle_spans(sequence, [(0, 16)], seed=3) == shuffle_spans(sequence, [(0, 16)], seed=3)


def test_control_spans_match_the_widths_and_avoid_the_hits():
    hits = [(10, 20), (50, 61)]
    control = matched_control_spans(hits, length=500, seed=0)
    assert sorted(end - start for start, end in control) == [10, 11]
    for start, end in control:
        for hit_start, hit_end in hits:
            assert end <= hit_start or start >= hit_end


def test_control_spans_do_not_overlap_each_other():
    control = matched_control_spans([(0, 10)] * 5, length=200, seed=0)
    ordered = sorted(control)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))


def test_dose_draws_distinct_spans_and_zero_means_all():
    spans = [(i * 10, i * 10 + 5) for i in range(20)]
    drawn = draw_dose(spans, 6, np.random.default_rng(0))
    assert len(drawn) == 6 and len(set(drawn)) == 6
    assert set(drawn) <= set(spans)
    assert draw_dose(spans, 0, np.random.default_rng(0)) == spans


def test_dose_larger_than_the_pool_fails_loudly():
    with pytest.raises(ValueError, match="Cannot draw"):
        draw_dose([(0, 5)], 3, np.random.default_rng(0))


def test_control_placement_gives_up_rather_than_hanging():
    # No room left: every control draw should fail and return nothing.
    assert matched_control_spans([(0, 10)], length=10, seed=0) == []
