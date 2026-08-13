"""Unit tests for scripts/locus_survey.py."""

import numpy as np
import pandas as pd
import pytest

from scripts.locus_survey import choose_loci, cpg_observed_expected


@pytest.fixture
def windows():
    return pd.DataFrame({
        "peak_id": [f"p{i}" for i in range(100)],
        "peak_score": np.arange(100, dtype=float),
        "chrom": "chr1",
    })


def test_cpg_oe_separates_cpg_rich_from_cpg_poor_at_identical_composition():
    # Both have 50 C and 50 G; only the arrangement differs, which is exactly
    # what o/e is meant to detect.
    assert cpg_observed_expected("CG" * 50) > 1.5
    assert cpg_observed_expected("C" * 50 + "G" * 50) < 0.1


def test_cpg_oe_is_nan_without_both_bases():
    assert np.isnan(cpg_observed_expected("AAAA"))
    assert np.isnan(cpg_observed_expected("CCCC"))


def test_loci_are_spread_across_the_score_range(windows):
    chosen = choose_loci(windows, n_loci=10, seed=0)
    assert len(chosen) == 10
    # One pick per decile of the score range, so the spread covers the extremes.
    assert chosen.peak_score.min() < 10 and chosen.peak_score.max() >= 90


def test_locus_choice_is_reproducible(windows):
    a = choose_loci(windows, n_loci=8, seed=3)
    b = choose_loci(windows, n_loci=8, seed=3)
    assert list(a.peak_id) == list(b.peak_id)
    assert list(choose_loci(windows, 8, seed=4).peak_id) != list(a.peak_id)


def test_requesting_more_loci_than_exist_fails_loudly(windows):
    with pytest.raises(ValueError, match="only 100 windows"):
        choose_loci(windows, n_loci=200, seed=0)
