"""Unit tests for scripts/compare_selected_grammar.py."""

import json

import pytest

from scripts.compare_selected_grammar import benjamini_hochberg, gc_fraction, split_tails


def test_bh_is_monotone_and_never_below_the_raw_p():
    p_values = [0.001, 0.008, 0.02, 0.04, 0.9]
    q_values = benjamini_hochberg(p_values)
    assert all(q >= p for p, q in zip(p_values, q_values))
    assert q_values == sorted(q_values)


def test_bh_scales_the_smallest_p_by_the_number_of_tests():
    # Only the first is significant: q = p * m / rank = 0.001 * 4 / 1.
    assert benjamini_hochberg([0.001, 0.5, 0.6, 0.7])[0] == pytest.approx(0.004)


def test_bh_is_reachable_where_bonferroni_is_not():
    # 726 tests, permutation floor 1e-4: Bonferroni (6.9e-5) is unreachable but
    # a q-value below 0.05 is not, which is the reason for using BH here.
    p_values = [1e-4] * 26 + [0.5] * 700
    q_values = benjamini_hochberg(p_values)
    assert q_values[0] < 0.05 and q_values[25] < 0.05


def test_bh_rejects_unsorted_input():
    with pytest.raises(ValueError, match="sorted ascending"):
        benjamini_hochberg([0.5, 0.01])


def test_bh_rejects_an_empty_test_set():
    with pytest.raises(ValueError, match="empty"):
        benjamini_hochberg([])


def test_gc_fraction_counts_both_strandwise_bases():
    assert list(gc_fraction(["GGCC", "ATAT", "ACGT"])) == [1.0, 0.0, 0.5]


@pytest.fixture
def scored_locus(tmp_path):
    """Four candidates plus the real element, with known MSSI ordering."""
    fasta = tmp_path / "sel.fasta"
    fasta.write_text(">real\nAAAA\n>c1\nCCCC\n>c2\nGGGG\n>c3\nTTTT\n>c4\nACGT\n")
    report = tmp_path / "sel.json"
    report.write_text(json.dumps({"sequences": {
        "real": {"mssi_score": 9.0}, "c1": {"mssi_score": 0.4}, "c2": {"mssi_score": 0.3},
        "c3": {"mssi_score": 0.2}, "c4": {"mssi_score": 0.1},
    }}))
    return report, fasta


def test_tails_exclude_the_real_element(scored_locus):
    report, fasta = scored_locus
    top, bottom = split_tails(report, fasta, fraction=0.5)
    # 'real' scores highest but is not a generated candidate, so it must not appear.
    assert "AAAA" not in top and "AAAA" not in bottom
    assert top == ["CCCC", "GGGG"] and bottom == ["TTTT", "ACGT"]


def test_fraction_rounding_to_an_empty_tail_fails_loudly(scored_locus):
    report, fasta = scored_locus
    with pytest.raises(ValueError, match="rounds to zero"):
        split_tails(report, fasta, fraction=0.1)


def test_out_of_range_fraction_fails_loudly(scored_locus):
    report, fasta = scored_locus
    with pytest.raises(ValueError, match="fraction"):
        split_tails(report, fasta, fraction=0.8)
