"""Unit tests for scripts/cpg_swap.py."""

import pytest

from scripts.cpg_swap import cpg_count, swap_dinucleotides


def test_swap_preserves_the_exact_base_composition():
    sequence = "AGCTGCAAGCTT"
    swapped, _ = swap_dinucleotides(sequence, "GC", count=3, seed=0)
    assert sorted(swapped) == sorted(sequence)
    assert len(swapped) == len(sequence)


def test_gc_swap_creates_cpgs_and_cg_swap_removes_them():
    sequence = "AAGCAAGCAAGCAA"
    raised, made = swap_dinucleotides(sequence, "GC", count=3, seed=0)
    assert made == 3
    assert cpg_count(raised) == cpg_count(sequence) + 3

    lowered, removed = swap_dinucleotides(raised, "CG", count=3, seed=0)
    assert removed == 3
    assert cpg_count(lowered) == cpg_count(sequence)


def test_swap_count_is_capped_by_availability():
    # Only two GC occurrences exist, so asking for ten yields two.
    _, made = swap_dinucleotides("AAGCAAGCAA", "GC", count=10, seed=0)
    assert made == 2


def test_overlapping_occurrences_are_not_swapped_twice():
    # "GCGCGC" has GC at 0, 2, 4 — none adjacent-overlapping, but "GCG" shares
    # bases between positions 0 and 1, so a swap at 0 must exclude 1.
    swapped, made = swap_dinucleotides("GCGCGC", "GC", count=10, seed=0)
    assert sorted(swapped) == sorted("GCGCGC")
    assert made <= 3


def test_absent_dinucleotide_is_a_no_op():
    swapped, made = swap_dinucleotides("AAAAAA", "GC", count=5, seed=0)
    assert swapped == "AAAAAA" and made == 0


def test_swap_is_reproducible_under_a_seed():
    sequence = "AGCTGCAAGCTTGCGC"
    assert swap_dinucleotides(sequence, "GC", 2, seed=4) == swap_dinucleotides(sequence, "GC", 2, seed=4)


def test_identical_bases_are_rejected():
    with pytest.raises(ValueError, match="two distinct bases"):
        swap_dinucleotides("AACC", "AA", count=1, seed=0)
