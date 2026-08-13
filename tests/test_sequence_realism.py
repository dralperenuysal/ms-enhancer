"""Unit tests for src/evaluation/sequence_realism.py."""

import random

import pytest

from src.evaluation.sequence_realism import SequenceRealism


@pytest.fixture
def analyzer():
    return SequenceRealism()


def test_gc_content(analyzer):
    """GC fraction counts G and C across all sequences."""
    assert analyzer.gc_content(["GGCC", "AATT"]) == pytest.approx(0.5)
    assert analyzer.gc_content(["AAAA"]) == 0.0


def test_kmer_entropy_is_maximal_for_uniform_kmers(analyzer):
    """All 16 dinucleotides equally frequent gives the 4-bit maximum."""
    every_dinucleotide = "".join(a + b for a in "ACGT" for b in "ACGT")
    # Repeat so counts even out; entropy of a near-uniform distribution ~ 4 bits.
    entropy = analyzer.kmer_entropy([every_dinucleotide * 50], k=2)

    assert 3.9 < entropy <= 4.0


def test_kmer_entropy_is_zero_for_a_single_repeated_kmer(analyzer):
    """A homopolymer contains one distinct k-mer, so entropy is zero."""
    assert analyzer.kmer_entropy(["A" * 100], k=2) == pytest.approx(0.0)


def test_cpg_observed_expected_detects_depletion(analyzer):
    """CpG-free sequence scores 0; CpG-rich scores well above 1."""
    depleted = ["AGCTAGCTAGCT" * 20]  # contains C and G but no CG dinucleotide
    assert analyzer.cpg_observed_expected(depleted) == pytest.approx(0.0)

    enriched = ["CG" * 200]
    assert analyzer.cpg_observed_expected(enriched) > 1.5


def test_cpg_observed_expected_near_one_for_random(analyzer):
    """Random sequence has CpG at the frequency composition predicts."""
    control = analyzer.random_control(200, 500, seed=3)

    assert 0.9 < analyzer.cpg_observed_expected(control) < 1.1


def test_median_longest_run_catches_homopolymers(analyzer):
    """The argmax decoding artefact shows up as a long single-base run."""
    assert analyzer.median_longest_run(["A" * 50 + "CGCGCG"]) == 50.0
    assert analyzer.median_longest_run(["ACGT" * 25]) == 1.0


def test_compare_scores_reference_as_one_and_random_as_zero(analyzer):
    """A set identical to the reference scores 1.0; a random set scores ~0."""
    rng = random.Random(11)
    # Reference with strong CpG depletion, like real vertebrate sequence.
    reference = ["".join(rng.choice("ACGT") for _ in range(400)).replace("CG", "CA") for _ in range(100)]

    same = analyzer.compare(reference, reference)
    assert same["realism"]["cpg_observed_expected"] == pytest.approx(1.0, abs=1e-9)

    control = analyzer.random_control(100, 400, seed=5)
    versus_random = analyzer.compare(control, reference)
    assert abs(versus_random["realism"]["cpg_observed_expected"]) < 0.2


def test_compare_rejects_empty_input(analyzer):
    """Comparing against nothing is a usage error, not an empty report."""
    with pytest.raises(ValueError, match="non-empty"):
        analyzer.compare([], ["ACGT"])
