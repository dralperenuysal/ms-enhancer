"""Unit tests for src/evaluation/markov_baseline.py."""

import pytest

from src.evaluation.markov_baseline import MarkovBaseline
from src.evaluation.sequence_realism import SequenceRealism


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        MarkovBaseline(order=-1)
    with pytest.raises(ValueError):
        MarkovBaseline(order=2, pseudocount=0.0)


def test_sample_before_fit_raises():
    with pytest.raises(RuntimeError):
        MarkovBaseline(order=2).sample(num_sequences=1, length=10, seed=0)


def test_order_zero_reproduces_base_composition():
    """A zeroth-order chain matches GC content but carries no dinucleotide structure."""
    training = ["GCGCGCGCGC" * 20] * 10
    chain = MarkovBaseline(order=0).fit(training)

    sampled = chain.sample(num_sequences=20, length=200, seed=0)

    gc = SequenceRealism.gc_content(sampled)
    assert gc > 0.9, f"order-0 chain should keep the GC-rich composition, got {gc}"


def test_order_one_reproduces_dinucleotide_structure():
    """An order-1 chain learns that C is always followed by G here."""
    # ACGT tiles seamlessly, so C is followed by G at every position including
    # the repeat junctions. A unit that did not tile would teach the chain a
    # junction transition that says nothing about the intended structure.
    chain = MarkovBaseline(order=1).fit(["ACGT" * 50] * 10)

    sampled = chain.sample(num_sequences=10, length=200, seed=0)

    # Every C in the training data is followed by G, so CG must be frequent.
    joined = "".join(sampled)
    assert joined.count("CG") / max(joined.count("C"), 1) > 0.9


def test_deterministic_under_a_fixed_seed():
    chain = MarkovBaseline(order=2).fit(["ACGTTGCAACGT" * 20] * 5)

    assert chain.sample(3, 60, seed=7) == chain.sample(3, 60, seed=7)
    assert chain.sample(3, 60, seed=7) != chain.sample(3, 60, seed=8)


def test_fit_requires_usable_sequences():
    with pytest.raises(ValueError):
        MarkovBaseline(order=6).fit(["NNNNNNNNNNNN"])


def test_sampled_sequences_have_requested_shape():
    chain = MarkovBaseline(order=3).fit(["ACGTACGTTTGCA" * 20] * 5)
    sampled = chain.sample(num_sequences=4, length=150, seed=1)

    assert len(sampled) == 4
    assert all(len(s) == 150 for s in sampled)
    assert set("".join(sampled)) <= set("ACGT")
