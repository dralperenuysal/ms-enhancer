"""Unit tests for scripts/mpra_scoring_set.py."""

import pandas as pd
import pytest

from scripts.mpra_scoring_set import stratified_sample


@pytest.fixture
def measurements():
    return pd.DataFrame({"ID": [f"e{i}" for i in range(100)], "spec": range(100)})


def test_strata_have_the_requested_sizes(measurements):
    sample = stratified_sample(measurements, "spec", n_random=20, n_extreme=5, seed=0)
    assert sample.stratum.value_counts().to_dict() == {"random": 20, "top": 5, "bottom": 5}


def test_extremes_are_the_actual_extremes(measurements):
    sample = stratified_sample(measurements, "spec", n_random=20, n_extreme=5, seed=0)
    assert sorted(sample[sample.stratum == "top"].spec) == [95, 96, 97, 98, 99]
    assert sorted(sample[sample.stratum == "bottom"].spec) == [0, 1, 2, 3, 4]


def test_random_stratum_excludes_the_extremes(measurements):
    sample = stratified_sample(measurements, "spec", n_random=20, n_extreme=5, seed=0)
    random = sample[sample.stratum == "random"]
    # The random subset must stay unbiased, so it cannot reuse the tails.
    assert random.spec.between(5, 94).all()


def test_no_element_appears_twice(measurements):
    sample = stratified_sample(measurements, "spec", n_random=40, n_extreme=10, seed=1)
    assert sample.ID.is_unique


def test_sampling_is_reproducible(measurements):
    a = stratified_sample(measurements, "spec", 20, 5, seed=7)
    b = stratified_sample(measurements, "spec", 20, 5, seed=7)
    assert list(a.ID) == list(b.ID)


def test_over_request_fails_loudly(measurements):
    with pytest.raises(ValueError, match="only 100"):
        stratified_sample(measurements, "spec", n_random=95, n_extreme=10, seed=0)
