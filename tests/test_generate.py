"""Unit tests for generate.py."""

import argparse

import pandas as pd
import pytest
import torch

from generate import assign_host_loci, build_condition_vector, sample_markov


@pytest.fixture
def dataset_file(tmp_path):
    """A processed dataset carrying cell types and normalization constants."""
    path = tmp_path / "processed.pt"
    torch.save(
        {
            "cell_types": ["CD4_T_cell", "B_cell", "microglia"],
            "normalization_stats": {
                "method": "zscore",
                "features": ["peak_score", "fold_enrichment"],
                "center": [200.0, 10.0],
                "scale": [50.0, 2.0],
            },
        },
        path,
    )
    return str(path)


def make_args(dataset, cell_type="B_cell", peak_score=None, fold_enrichment=None):
    return argparse.Namespace(
        dataset=dataset, cell_type=cell_type,
        peak_score=peak_score, fold_enrichment=fold_enrichment,
    )


def test_build_condition_vector_selects_requested_cell_type(dataset_file):
    """The one-hot slot follows --cell_type instead of being hardcoded to index 0."""
    cond = build_condition_vector(make_args(dataset_file, "microglia"), condition_dim=5)

    assert cond.tolist()[:3] == [0.0, 0.0, 1.0]


def test_build_condition_vector_normalizes_raw_units(dataset_file):
    """Raw signal values are z-scored with the training constants."""
    cond = build_condition_vector(
        make_args(dataset_file, "CD4_T_cell", peak_score=300.0, fold_enrichment=14.0),
        condition_dim=5,
    )

    assert cond[3].item() == pytest.approx((300.0 - 200.0) / 50.0)
    assert cond[4].item() == pytest.approx((14.0 - 10.0) / 2.0)


def test_build_condition_vector_defaults_to_training_mean(dataset_file):
    """Unspecified features sit at the training mean, i.e. zero after z-scoring."""
    cond = build_condition_vector(make_args(dataset_file), condition_dim=5)

    assert cond[3].item() == 0.0
    assert cond[4].item() == 0.0


def test_build_condition_vector_rejects_unknown_cell_type(dataset_file):
    """An unknown cell type fails instead of silently generating something else."""
    with pytest.raises(ValueError, match="Unknown cell type"):
        build_condition_vector(make_args(dataset_file, "hepatocyte"), condition_dim=5)


def test_build_condition_vector_rejects_width_mismatch(dataset_file):
    """A checkpoint that disagrees with the dataset is caught before sampling."""
    with pytest.raises(ValueError, match="condition_dim"):
        build_condition_vector(make_args(dataset_file), condition_dim=7)


def test_assign_host_loci_draws_from_requested_cell_type(tmp_path):
    """Host loci come from real windows of the same cell type, reproducibly."""
    path = tmp_path / "windows.csv"
    pd.DataFrame(
        [
            {"peak_id": "a", "chrom": "chr1", "start": 100, "end": 1100, "cell_type": "B_cell"},
            {"peak_id": "b", "chrom": "chr2", "start": 200, "end": 1200, "cell_type": "B_cell"},
            {"peak_id": "c", "chrom": "chr3", "start": 300, "end": 1300, "cell_type": "microglia"},
        ]
    ).to_csv(path, index=False)

    hosts = assign_host_loci(str(path), "B_cell", num_samples=5, seed=1)

    assert len(hosts) == 5
    assert set(hosts["cell_type"]) == {"B_cell"}
    assert set(hosts["chrom"]).issubset({"chr1", "chr2"})
    # Same seed, same assignment.
    assert hosts.equals(assign_host_loci(str(path), "B_cell", num_samples=5, seed=1))


def test_assign_host_loci_rejects_cell_type_with_no_windows(tmp_path):
    """A cell type absent from the window file is an error, not an empty result."""
    path = tmp_path / "windows.csv"
    pd.DataFrame(
        [{"peak_id": "a", "chrom": "chr1", "start": 100, "end": 1100, "cell_type": "B_cell"}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no windows for cell type"):
        assign_host_loci(str(path), "microglia", num_samples=2, seed=1)


def test_decode_by_sampling_follows_the_distribution():
    """Sampling respects the decoder's probabilities instead of collapsing to the mode."""
    from generate import decode_by_sampling
    from src.utils.helpers import seeded_generator

    # Every position: 70% A, 30% C. argmax would return 1000 A's.
    probs = torch.zeros(1, 4, 1000)
    probs[:, 0, :] = 0.7
    probs[:, 1, :] = 0.3

    seq = decode_by_sampling(probs, seeded_generator(0))[0]

    assert len(seq) == 1000
    assert set(seq) == {"A", "C"}
    assert 0.6 < seq.count("A") / 1000 < 0.8


def test_decode_by_sampling_is_reproducible():
    """The same generator seed yields the same sequences."""
    from generate import decode_by_sampling
    from src.utils.helpers import seeded_generator

    probs = torch.full((3, 4, 200), 0.25)

    first = decode_by_sampling(probs, seeded_generator(7))
    second = decode_by_sampling(probs, seeded_generator(7))

    assert first == second
    assert len(set(first)) == 3  # Distinct sequences within a batch.


@pytest.fixture
def markov_inputs(tmp_path):
    """Real-window FASTA and metadata with a distinguishable sequence per cell type."""
    fasta = tmp_path / "windows.fasta"
    fasta.write_text(
        "".join(f">b{i}\n{'ACGT' * 50}\n" for i in range(5))
        + "".join(f">m{i}\n{'GGCC' * 50}\n" for i in range(5))
    )
    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "peak_id": [f"b{i}" for i in range(5)] + [f"m{i}" for i in range(5)],
            "chrom": ["chr1"] * 10,
            "start": list(range(0, 10000, 1000)),
            "end": list(range(200, 10200, 1000)),
            "cell_type": ["B_cell"] * 5 + ["microglia"] * 5,
        }
    ).to_csv(metadata, index=False)
    return str(fasta), str(metadata)


def markov_args(markov_inputs, cell_type="microglia", order=2, num_samples=3):
    fasta, metadata = markov_inputs
    return argparse.Namespace(
        windows_fasta=fasta, host_loci=metadata, cell_type=cell_type,
        markov_order=order, num_samples=num_samples, seed=0,
        config="configs/model_config.yaml",
    )


def test_sample_markov_learns_only_the_requested_cell_type(markov_inputs):
    """The chain is conditioned by which windows it is fitted on, not by a vector."""
    def gc(sequences):
        joined = "".join(sequences)
        return (joined.count("G") + joined.count("C")) / len(joined)

    microglia = sample_markov(markov_args(markov_inputs, "microglia"), sequence_length=120)
    b_cell = sample_markov(markov_args(markov_inputs, "B_cell"), sequence_length=120)

    assert len(microglia) == 3
    assert all(len(s) == 120 for s in microglia)
    # The microglia windows are GC-only and the B_cell windows are half AT, so the
    # composition has to follow the cell type the chain was fitted on. Neither is
    # pure: pseudocount smoothing leaves every base a small probability, and an
    # unobserved context backs off all the way to uniform.
    assert gc(microglia) > 0.85
    assert gc(b_cell) < 0.65


def test_sample_markov_is_reproducible(markov_inputs):
    args = markov_args(markov_inputs, "B_cell")

    assert sample_markov(args, 120) == sample_markov(args, 120)


def test_sample_markov_rejects_unknown_cell_type(markov_inputs):
    with pytest.raises(ValueError, match="no windows for cell type"):
        sample_markov(markov_args(markov_inputs, "CD4_T_cell"), sequence_length=120)


def test_sample_markov_requires_the_real_windows(markov_inputs, tmp_path):
    args = markov_args(markov_inputs)
    args.windows_fasta = str(tmp_path / "absent.fasta")

    with pytest.raises(FileNotFoundError, match="build_dataset"):
        sample_markov(args, sequence_length=120)


def test_assign_host_loci_can_pin_every_candidate_to_one_locus(tmp_path):
    """Selection needs a fixed locus: the host explains far more MSSI than the insert."""
    path = tmp_path / "windows.csv"
    pd.DataFrame(
        [
            {"peak_id": "a", "chrom": "chr1", "start": 100, "end": 1100, "cell_type": "B_cell"},
            {"peak_id": "b", "chrom": "chr2", "start": 200, "end": 1200, "cell_type": "B_cell"},
        ]
    ).to_csv(path, index=False)

    hosts = assign_host_loci(str(path), "B_cell", num_samples=4, seed=1, locus_id="b")

    assert len(hosts) == 4
    assert set(hosts["chrom"]) == {"chr2"}


def test_assign_host_loci_rejects_a_locus_of_another_cell_type(tmp_path):
    path = tmp_path / "windows.csv"
    pd.DataFrame(
        [
            {"peak_id": "a", "chrom": "chr1", "start": 100, "end": 1100, "cell_type": "B_cell"},
            {"peak_id": "z", "chrom": "chr9", "start": 100, "end": 1100, "cell_type": "microglia"},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not a 'B_cell' window"):
        assign_host_loci(str(path), "B_cell", num_samples=2, seed=1, locus_id="z")
