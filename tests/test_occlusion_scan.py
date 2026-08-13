"""Unit tests for scripts/occlusion_scan.py."""

import json

import pytest

from scripts.occlusion_scan import load_ranked_sequences, occlude, read_fasta


@pytest.fixture
def scored_set(tmp_path):
    """A two-sequence FASTA with a matching Enformer-style score report."""
    fasta = tmp_path / "sel.fasta"
    fasta.write_text(">low\nAAAACCCC\n>high\nGGGGTTTT\n")
    report = tmp_path / "sel.json"
    report.write_text(
        json.dumps({"sequences": {"low": {"mssi_score": 0.1}, "high": {"mssi_score": 0.9}}})
    )
    return fasta, report


def test_read_fasta_joins_wrapped_lines(tmp_path):
    path = tmp_path / "w.fasta"
    path.write_text(">a\nACGT\nACGT\n>b\nTTTT\n")
    assert read_fasta(path) == {"a": "ACGTACGT", "b": "TTTT"}


def test_ranking_is_by_descending_mssi(scored_set):
    fasta, report = scored_set
    ranked = load_ranked_sequences(report, fasta)
    assert [seq_id for seq_id, _, _ in ranked] == ["high", "low"]
    assert ranked[0][1] == "GGGGTTTT"


def test_missing_report_fails_loudly(tmp_path, scored_set):
    fasta, _ = scored_set
    with pytest.raises(FileNotFoundError):
        load_ranked_sequences(tmp_path / "absent.json", fasta)


def test_report_referencing_unknown_sequence_fails_loudly(tmp_path, scored_set):
    fasta, _ = scored_set
    report = tmp_path / "bad.json"
    report.write_text(json.dumps({"sequences": {"ghost": {"mssi_score": 0.5}}}))
    with pytest.raises(ValueError, match="absent"):
        load_ranked_sequences(report, fasta)


def test_empty_report_fails_loudly(tmp_path, scored_set):
    fasta, _ = scored_set
    report = tmp_path / "empty.json"
    report.write_text(json.dumps({"sequences": {}}))
    with pytest.raises(ValueError, match="No scored sequences"):
        load_ranked_sequences(report, fasta)


def test_occlusion_replaces_exactly_one_tile_and_keeps_length():
    sequence = "ACGT" * 25  # 100 bp, occluded in four 25 bp tiles.
    variants = occlude(sequence, tile_size=25, seed=0)
    assert len(variants) == 4
    for index, variant in variants:
        assert len(variant) == len(sequence)
        # Everything outside the occluded tile is untouched.
        assert variant[: index * 25] == sequence[: index * 25]
        assert variant[(index + 1) * 25 :] == sequence[(index + 1) * 25 :]


def test_occlusion_is_reproducible_under_a_seed():
    sequence = "ACGT" * 25
    assert occlude(sequence, 25, seed=7) == occlude(sequence, 25, seed=7)
    assert occlude(sequence, 25, seed=7) != occlude(sequence, 25, seed=8)


def test_trailing_partial_tile_is_covered():
    # 10 bp in 4 bp tiles: the last tile is only 2 bp but must still be scanned.
    variants = occlude("ACGTACGTAC", tile_size=4, seed=0)
    assert len(variants) == 3
    assert all(len(variant) == 10 for _, variant in variants)


def test_non_positive_tile_size_fails_loudly():
    with pytest.raises(ValueError, match="tile_size"):
        occlude("ACGT", tile_size=0, seed=0)
