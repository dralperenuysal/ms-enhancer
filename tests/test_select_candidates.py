"""Unit tests for scripts/select_candidates.py."""

import json
import sys

import pandas as pd
import pytest
from Bio import SeqIO

from scripts.select_candidates import load_scores, main


@pytest.fixture
def scored_candidates(tmp_path):
    """Four candidates with known MSSI scores, plus their FASTA and metadata."""
    fasta = tmp_path / "cand.fasta"
    fasta.write_text("".join(f">c{i}\nACGT\n" for i in range(4)))

    pd.DataFrame(
        {
            "peak_id": [f"c{i}" for i in range(4)],
            "chrom": ["chr1"] * 4,
            "start": [0, 100, 200, 300],
            "end": [1000, 1100, 1200, 1300],
            "cell_type": ["B_cell"] * 4,
        }
    ).to_csv(tmp_path / "cand_metadata.csv", index=False)

    report = tmp_path / "report.json"
    scores = {"c0": 0.5, "c1": 2.5, "c2": -1.0, "c3": 1.5}
    report.write_text(json.dumps({"sequences": {k: {"mssi_score": v} for k, v in scores.items()}}))

    return tmp_path, str(fasta), str(report)


def test_load_scores_reads_every_sequence(scored_candidates):
    _, _, report = scored_candidates

    assert load_scores(report) == {"c0": 0.5, "c1": 2.5, "c2": -1.0, "c3": 1.5}


def test_load_scores_requires_the_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="evaluate.py"):
        load_scores(str(tmp_path / "absent.json"))


def test_load_scores_rejects_an_empty_report(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"sequences": {}}))

    with pytest.raises(ValueError, match="no scored sequences"):
        load_scores(str(path))


def test_selection_keeps_the_highest_scoring_candidates(scored_candidates, monkeypatch):
    """Selection is by descending MSSI, and the metadata follows the same order."""
    tmp_path, fasta, report = scored_candidates
    out_fasta = tmp_path / "selected.fasta"

    monkeypatch.setattr(sys, "argv", [
        "select_candidates.py", "--report", report, "--fasta", fasta,
        "--top_k", "2", "--out_fasta", str(out_fasta),
    ])
    main()

    assert [rec.id for rec in SeqIO.parse(str(out_fasta), "fasta")] == ["c1", "c3"]

    meta = pd.read_csv(tmp_path / "selected_metadata.csv")
    assert list(meta["peak_id"]) == ["c1", "c3"]
    assert list(meta["mssi_score"]) == [2.5, 1.5]


def test_selection_refuses_unscored_sequences(scored_candidates, monkeypatch, tmp_path):
    """Scoring one FASTA and selecting from another would silently mis-rank."""
    _, _, report = scored_candidates
    other = tmp_path / "other.fasta"
    other.write_text(">unknown\nACGT\n")

    monkeypatch.setattr(sys, "argv", [
        "select_candidates.py", "--report", report, "--fasta", str(other),
        "--out_fasta", str(tmp_path / "out.fasta"),
    ])

    with pytest.raises(ValueError, match="no score"):
        main()
