"""Unit tests for src/evaluation/motif_analyzer.py and src/evaluation/enformer_oracle.py."""

import os
import tempfile
import json
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.evaluation.motif_analyzer import MotifAnalyzer
from src.evaluation.enformer_oracle import EnformerOracle


@pytest.fixture
def sample_fasta():
    """Create a temporary FASTA file with sequence containing NFKB1 motif (GGGAATTTCC)."""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        seq = "A" * 100 + "GGGAATTTCC" + "C" * 890  # Exactly 1000 bp
        rec = SeqRecord(Seq(seq), id="syn_seq_1", description="")
        SeqIO.write([rec], f, "fasta")
        fasta_path = f.name
    yield fasta_path
    if os.path.exists(fasta_path):
        os.remove(fasta_path)


def test_motif_analyzer_scan_finds_real_binding_site():
    """A canonical NF-kB site is detected with a real, high log-odds score."""
    analyzer = MotifAnalyzer()
    seq_with_nfkb = "ACGT" * 50 + "GGGGAATTCCC" + "ACGT" * 50

    hits = analyzer.scan_sequence(seq_with_nfkb)
    nfkb_hits = [h for h in hits if h["tf"] == "NFKB1"]

    assert nfkb_hits
    best = max(nfkb_hits, key=lambda h: h["score"])
    assert best["relative_score"] > 0.8
    assert best["matrix_id"].startswith("MA0105")
    # The hit must land on the inserted site, not somewhere in the ACGT filler.
    assert 190 <= best["start"] <= 210
    # NF-kB's motif is near-palindromic, so both strands should score.
    assert {h["strand"] for h in nfkb_hits} == {"+", "-"}


def test_motif_analyzer_discriminates_by_score():
    """A degraded site drops below threshold; the regex scanner could not tell."""
    analyzer = MotifAnalyzer()
    strong = "ACGT" * 50 + "GGGGAATTCCC" + "ACGT" * 50
    degraded = "ACGT" * 50 + "GGGGAATTCCA" + "ACGT" * 50

    strong_hits = [h for h in analyzer.scan_sequence(strong) if h["tf"] == "NFKB1"]
    degraded_hits = [h for h in analyzer.scan_sequence(degraded) if h["tf"] == "NFKB1"]

    assert strong_hits
    assert not degraded_hits
    # Scores are graded, not the constant 1.0 the previous implementation returned.
    assert all(0.8 <= h["relative_score"] < 1.0 for h in strong_hits)


def test_motif_analyzer_rejects_unknown_tf(tmp_path):
    """A TF absent from JASPAR fails loudly instead of being silently skipped."""
    from src.evaluation.motif_analyzer import MotifDatabaseError

    config = tmp_path / "model_config.yaml"
    config.write_text(
        "evaluation:\n"
        "  motif_analysis:\n"
        "    tfs:\n"
        "      - NOT_A_REAL_TF_XYZ\n"
    )

    with pytest.raises(MotifDatabaseError, match="NOT_A_REAL_TF_XYZ"):
        MotifAnalyzer(config_path=str(config))


def test_motif_analyzer_requires_configured_tfs(tmp_path):
    """An empty TF list is a configuration error, not a silent default."""
    config = tmp_path / "model_config.yaml"
    config.write_text("evaluation:\n  motif_analysis:\n    tfs: []\n")

    with pytest.raises(ValueError, match="No transcription factors configured"):
        MotifAnalyzer(config_path=str(config))


def test_motif_analyzer_analyze_fasta(sample_fasta):
    """Test analyzing FASTA file with MotifAnalyzer."""
    analyzer = MotifAnalyzer()
    report = analyzer.analyze_fasta(sample_fasta)

    assert "sequences" in report
    assert "tf_counts" in report
    assert report["jaspar_release"] == "JASPAR2024"
    assert set(report["tf_mean_relative_score"]) == set(report["tf_counts"])


def test_enformer_oracle_context_window_with_explicit_flanks():
    """Test Enformer context window padding to 196,608 bp."""
    oracle = EnformerOracle()
    seq_1000 = "A" * 1000
    flanks = "C" * (196608 - 1000)

    full_seq = oracle.construct_context_window(seq_1000, flanking_seq=flanks)
    assert len(full_seq) == 196608
    # 1000 bp target sequence should be placed in the center
    offset = (196608 - 1000) // 2
    assert full_seq[offset:offset + 1000] == seq_1000


def test_enformer_oracle_refuses_synthetic_flanks():
    """Without real flanking context the oracle must refuse to build a window."""
    oracle = EnformerOracle()

    with pytest.raises(ValueError, match="no real flanking sequence"):
        oracle.construct_context_window("A" * 1000)


def test_enformer_oracle_rejects_unknown_cell_type():
    """Scoring requires a cell type with configured target tracks."""
    oracle = EnformerOracle()

    with pytest.raises(KeyError, match="hepatocyte"):
        oracle.predict_cell_specificity("A" * 1000, cell_type="hepatocyte")


def test_enformer_oracle_mssi_uses_configured_tracks(monkeypatch):
    """MSSI is mean(target tracks) - mean(background tracks) on the real output."""
    import torch

    oracle = EnformerOracle()
    cell_type = next(iter(oracle.target_tracks_by_cell_type))
    targets = oracle.target_tracks_by_cell_type[cell_type]

    # Stand-in for Enformer: track i predicts a constant signal of i.
    predictions = torch.arange(oracle.n_tracks, dtype=torch.float32).repeat(1, 896, 1)

    class FakeEnformer:
        def __call__(self, one_hot):
            # enformer-pytorch takes (batch, length, 4); catch a transposed tensor.
            assert one_hot.shape == (1, oracle.context_length, 4), one_hot.shape
            return {"human": predictions}

    oracle.model = FakeEnformer()
    monkeypatch.setattr(oracle, "_load_model", lambda: None)

    res = oracle.predict_cell_specificity(
        "A" * 1000, cell_type=cell_type, flanking_seq="C" * (oracle.context_length - 1000)
    )

    expected = sum(targets) / len(targets) - sum(oracle.background_tracks) / len(oracle.background_tracks)
    assert res["mssi_score"] == pytest.approx(expected, abs=1e-3)
    assert res["cell_type"] == cell_type


def test_enformer_oracle_requires_metadata_for_every_sequence(sample_fasta, tmp_path):
    """A sequence with no cell type / locus cannot be scored."""
    import pandas as pd

    oracle = EnformerOracle()
    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"peak_id": "other_seq", "chrom": "chr1", "start": 0, "end": 1000, "cell_type": "B_cell"}]
    ).to_csv(metadata, index=False)
    reference = tmp_path / "ref.fa"
    reference.write_text(">chr1\nACGT\n")

    with pytest.raises(ValueError, match="no metadata row"):
        oracle.evaluate_fasta(
            sample_fasta,
            metadata_path=str(metadata),
            reference_fasta_path=str(reference),
            output_report_path=str(tmp_path / "report.json"),
        )
