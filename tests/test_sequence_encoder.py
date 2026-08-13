"""Unit tests for src/data_processing/sequence_encoder.py."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from src.data_processing.sequence_encoder import SequenceEncoder


@pytest.fixture
def temp_config_file():
    """Create a temporary data config file for testing."""
    config_content = """
sequence_encoding:
  sequence_length: 1000
  n_handling: "uniform"

condition_encoding:
  cell_types:
    - "CD4_T_cell"
    - "B_cell"
    - "microglia"
  normalization_method: "zscore"
  continuous_features:
    - "peak_score"
    - "fold_enrichment"

paths:
  processed_dir: "{temp_dir}"
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "data_config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content.format(temp_dir=temp_dir))
        yield config_path, temp_dir


@pytest.fixture
def sample_fasta_file():
    """Create a temporary FASTA file with 2 sequences of length 1000."""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        seq1 = "A" * 500 + "C" * 500
        seq2 = "G" * 500 + "T" * 500
        rec1 = SeqRecord(Seq(seq1), id="seq1", description="test seq 1")
        rec2 = SeqRecord(Seq(seq2), id="seq2", description="test seq 2")
        SeqIO.write([rec1, rec2], f, "fasta")
        fasta_path = f.name
    yield fasta_path
    if os.path.exists(fasta_path):
        os.remove(fasta_path)


def test_encoder_init_missing_config():
    """Test initializing encoder with missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SequenceEncoder(config_path="non_existent_config.yaml")


def test_one_hot_encode_sequence_valid(temp_config_file):
    """Test one-hot encoding a valid 1000 bp sequence."""
    config_path, _ = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    seq = "A" * 250 + "C" * 250 + "G" * 250 + "T" * 250
    encoded = encoder.one_hot_encode_sequence(seq)

    assert encoded.shape == (4, 1000)
    assert np.all(encoded[0, :250] == 1.0)
    assert np.all(encoded[1, 250:500] == 1.0)
    assert np.all(encoded[2, 500:750] == 1.0)
    assert np.all(encoded[3, 750:] == 1.0)


def test_one_hot_encode_sequence_invalid_length(temp_config_file):
    """Test encoding sequence with wrong length raises ValueError."""
    config_path, _ = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    short_seq = "ACGT" * 10  # 40 bp != 1000 bp
    with pytest.raises(ValueError, match="does not match required length"):
        encoder.one_hot_encode_sequence(short_seq)


def test_one_hot_encode_ambiguous_base(temp_config_file):
    """Test handling of ambiguous base N (uniform vs zero)."""
    config_path, _ = temp_config_file
    encoder_uniform = SequenceEncoder(config_path=config_path)

    seq_with_n = "N" * 1000
    encoded_uniform = encoder_uniform.one_hot_encode_sequence(seq_with_n)
    assert np.allclose(encoded_uniform, 0.25)

    # Test zero handling
    encoder_uniform.n_handling = "zero"
    encoded_zero = encoder_uniform.one_hot_encode_sequence(seq_with_n)
    assert np.allclose(encoded_zero, 0.0)


def test_decode_one_hot(temp_config_file):
    """Test decoding encoded matrix back to DNA string."""
    config_path, _ = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    seq = "A" * 250 + "C" * 250 + "G" * 250 + "T" * 250
    encoded = encoder.one_hot_encode_sequence(seq)
    decoded = encoder.decode_one_hot(encoded)

    assert decoded == seq


def test_encode_fasta(temp_config_file, sample_fasta_file):
    """Test encoding FASTA file into PyTorch tensor."""
    config_path, _ = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    tensor, headers = encoder.encode_fasta(sample_fasta_file)

    assert isinstance(tensor, torch.FloatTensor)
    assert tensor.shape == (2, 4, 1000)
    assert headers == ["seq1", "seq2"]


def test_build_condition_vectors(temp_config_file):
    """Test building condition tensor from metadata DataFrame."""
    config_path, _ = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    df = pd.DataFrame({
        "cell_type": ["CD4_T_cell", "B_cell", "microglia"],
        "peak_score": [10.0, 20.0, 30.0],
        "fold_enrichment": [1.5, 2.5, 3.5]
    })

    cond_tensor = encoder.build_condition_vectors(df)

    assert isinstance(cond_tensor, torch.FloatTensor)
    # Shape: 3 samples x (3 cell types one-hot + 2 continuous features) = (3, 5)
    assert cond_tensor.shape == (3, 5)

    # Check cell_type one-hot encoding
    assert torch.equal(cond_tensor[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(cond_tensor[1, :3], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(cond_tensor[2, :3], torch.tensor([0.0, 0.0, 1.0]))


def test_process_and_save_dataset(temp_config_file, sample_fasta_file):
    """Test end-to-end dataset processing and saving."""
    config_path, temp_dir = temp_config_file
    encoder = SequenceEncoder(config_path=config_path)

    df = pd.DataFrame({
        "cell_type": ["CD4_T_cell", "B_cell"],
        "peak_score": [10.0, 20.0],
        "fold_enrichment": [1.5, 2.5]
    })
    meta_path = os.path.join(temp_dir, "metadata.csv")
    df.to_csv(meta_path, index=False)

    saved_path = encoder.process_and_save_dataset(
        fasta_path=sample_fasta_file,
        metadata_path=meta_path,
        output_filename="test_dataset.pt"
    )

    assert os.path.exists(saved_path)
    loaded_data = torch.load(saved_path)

    assert "sequences" in loaded_data
    assert "conditions" in loaded_data
    assert loaded_data["sequences"].shape == (2, 4, 1000)
    assert loaded_data["conditions"].shape == (2, 5)


def test_build_condition_vectors_reuses_supplied_stats(temp_config_file):
    """Supplied normalization constants are applied instead of being re-estimated."""
    encoder = SequenceEncoder(config_path=temp_config_file[0])

    training = pd.DataFrame(
        {
            "cell_type": ["CD4_T_cell", "B_cell", "microglia"],
            "peak_score": [100.0, 200.0, 300.0],
            "fold_enrichment": [1.0, 2.0, 3.0],
        }
    )
    encoder.build_condition_vectors(training)
    stats = encoder.normalization_stats

    assert stats["method"] == "zscore"
    assert stats["center"] == pytest.approx([200.0, 2.0])

    # One row, normalized against the training constants rather than its own.
    single = training.iloc[[0]].copy()
    vector = encoder.build_condition_vectors(single, stats=stats)

    expected = (100.0 - stats["center"][0]) / stats["scale"][0]
    assert vector[0, -2].item() == pytest.approx(expected, abs=1e-5)


def test_build_condition_vectors_without_stats_is_self_referential(temp_config_file):
    """Re-fitting on a single row collapses it to zero — the bug stats prevent."""
    encoder = SequenceEncoder(config_path=temp_config_file[0])
    single = pd.DataFrame(
        {"cell_type": ["B_cell"], "peak_score": [100.0], "fold_enrichment": [1.0]}
    )

    vector = encoder.build_condition_vectors(single)

    assert vector[0, -2].item() == pytest.approx(0.0)


def test_one_hot_encode_handles_ambiguous_bases(temp_config_file):
    """Ambiguous codes get the uniform column under n_handling='uniform'."""
    encoder = SequenceEncoder(config_path=temp_config_file[0])
    encoder.sequence_length = 6

    encoded = encoder.one_hot_encode_sequence("ACGTNR")

    assert encoded[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert encoded[:, 3].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert encoded[:, 4].tolist() == [0.25] * 4  # N
    assert encoded[:, 5].tolist() == [0.25] * 4  # R, a non-N IUPAC code
