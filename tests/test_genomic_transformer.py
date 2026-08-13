"""Unit tests for src/models/genomic_transformer.py."""

import os
import tempfile
import pytest
import torch

from src.models.genomic_transformer import GenomicTransformer


@pytest.fixture
def sample_inputs():
    """Create sample sequence and condition tensors for testing."""
    batch_size = 2
    seq_len = 100  # Smaller seq_len for fast unit testing
    cond_dim = 5

    # Random sequence one-hot: (B, 4, L)
    indices = torch.randint(0, 4, (batch_size, seq_len))
    x = torch.zeros(batch_size, 4, seq_len)
    x.scatter_(1, indices.unsqueeze(1), 1.0)

    # Condition vector: (B, 5)
    c = torch.randn(batch_size, cond_dim)

    return x, c, seq_len


def test_genomic_transformer_init():
    """Test initializing GenomicTransformer with custom parameters."""
    model = GenomicTransformer(
        sequence_length=1000,
        num_tokens=4,
        condition_dim=5,
        d_model=64,
        nhead=2,
        num_layers=2,
        seed=42
    )

    assert model.sequence_length == 1000
    assert model.num_tokens == 4
    assert model.condition_dim == 5
    assert model.d_model == 64


def test_genomic_transformer_forward(sample_inputs):
    """Test forward pass returns correct logits tensor shape."""
    x, c, seq_len = sample_inputs
    model = GenomicTransformer(sequence_length=seq_len, d_model=32, nhead=2, num_layers=1)

    logits = model(x, c)

    assert logits.shape == (2, 4, seq_len)


def test_genomic_transformer_compute_loss(sample_inputs):
    """Test loss computation for GenomicTransformer."""
    x, c, seq_len = sample_inputs
    model = GenomicTransformer(sequence_length=seq_len, d_model=32, nhead=2, num_layers=1)

    logits = model(x, c)
    loss_dict = model.compute_loss(logits, x)

    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_genomic_transformer_sample():
    """Test autoregressive sampling with GenomicTransformer."""
    seq_len = 20  # Fast test sequence length
    model = GenomicTransformer(sequence_length=seq_len, d_model=32, nhead=2, num_layers=1)
    condition = torch.tensor([1.0, 0.0, 0.0, 0.5, -0.2])

    samples = model.sample(condition, num_samples=3, seed=123)

    assert samples.shape == (3, 4, seq_len)
    prob_sums = samples.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)


def test_genomic_transformer_reproducible_sampling():
    """Test reproducible sampling using explicit seed."""
    seq_len = 20
    model = GenomicTransformer(sequence_length=seq_len, d_model=32, nhead=2, num_layers=1)
    condition = torch.tensor([1.0, 0.0, 0.0, 0.5, -0.2])

    s1 = model.sample(condition, num_samples=2, seed=777)
    s2 = model.sample(condition, num_samples=2, seed=777)

    assert torch.allclose(s1, s2)


def test_genomic_transformer_checkpoint():
    """Test saving model checkpoint."""
    model = GenomicTransformer(sequence_length=100, d_model=32, nhead=2, num_layers=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, "transformer.pt")
        model.save_checkpoint(ckpt_path, epoch=3, loss=0.8)

        assert os.path.exists(ckpt_path)
        loaded = torch.load(ckpt_path)
        assert loaded["epoch"] == 3
        assert loaded["d_model"] == 32


def test_compute_loss_is_shifted_not_identity():
    """The objective must not be satisfiable by copying the input token."""
    import math

    model = GenomicTransformer(sequence_length=32, condition_dim=3, d_model=16, nhead=2, num_layers=1, seed=0)

    x = torch.zeros(2, 4, 32)
    x[:, 0, :] = 1.0  # all-A sequences

    # Logits that perfectly reproduce the *input* token at each position. Under an
    # unshifted loss this scores ~0; under teacher forcing it is only correct
    # because the next token also happens to be A.
    identity_logits = torch.full((2, 4, 32), -10.0)
    identity_logits[:, 0, :] = 10.0
    assert model.compute_loss(identity_logits, x)["loss"].item() < 0.01

    # A sequence that alternates: copying the current token is now always wrong.
    alternating = torch.zeros(2, 4, 32)
    alternating[:, 0, 0::2] = 1.0  # A at even positions
    alternating[:, 1, 1::2] = 1.0  # C at odd positions

    copy_logits = alternating * 20.0 - 10.0
    loss = model.compute_loss(copy_logits, alternating)["loss"]

    # If the loss were unshifted this would be ~0; shifted, copying is maximally wrong.
    assert loss.item() > math.log(4)


def test_compute_loss_ignores_ambiguous_positions():
    """Uniform (N) columns are excluded from the objective."""
    model = GenomicTransformer(sequence_length=8, condition_dim=3, d_model=16, nhead=2, num_layers=1, seed=0)

    x = torch.zeros(1, 4, 8)
    x[:, 0, :4] = 1.0   # real A positions
    x[:, :, 4:] = 0.25  # ambiguous

    logits = torch.full((1, 4, 8), -10.0)
    logits[:, 0, :] = 10.0  # always predicts A

    # Positions 1..3 are A (correct); 4..7 are N and must not be scored at all.
    assert model.compute_loss(logits, x)["loss"].item() < 0.01


def test_sample_returns_the_tokens_it_drew():
    """sample() returns a one-hot sequence, not an off-by-one distribution."""
    model = GenomicTransformer(sequence_length=12, condition_dim=3, d_model=16, nhead=2, num_layers=1, seed=0)
    cond = torch.zeros(3)
    cond[0] = 1.0

    out = model.sample(condition=cond, num_samples=2, seed=1)

    assert out.shape == (2, 4, 12)
    assert torch.all(out.sum(dim=1) == 1.0)  # exactly one base per position
    assert set(out.unique().tolist()) == {0.0, 1.0}


def test_checkpoint_records_full_architecture(tmp_path):
    """A checkpoint must carry every field needed to rebuild the same model."""
    model = GenomicTransformer(
        sequence_length=16, condition_dim=3, d_model=32, nhead=2,
        num_layers=3, dim_feedforward=64, seed=0,
    )
    path = tmp_path / "transformer.pt"
    model.save_checkpoint(str(path), epoch=1)

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    assert ckpt["nhead"] == 2
    assert ckpt["num_layers"] == 3
    assert ckpt["dim_feedforward"] == 64

    # Rebuilding from the checkpoint alone must accept the saved weights.
    rebuilt = GenomicTransformer(
        sequence_length=ckpt["sequence_length"], num_tokens=ckpt["num_tokens"],
        condition_dim=ckpt["condition_dim"], d_model=ckpt["d_model"],
        nhead=ckpt["nhead"], num_layers=ckpt["num_layers"],
        dim_feedforward=ckpt["dim_feedforward"], seed=ckpt["seed"],
    )
    rebuilt.load_state_dict(ckpt["model_state_dict"])
