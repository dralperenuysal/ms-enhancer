"""Unit tests for src/models/cvae_generator.py."""

import os
import tempfile
import pytest
import torch

from src.models.cvae_generator import CVAEGenerator


@pytest.fixture
def sample_inputs():
    """Create sample sequence and condition tensors for testing."""
    batch_size = 4
    seq_len = 1000
    cond_dim = 5

    # Random one-hot sequences: shape (B, 4, 1000)
    indices = torch.randint(0, 4, (batch_size, seq_len))
    x = torch.zeros(batch_size, 4, seq_len)
    x.scatter_(1, indices.unsqueeze(1), 1.0)

    # Condition vector: shape (B, 5)
    c = torch.randn(batch_size, cond_dim)

    return x, c


def test_cvae_initialization():
    """Test initializing CVAEGenerator with custom dimensions."""
    model = CVAEGenerator(
        in_channels=4,
        sequence_length=1000,
        condition_dim=5,
        latent_dim=128,
        hidden_dims=[64, 128, 256],
        seed=42
    )

    assert model.in_channels == 4
    assert model.sequence_length == 1000
    assert model.condition_dim == 5
    assert model.latent_dim == 128


def test_cvae_forward_pass(sample_inputs):
    """Test forward pass returns correct tensor shapes."""
    x, c = sample_inputs
    model = CVAEGenerator(latent_dim=64, hidden_dims=[32, 64])

    logits, mu, logvar = model(x, c)

    assert logits.shape == (4, 4, 1000)
    assert mu.shape == (4, 64)
    assert logvar.shape == (4, 64)


def test_cvae_reparameterize():
    """Test reparameterization trick outputs correct latent shape."""
    model = CVAEGenerator(latent_dim=32)
    mu = torch.zeros(4, 32)
    logvar = torch.zeros(4, 32)

    z = model.reparameterize(mu, logvar)

    assert z.shape == (4, 32)


def test_cvae_compute_loss(sample_inputs):
    """Test cVAE loss computation (reconstruction + KL divergence)."""
    x, c = sample_inputs
    model = CVAEGenerator(latent_dim=64)

    logits, mu, logvar = model(x, c)
    loss_dict = model.compute_loss(x, logits, mu, logvar, beta=1.0)

    assert "loss" in loss_dict
    assert "recon_loss" in loss_dict
    assert "kl_loss" in loss_dict

    assert loss_dict["loss"].item() > 0
    assert loss_dict["recon_loss"].item() > 0
    assert loss_dict["kl_loss"].item() >= 0


def test_cvae_sample():
    """Test sampling synthetic sequences from cVAE generator."""
    model = CVAEGenerator(latent_dim=64, hidden_dims=[32, 64])
    condition = torch.tensor([1.0, 0.0, 0.0, 0.5, -0.2])  # (5,)

    num_samples = 10
    samples = model.sample(condition, num_samples=num_samples, seed=123)

    assert samples.shape == (10, 4, 1000)
    # Check that outputs are valid probabilities (softmax along channel dim)
    prob_sums = samples.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)


def test_cvae_reproducible_sampling():
    """Test sampling with fixed seed yields identical sequences."""
    model = CVAEGenerator(latent_dim=64)
    condition = torch.tensor([1.0, 0.0, 0.0, 0.5, -0.2])

    sample1 = model.sample(condition, num_samples=5, seed=999)
    sample2 = model.sample(condition, num_samples=5, seed=999)

    assert torch.allclose(sample1, sample2)


def test_cvae_checkpoint_save_and_load(sample_inputs):
    """Test saving and loading model checkpoints."""
    x, c = sample_inputs
    model = CVAEGenerator(latent_dim=32, hidden_dims=[32, 64])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, "model_test.pt")
        model.save_checkpoint(ckpt_path, epoch=5, optimizer=optimizer, loss=0.5)

        assert os.path.exists(ckpt_path)

        loaded_ckpt = torch.load(ckpt_path)
        assert loaded_ckpt["epoch"] == 5
        assert loaded_ckpt["latent_dim"] == 32
        assert loaded_ckpt["loss"] == 0.5


def test_compute_loss_is_per_nucleotide_scaled():
    """Both loss terms live on the per-nucleotide scale, not sequence-summed."""
    import math

    model = CVAEGenerator(
        in_channels=4, sequence_length=64, condition_dim=3,
        latent_dim=8, hidden_dims=[8, 16], seed=0,
    )
    x = torch.zeros(2, 4, 64)
    x[:, 0, :] = 1.0  # all-A sequences
    logits = torch.zeros(2, 4, 64)  # uniform prediction
    mu = torch.zeros(2, 8)
    logvar = torch.zeros(2, 8)

    losses = model.compute_loss(x, logits, mu, logvar, beta=1.0)

    # Uniform logits over 4 nucleotides => exactly ln(4) nats per position.
    assert losses["recon_loss"].item() == pytest.approx(math.log(4), abs=1e-5)
    # mu=0, logvar=0 is the prior, so KL is exactly zero.
    assert losses["kl_loss"].item() == pytest.approx(0.0, abs=1e-6)

    # With a displaced posterior the KL must be divided by the sequence length,
    # which is what keeps it comparable to the reconstruction term.
    mu_shifted = torch.ones(2, 8)
    shifted = model.compute_loss(x, logits, mu_shifted, logvar, beta=1.0)
    expected_kl = 0.5 * 8 / 64  # 0.5 * sum(mu^2) over latent dims, per nucleotide
    assert shifted["kl_loss"].item() == pytest.approx(expected_kl, abs=1e-5)


def test_compute_loss_ignores_ambiguous_positions():
    """Uniform (N) columns are excluded from the reconstruction term."""
    model = CVAEGenerator(
        in_channels=4, sequence_length=8, condition_dim=3,
        latent_dim=4, hidden_dims=[8], seed=0,
    )
    x = torch.zeros(1, 4, 8)
    x[:, 0, :4] = 1.0     # four real A positions
    x[:, :, 4:] = 0.25    # four ambiguous positions

    # Confidently correct on the real positions, confidently wrong on the N ones.
    logits = torch.zeros(1, 4, 8)
    logits[:, 0, :4] = 20.0
    logits[:, 3, 4:] = 20.0

    mu = torch.zeros(1, 4)
    logvar = torch.zeros(1, 4)
    recon = model.compute_loss(x, logits, mu, logvar)["recon_loss"]

    assert recon.item() == pytest.approx(0.0, abs=1e-4)
