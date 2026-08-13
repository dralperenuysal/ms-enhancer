"""Conditional Variational Autoencoder (cVAE) for DNA Sequence Generation.

Implements a 1D Convolutional Conditional VAE architecture that generates 1000 bp DNA
sequences conditioned on cell-type epigenomic signals for MS regulatory region design.
"""

import os
import logging
from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Handlers are configured centrally by src.utils.helpers.setup_logging; attaching
# one here as well would emit every record twice.
logger = logging.getLogger(__name__)


class Conv1DBlock(nn.Module):
    """Convolutional block with Conv1d, BatchNorm, and LeakyReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, padding: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Deconv1DBlock(nn.Module):
    """Deconvolutional block with ConvTranspose1d, BatchNorm, and LeakyReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, padding: int = 3) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class CVAEGenerator(nn.Module):
    """Conditional VAE for generating 1000 bp DNA sequences."""

    def __init__(
        self,
        in_channels: int = 4,
        sequence_length: int = 1000,
        condition_dim: int = 5,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42
    ) -> None:
        """Initialize the Conditional VAE model.

        Args:
            in_channels: Number of sequence input channels (4 for A, C, G, T).
            sequence_length: Length of DNA sequence (default: 1000 bp).
            condition_dim: Dimension of condition vector C.
            latent_dim: Dimension of latent space z.
            hidden_dims: List of channel dimensions for Conv1d encoder layers.
            seed: Seed for PyTorch random number generators.
        """
        super().__init__()
        self.seed = seed
        self.set_seed(seed)

        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 128, 256]

        # Log hardware device info
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.info(f"Initializing CVAEGenerator on device: {device_type}")

        # Encoder: Conv1d layers with concatenated condition along channel dim
        # Input channel dim: 4 (sequence) + condition_dim (broadcasted along sequence length)
        encoder_in_dim = self.in_channels + self.condition_dim
        encoder_layers = []

        curr_channels = encoder_in_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    Conv1DBlock(curr_channels, h_dim, kernel_size=7, stride=2, padding=3),
                    nn.Dropout1d(0.1)
                )
            )
            curr_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate shape after convolutional downsampling (stride=2 for each layer)
        self.reduced_length = self.sequence_length
        for _ in self.hidden_dims:
            self.reduced_length = (self.reduced_length + 2 * 3 - 7) // 2 + 1  # Standard Conv1d shape formula

        self.flatten_dim = self.hidden_dims[-1] * self.reduced_length

        # Latent projections (mu and logvar)
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)

        # Decoder: Latent z + condition C -> FC layer -> Deconv1d layers -> Output logits (4, 1000)
        decoder_in_dim = self.latent_dim + self.condition_dim
        self.decoder_fc = nn.Linear(decoder_in_dim, self.flatten_dim)

        decoder_layers = []
        rev_dims = list(reversed(self.hidden_dims))

        for i in range(len(rev_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        rev_dims[i],
                        rev_dims[i + 1],
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        output_padding=1
                    ),
                    nn.BatchNorm1d(rev_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )

        # Final transposed conv layer to output 4 channels (A, C, G, T) and exact sequence length 1000
        self.decoder_deconv = nn.Sequential(*decoder_layers)
        self.final_layer = nn.ConvTranspose1d(
            rev_dims[-1],
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            output_padding=1
        )

        logger.info(
            f"CVAEGenerator constructed successfully (latent_dim={self.latent_dim}, "
            f"condition_dim={self.condition_dim}, reduced_length={self.reduced_length})."
        )

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence x and condition c into latent mean mu and logvar.

        Args:
            x: Sequence tensor of shape (B, 4, L).
            c: Condition tensor of shape (B, condition_dim).

        Returns:
            Tuple of (mu, logvar), each of shape (B, latent_dim).
        """
        batch_size, _, seq_len = x.shape
        # Broadcast condition vector c along sequence length: (B, condition_dim) -> (B, condition_dim, L)
        c_broadcast = c.unsqueeze(2).expand(batch_size, self.condition_dim, seq_len)

        # Concatenate along channel dimension: (B, 4 + condition_dim, L)
        xc = torch.cat([x, c_broadcast], dim=1)

        conv_out = self.encoder_conv(xc)
        flat = torch.flatten(conv_out, start_dim=1)

        mu = self.fc_mu(flat)
        logvar = torch.clamp(self.fc_logvar(flat), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick z = mu + std * eps.

        Args:
            mu: Mean tensor of shape (B, latent_dim).
            logvar: Log-variance tensor of shape (B, latent_dim).

        Returns:
            Sampled latent vector z of shape (B, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z and condition c into nucleotide sequence logits.

        Args:
            z: Latent space tensor of shape (B, latent_dim).
            c: Condition tensor of shape (B, condition_dim).

        Returns:
            Logits tensor of shape (B, 4, sequence_length).
        """
        zc = torch.cat([z, c], dim=1)
        flat = self.decoder_fc(zc)
        conv_in = flat.view(-1, self.hidden_dims[-1], self.reduced_length)

        deconv_out = self.decoder_deconv(conv_in)
        logits = self.final_layer(deconv_out)

        # Adjust length if off by a small amount due to convTranspose rounding
        if logits.shape[2] != self.sequence_length:
            logits = F.interpolate(logits, size=self.sequence_length, mode='linear', align_corners=False)

        return logits

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through cVAE encoder and decoder.

        Args:
            x: Input sequence tensor of shape (B, 4, L).
            c: Condition tensor of shape (B, condition_dim).

        Returns:
            Tuple of (reconstructed_logits, mu, logvar).
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, c)
        return recon_logits, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute cVAE loss: Reconstruction (CrossEntropy) + beta * KL Divergence.

        Args:
            x: Target ground-truth sequence tensor of shape (B, 4, L).
            logits: Predicted sequence logits of shape (B, 4, L).
            mu: Latent mean vector of shape (B, latent_dim).
            logvar: Latent logvar vector of shape (B, latent_dim).
            beta: Weight factor for KL divergence loss term (for KL annealing).

        Both terms are expressed in nats *per nucleotide*: the reconstruction term
        is averaged over scored positions and the KL term is divided by the sequence
        length. Without this the KL is summed over the latent dimension (order 10^2
        nats) while the reconstruction is a per-position mean (bounded by ln 4 =
        1.386), so beta=1 would drive the model straight into posterior collapse.
        The per-nucleotide scale also makes recon_loss directly comparable to
        ln 4 = 1.386, the value of a model that has learned nothing.

        Returns:
            Dictionary containing 'loss', 'recon_loss', and 'kl_loss' scalar Tensors.
        """
        # Targets for CrossEntropy: class indices 0..3 from max encoding
        targets = torch.argmax(x, dim=1)  # Shape: (B, L)

        # Ambiguous positions are encoded as a uniform (0.25) or zero column and
        # carry no nucleotide identity; argmax would silently label them 'A' and
        # train the model to predict A wherever the reference has an N.
        scored = x.max(dim=1).values > 0.5  # Shape: (B, L)
        n_scored = scored.sum().clamp(min=1)

        # Reconstruction loss (CrossEntropy over nucleotides), N positions excluded
        per_position = F.cross_entropy(logits, targets, reduction='none')  # (B, L)
        recon_loss = (per_position * scored).sum() / n_scored

        # KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per sequence,
        # averaged over the batch and rescaled to the same per-nucleotide units.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() / self.sequence_length

        total_loss = recon_loss + beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Sample synthetic 1000 bp DNA sequences given condition vector c.

        Args:
            condition: Condition tensor C of shape (condition_dim,) or (num_samples, condition_dim).
            num_samples: Number of sequences to sample if condition is 1D.
            seed: Optional seed for reproducible sampling.

        Returns:
            Generated sequence tensor of shape (num_samples, 4, sequence_length) with one-hot/probabilities.
        """
        self.eval()
        if seed is not None:
            self.set_seed(seed)

        device = next(self.parameters()).device

        if condition.ndim == 1:
            c = condition.unsqueeze(0).repeat(num_samples, 1).to(device)
        else:
            c = condition.to(device)
            num_samples = c.shape[0]

        # Sample z from standard Gaussian prior N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decode(z, c)
        probs = F.softmax(logits, dim=1)

        return probs

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint file to specified filepath.

        Args:
            filepath: Target file path for model checkpoint (.pt).
            epoch: Current training epoch index.
            optimizer: Optional PyTorch optimizer instance.
            loss: Optional validation loss value.
            extra: Additional entries to store alongside the model state, e.g. the
                RNG states needed to resume training without changing the random
                stream. Keys must not collide with the standard fields.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "sequence_length": self.sequence_length,
            "condition_dim": self.condition_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "loss": loss,
            # Recorded so a checkpoint carries what is needed to reproduce it.
            "seed": self.seed,
            "torch_version": str(torch.__version__),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            collisions = set(extra).intersection(checkpoint)
            if collisions:
                raise ValueError(f"extra keys collide with checkpoint fields: {sorted(collisions)}")
            checkpoint.update(extra)

        torch.save(checkpoint, filepath)
        logger.info(f"Saved CVAE checkpoint to: {filepath} (Epoch {epoch})")

    @classmethod
    def load_from_config(cls, config_path: str = "configs/model_config.yaml") -> "CVAEGenerator":
        """Instantiate CVAEGenerator using hyperparameters from configuration YAML.

        Args:
            config_path: Path to model_config.yaml.

        Returns:
            Initialized CVAEGenerator instance.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("cvae", {})

        return cls(
            in_channels=cfg.get("in_channels", 4),
            sequence_length=cfg.get("sequence_length", 1000),
            condition_dim=cfg.get("condition_dim", 5),
            latent_dim=cfg.get("latent_dim", 128),
            hidden_dims=cfg.get("hidden_dims", [64, 128, 256]),
            seed=cfg.get("seed", 42)
        )
