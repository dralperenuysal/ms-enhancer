"""Genomic Causal Transformer module for MS-ENHANCER-GEN.

Implements an autoregressive 1D Transformer decoder over nucleotide tokens
conditioned on cell-type epigenomic signals via cross-attention.
"""

import os
import math
import logging
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Handlers are configured centrally by src.utils.helpers.setup_logging; attaching
# one here as well would emit every record twice.
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor of shape (B, L, d_model)."""
        return x + self.pe[:, :x.size(1)]


class TransformerDecoderLayerWithCrossAttn(nn.Module):
    """Transformer decoder block with self-attention, cross-attention over condition memory, and FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run one decoder block.

        Args:
            tgt: Target embeddings of shape (B, L, d_model).
            memory: Condition memory of shape (B, 1, d_model).
            tgt_mask: Additive causal mask for self-attention.

        Returns:
            Updated target embeddings of shape (B, L, d_model).
        """
        # need_weights=False matters for speed: while the attention weights are
        # requested, PyTorch cannot use its fused scaled-dot-product kernel and
        # must materialise the full L x L matrix. Nothing here consumes them.
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False
        )
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 2. Cross-attention over condition memory
        tgt2, _ = self.cross_attn(tgt, memory, memory, need_weights=False)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 3. Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class GenomicTransformer(nn.Module):
    """Causal Transformer Decoder for generating 1000 bp DNA sequences."""

    def __init__(
        self,
        sequence_length: int = 1000,
        num_tokens: int = 4,
        condition_dim: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seed: int = 42
    ) -> None:
        """Initialize the Genomic Causal Transformer.

        Args:
            sequence_length: Target sequence length (1000 bp).
            num_tokens: Number of nucleotide tokens (4 for A, C, G, T).
            condition_dim: Dimension of condition vector C.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of decoder layers.
            dim_feedforward: FFN dimension.
            dropout: Dropout probability.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.seed = seed
        self.set_seed(seed)

        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.condition_dim = condition_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        # Token embedding & positional encoding
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 10)

        # Condition projection to memory space (B, 1, d_model)
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Decoder layers with cross-attention
        self.layers = nn.ModuleList([
            TransformerDecoderLayerWithCrossAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Head to output nucleotide token logits (B, L, 4)
        self.head = nn.Linear(d_model, num_tokens)

        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.info(
            f"GenomicTransformer constructed on {device_type} "
            f"(d_model={d_model}, nhead={nhead}, layers={num_layers}, cond_dim={condition_dim})."
        )

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed for PyTorch and NumPy."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Build the additive causal mask for self-attention.

        Args:
            sz: Sequence length.
            device: Device to allocate the mask on.

        Returns:
            Float mask of shape (sz, sz), 0.0 where attention is allowed and
            -inf where it is not.
        """
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal Transformer decoder.

        Args:
            x: Input sequence tensor of shape (B, 4, L) or token indices (B, L).
            c: Condition tensor of shape (B, condition_dim).

        Returns:
            Logits tensor of shape (B, 4, L).
        """
        if x.ndim == 3 and x.shape[1] == 4:
            tokens = torch.argmax(x, dim=1)  # Shape: (B, L)
        else:
            tokens = x

        # Token + Positional embeddings
        tgt = self.token_embedding(tokens)  # (B, L, d_model)
        tgt = self.pos_encoder(tgt)

        # Condition memory projection: (B, condition_dim) -> (B, 1, d_model)
        memory = self.condition_proj(c).unsqueeze(1)

        causal_mask = self.generate_square_subsequent_mask(tokens.shape[1], device=tokens.device)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=causal_mask)

        # Head to nucleotide logits: (B, L, d_model) -> (B, L, 4)
        logits_bl4 = self.head(tgt)

        # Rearrange to match cVAE shape: (B, 4, L)
        logits_b4l = logits_bl4.transpose(1, 2)
        return logits_b4l

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute CrossEntropy loss for predicted nucleotide logits.

        The prediction at position i is scored against the token at position i+1.
        Without this shift the objective is degenerate: the causal mask lets
        position i attend to itself, so the model can copy its own input token and
        drive the loss to ~0 while learning nothing about sequence structure. The
        shift also matches :meth:`sample`, which already treats the output at the
        last position as the distribution over the *next* nucleotide.

        Args:
            logits: Logits tensor of shape (B, 4, L).
            targets: Target tensor of shape (B, 4, L) or (B, L).

        Returns:
            Dictionary containing 'loss' and 'recon_loss' scalar Tensors, in nats
            per nucleotide so the value is comparable to ln 4 = 1.386.
        """
        if targets.ndim == 3 and targets.shape[1] == 4:
            target_indices = torch.argmax(targets, dim=1)  # Shape: (B, L)
            # Ambiguous positions encode as a uniform column and carry no identity.
            scored = targets.max(dim=1).values > 0.5  # Shape: (B, L)
        else:
            target_indices = targets
            scored = torch.ones_like(target_indices, dtype=torch.bool)

        # Teacher forcing: predict token i+1 from everything up to and including i.
        shifted_logits = logits[:, :, :-1]
        shifted_targets = target_indices[:, 1:]
        shifted_scored = scored[:, 1:]

        per_position = F.cross_entropy(shifted_logits, shifted_targets, reduction='none')
        loss = (per_position * shifted_scored).sum() / shifted_scored.sum().clamp(min=1)

        return {"loss": loss, "recon_loss": loss}

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressively sample synthetic 1000 bp DNA sequences given condition vector c.

        Args:
            condition: Condition tensor C of shape (condition_dim,) or (num_samples, condition_dim).
            num_samples: Number of sequences to sample if condition is 1D.
            seed: Optional seed for reproducible sampling.

        Returns:
            One-hot tensor of the sampled tokens, shape
            ``(num_samples, 4, sequence_length)``.

        Note:
            The tokens are drawn here, autoregressively, so the return value is
            the sampled sequence itself rather than a distribution. Re-running the
            model over the finished sequence would give the distribution over the
            *next* nucleotide at each position, which is offset by one from the
            token actually emitted there.
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

        # Initialize with random start tokens (0..3)
        tokens = torch.randint(0, self.num_tokens, (num_samples, 1), device=device)

        for _ in range(self.sequence_length - 1):
            logits = self.forward(tokens, c)  # Shape: (num_samples, 4, L_curr)
            last_logits = logits[:, :, -1]  # Shape: (num_samples, 4)
            probs = F.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # Shape: (num_samples, 1)
            tokens = torch.cat([tokens, next_tokens], dim=1)

        return F.one_hot(tokens, num_classes=self.num_tokens).permute(0, 2, 1).float()

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
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "sequence_length": self.sequence_length,
            "num_tokens": self.num_tokens,
            "condition_dim": self.condition_dim,
            "d_model": self.d_model,
            # Without these the architecture cannot be rebuilt from the checkpoint
            # alone; a loader would fall back to defaults and silently construct a
            # different model than the weights belong to.
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
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
        logger.info(f"Saved GenomicTransformer checkpoint to: {filepath} (Epoch {epoch})")

    @classmethod
    def load_from_config(cls, config_path: str = "configs/model_config.yaml") -> "GenomicTransformer":
        """Instantiate GenomicTransformer using hyperparameters from configuration YAML.

        Args:
            config_path: Path to model_config.yaml.

        Returns:
            Initialized GenomicTransformer instance.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("genomic_transformer", {})

        return cls(
            sequence_length=cfg.get("sequence_length", 1000),
            num_tokens=cfg.get("num_tokens", 4),
            condition_dim=cfg.get("condition_dim", 5),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
            seed=cfg.get("seed", 42)
        )
