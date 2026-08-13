"""Borzoi oracle for MS-ENHANCER-GEN.

The second, independent oracle. Borzoi was trained separately from Enformer, on a
larger track set and a 524 kb context rather than 197 kb, so agreement between
the two is evidence that a result reflects sequence rather than one model's
idiosyncrasy. Disagreement is equally informative and is the reason this exists.

Two differences from Enformer are easy to get silently wrong and are handled
here rather than in the shared base:

* Borzoi takes ``(batch, 4, length)`` where enformer-pytorch takes
  ``(batch, length, 4)``.
* Borzoi returns ``(batch, tracks, bins)`` where Enformer returns
  ``(batch, bins, tracks)``, so the bin axis to average over is the last one.

Getting either wrong produces a well-formed number rather than an error.

Borzoi also carries a real ATAC:Microglia track, which Enformer lacks; the
Enformer microglia arm uses a CD14+ monocyte proxy. The microglia comparison is
therefore not symmetric between the two oracles.
"""

import logging

import numpy as np
import torch

from src.evaluation.track_oracle import TrackOracle

logger = logging.getLogger(__name__)


class BorzoiOracle(TrackOracle):
    """In-silico evaluation oracle using Borzoi to predict cell-type specificity."""

    CONFIG_SECTION = "borzoi"
    MODEL_LABEL = "Borzoi"
    DEFAULT_CONTEXT_LENGTH = 524288
    DEFAULT_N_TRACKS = 7611
    DEFAULT_MODEL_NAME = "johahi/borzoi-replicate-0"

    def _load_model(self) -> None:
        """Load Borzoi weights on first use.

        Raises:
            RuntimeError: If ``borzoi_pytorch`` is unavailable or the weights
                cannot be loaded.
        """
        if self.model is not None:
            return

        try:
            from borzoi_pytorch import Borzoi
        except ImportError as error:
            raise RuntimeError(
                "borzoi_pytorch is not installed, so no Borzoi MSSI can be computed. "
                "Install it (`pip install borzoi-pytorch`) or skip Borzoi evaluation."
            ) from error

        if self.device.type == "cpu":
            logger.warning("No CUDA device found; Borzoi will run on CPU (minutes per sequence).")

        logger.info("Loading Borzoi weights from %s ...", self.model_name)
        try:
            self.model = Borzoi.from_pretrained(self.model_name).to(self.device).eval()
        except Exception as error:
            raise RuntimeError(
                f"Could not load Borzoi weights '{self.model_name}': {error}"
            ) from error

    def _predict_track_means(self, full_seq: str) -> np.ndarray:
        """Run Borzoi and average each track over its output bins.

        Args:
            full_seq: Context sequence of length ``context_length``.

        Returns:
            Array of shape ``(7611,)``.
        """
        self._load_model()

        # (length, 4) -> (1, 4, length): Borzoi is channels-first.
        one_hot = self._one_hot_encode(full_seq).transpose(0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(one_hot)  # (1, 7611, 6144)
            human = preds["human"] if isinstance(preds, dict) else preds
            # Bins are the last axis here, unlike Enformer.
            return human.mean(dim=-1).squeeze(0).float().cpu().numpy()
