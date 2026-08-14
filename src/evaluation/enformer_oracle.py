"""Enformer oracle for MS-ENHANCER-GEN.

Wraps ``enformer-pytorch`` in the shared :class:`TrackOracle` interface. All the
context construction and MSSI logic lives in ``track_oracle.py``; only weight
loading and the forward pass are Enformer-specific.

Enformer has no microglia track, so the microglia arm is scored against CD14+
monocyte tracks. That substitution is documented in ``configs/model_config.yaml``
and is a real limitation of this oracle, not a detail: Borzoi does carry an
ATAC:Microglia track, which is one reason to score both.
"""

import logging

import numpy as np
import torch

from src.evaluation.track_oracle import TrackOracle

logger = logging.getLogger(__name__)


class EnformerOracle(TrackOracle):
    """In-silico evaluation oracle using Enformer to predict cell-type specificity."""

    CONFIG_SECTION = "enformer"
    MODEL_LABEL = "Enformer"
    DEFAULT_CONTEXT_LENGTH = 196608
    DEFAULT_N_TRACKS = 5313
    DEFAULT_MODEL_NAME = "EleutherAI/enformer-official-rough"

    def _load_model(self) -> None:
        """Load Enformer weights on first use.

        Raises:
            RuntimeError: If ``enformer_pytorch`` is unavailable or the weights
                cannot be loaded. Scoring without the real model is not supported.
        """
        if self.model is not None:
            return

        try:
            from enformer_pytorch import Enformer
        except ImportError as error:
            raise RuntimeError(
                "enformer_pytorch is not installed, so no MSSI can be computed. "
                "Install it (`pip install enformer-pytorch`) or skip Enformer evaluation."
            ) from error

        if self.device.type == "cpu":
            logger.warning("No CUDA device found; Enformer will run on CPU (minutes per sequence).")

        logger.info("Loading Enformer weights from %s ...", self.model_name)
        try:
            try:
                import transformers.utils.import_utils
                import transformers.modeling_utils
                transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
                transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
            except Exception:
                pass

            self.model = Enformer.from_pretrained(self.model_name).to(self.device).eval()
        except Exception as error:
            raise RuntimeError(
                f"Could not load Enformer weights '{self.model_name}': {error}"
            ) from error

    def _predict_track_means(self, full_seq: str) -> np.ndarray:
        """Run Enformer and average each track over its output bins.

        Args:
            full_seq: Context sequence of length ``context_length``.

        Returns:
            Array of shape ``(5313,)``.
        """
        self._load_model()

        # enformer-pytorch expects (batch, length, 4); a (4, length) tensor would
        # be silently read as a 4 bp sequence with 196,608 channels.
        one_hot = self._one_hot_encode(full_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(one_hot)  # (1, 896, 5313)
            human = preds["human"] if isinstance(preds, dict) else preds
            return human.mean(dim=1).squeeze(0).float().cpu().numpy()
