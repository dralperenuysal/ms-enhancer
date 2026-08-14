"""Shared oracle machinery for MS-ENHANCER-GEN.

Scores synthetic 1000 bp DNA sequences for predicted cell-type-specific
activity and computes the MS Specificity Index (MSSI), defined as
``mean(target tracks) - mean(background tracks)``.

Everything here is model-agnostic: locating the insert in real genomic context,
reading flanks from the reference, and reducing a track prediction to MSSI. A
concrete oracle subclasses :class:`TrackOracle` and supplies three things — the
config section its tracks are enumerated in, how to load its weights, and how to
turn a context sequence into per-track means. See ``enformer_oracle.py`` and
``borzoi_oracle.py``.

Two deliberate absences, both because a fabricated score is indistinguishable
from a real one once written to a report: there is no synthetic-flank fallback
(a repeating or random background produces a number that measures nothing), and
there is no mock prediction path (a model that cannot be loaded raises).

Running two independently trained oracles is not redundancy. A single model that
is consistently wrong yields exactly the same internally consistent results as a
model that is right; only agreement between models trained on different data
with different architectures distinguishes the two cases.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO

logger = logging.getLogger(__name__)


class TrackOracle:
    """Base oracle: scores an insert in real genomic context against track sets.

    Subclasses set :attr:`CONFIG_SECTION`, :attr:`MODEL_LABEL` and the default
    geometry, then implement :meth:`_load_model` and :meth:`_predict_track_means`.
    """

    #: Key under ``evaluation:`` in the model config holding this oracle's tracks.
    CONFIG_SECTION: str = "oracle"
    #: Human-readable model name, used in errors and log lines.
    MODEL_LABEL: str = "oracle"
    #: Context window the model consumes, in bp.
    DEFAULT_CONTEXT_LENGTH: int = 196608
    #: Size of the model's human output head.
    DEFAULT_N_TRACKS: int = 5313
    #: Default pretrained weights identifier.
    DEFAULT_MODEL_NAME: str = ""

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        """Initialize the oracle from its section of the model configuration.

        Args:
            config_path: Path to model configuration YAML file.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            ValueError: If target or background tracks are not enumerated, are
                out of range, or overlap.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f).get("evaluation", {}).get(self.CONFIG_SECTION, {})

        self.context_length: int = self.config.get("context_length", self.DEFAULT_CONTEXT_LENGTH)
        self.model_name: str = self.config.get("model_name", self.DEFAULT_MODEL_NAME)
        self.n_tracks: int = int(self.config.get("n_tracks", self.DEFAULT_N_TRACKS))

        self.target_tracks_by_cell_type: Dict[str, List[int]] = self.config.get(
            "target_tracks_by_cell_type", {}
        )
        self.background_tracks: List[int] = self.config.get("background_tracks", [])

        if not self.target_tracks_by_cell_type or not self.background_tracks:
            raise ValueError(
                f"evaluation.{self.CONFIG_SECTION}.target_tracks_by_cell_type and background_tracks must be "
                f"enumerated explicitly in {config_path}. Track selection is not inferred at runtime."
            )

        background = set(self.background_tracks)
        for cell_type, tracks in self.target_tracks_by_cell_type.items():
            out_of_range = [t for t in tracks if not 0 <= t < self.n_tracks]
            if out_of_range:
                raise ValueError(
                    f"Target track indices {out_of_range} for '{cell_type}' are outside the "
                    f"0..{self.n_tracks - 1} range of the {self.MODEL_LABEL} human head."
                )
            overlap = background.intersection(tracks)
            if overlap:
                raise ValueError(
                    f"Tracks {sorted(overlap)} are listed as both target and background for "
                    f"'{cell_type}'; MSSI would subtract a track from itself."
                )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None  # Loaded on first prediction by _load_model().

    def _load_model(self) -> None:
        """Load model weights on first use; a no-op once loaded.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _load_model().")

    def _predict_track_means(self, full_seq: str) -> np.ndarray:
        """Predict per-track means over the output bins for one context window.

        Args:
            full_seq: Context sequence of length :attr:`context_length`.

        Returns:
            Array of shape ``(n_tracks,)``.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _predict_track_means().")

    # ------------------------------------------------------------------ context

    def _extract_genomic_flanks(
        self,
        reference_fasta_path: str,
        chrom: str,
        center_pos: int,
        seq_len: int,
    ) -> Optional[str]:
        """Extract real flanking sequence from the reference genome.

        The returned string has length ``context_length`` with a gap of
        ``seq_len`` in the centre (filled by the caller).

        Args:
            reference_fasta_path: Path to indexed reference FASTA.
            chrom: Chromosome name, e.g. ``"chr6"``.
            center_pos: Centre of the synthetic insert in genomic coordinates.
            seq_len: Length of the synthetic insert.

        Returns:
            Flanking string of length ``context_length - seq_len`` (left + right
            concatenated), or ``None`` if extraction fails.
        """
        try:
            from pyfaidx import Fasta  # lazy import — tests can run without hg38
        except ImportError:
            logger.warning("pyfaidx not installed; cannot extract real genomic flanks.")
            return None

        if not os.path.exists(reference_fasta_path):
            logger.warning("Reference FASTA not found at %s; using synthetic flanks.", reference_fasta_path)
            return None

        try:
            genome = Fasta(reference_fasta_path, sequence_always_upper=True, rebuild=False)
            if chrom not in genome:
                logger.warning("Chromosome %s not in reference FASTA; using synthetic flanks.", chrom)
                return None

            half_insert = seq_len // 2
            half_ctx = self.context_length // 2
            left_start = max(0, center_pos - half_ctx)
            left_end = center_pos - half_insert
            right_start = center_pos + half_insert + (seq_len % 2)
            right_end = center_pos + half_ctx

            chrom_len = len(genome[chrom])
            right_end = min(right_end, chrom_len)

            left_flank = str(genome[chrom][left_start:left_end])
            right_flank = str(genome[chrom][right_start:right_end])

            total_needed = self.context_length - seq_len
            left_needed = total_needed // 2
            right_needed = total_needed - left_needed

            # Pad if near chromosome boundaries
            if len(left_flank) < left_needed:
                left_flank = "N" * (left_needed - len(left_flank)) + left_flank
            else:
                left_flank = left_flank[-left_needed:]

            if len(right_flank) < right_needed:
                right_flank = right_flank + "N" * (right_needed - len(right_flank))
            else:
                right_flank = right_flank[:right_needed]

            logger.debug(
                "Extracted %d bp left + %d bp right genomic flank for %s:%d.",
                len(left_flank), len(right_flank), chrom, center_pos,
            )
            return left_flank + right_flank
        except Exception as exc:
            logger.warning("Failed to extract genomic flanks (%s); falling back to synthetic.", exc)
            return None

    def construct_context_window(
        self,
        sequence_str: str,
        flanking_seq: Optional[str] = None,
        reference_fasta_path: Optional[str] = None,
        chrom: Optional[str] = None,
        center_pos: Optional[int] = None,
    ) -> str:
        """Place 1000 bp synthetic sequence at centre of 196,608 bp context window.

        Flanking bases come either from an explicit ``flanking_seq`` or from the
        reference genome at ``chrom:center_pos``. There is no synthetic fallback:
        Enformer's prediction is dominated by the surrounding ~200 kb, so a made-up
        background yields a score that reflects the padding, not the insert.

        Args:
            sequence_str: Synthetic sequence (typically 1000 bp).
            flanking_seq: Pre-computed flanking sequence (left + right concatenated).
            reference_fasta_path: Path to indexed reference FASTA for real flanks.
            chrom: Chromosome of the source locus.
            center_pos: Centre position of the source locus on ``chrom``.

        Returns:
            Padded sequence string of length ``context_length`` (196608).

        Raises:
            ValueError: If ``sequence_str`` exceeds ``context_length``, or if no
                real flanking sequence could be obtained.
        """
        target_len = self.context_length
        seq_len = len(sequence_str)

        if seq_len > target_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds context length ({target_len}).")

        pad_needed = target_len - seq_len
        left_pad_len = pad_needed // 2

        flanks = flanking_seq
        if not (flanks and len(flanks) >= pad_needed):
            if reference_fasta_path and chrom and center_pos is not None:
                flanks = self._extract_genomic_flanks(reference_fasta_path, chrom, center_pos, seq_len)
            else:
                flanks = None

        if not flanks or len(flanks) < pad_needed:
            raise ValueError(
                f"Cannot build a {target_len} bp context window: no real flanking sequence "
                f"available (chrom={chrom}, center_pos={center_pos}, "
                f"reference={reference_fasta_path}). Pass flanking_seq, or a reference FASTA "
                f"plus the source locus. Synthetic padding is not supported because the score "
                f"would reflect the padding rather than the insert."
            )

        return flanks[:left_pad_len] + sequence_str + flanks[left_pad_len:pad_needed]

    # ----------------------------------------------------------------- predict

    def predict_cell_specificity(
        self,
        sequence_str: str,
        cell_type: str,
        flanking_seq: Optional[str] = None,
        reference_fasta_path: Optional[str] = None,
        chrom: Optional[str] = None,
        center_pos: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict target vs background track signals and compute MSSI.

        MSSI is ``mean(target tracks) - mean(background tracks)`` on the tracks
        configured for ``cell_type``.

        Args:
            sequence_str: Synthetic 1000 bp DNA sequence string.
            cell_type: Cell type whose target tracks to score against; must be a
                key of ``target_tracks_by_cell_type``.
            flanking_seq: Optional pre-computed flanking sequence.
            reference_fasta_path: Optional path to reference FASTA for real flanks.
            chrom: Optional chromosome of the source locus.
            center_pos: Optional centre position of the source locus.

        Returns:
            Dictionary containing ``mssi_score``, ``target_signal``,
            ``background_signal``, ``cell_type`` and ``target_tracks``.

        Note:
            MSSI is comparable only between sequences scored at the same locus.
            The host locus explains far more of its variance than the insert
            does, so ranking sequences scored at different loci ranks loci.

        Raises:
            KeyError: If ``cell_type`` has no configured target tracks.
        """
        if cell_type not in self.target_tracks_by_cell_type:
            raise KeyError(
                f"No Enformer target tracks configured for cell type '{cell_type}'. "
                f"Known: {sorted(self.target_tracks_by_cell_type)}."
            )
        target_tracks = self.target_tracks_by_cell_type[cell_type]

        full_seq = self.construct_context_window(
            sequence_str,
            flanking_seq=flanking_seq,
            reference_fasta_path=reference_fasta_path,
            chrom=chrom,
            center_pos=center_pos,
        )

        track_means = self._predict_track_means(full_seq)
        target_signal = track_means[target_tracks].tolist()
        bg_signal = track_means[self.background_tracks].tolist()

        return {
            "mssi_score": float(np.mean(target_signal) - np.mean(bg_signal)),
            "target_signal": target_signal,
            "background_signal": bg_signal,
            "cell_type": cell_type,
            "target_tracks": list(target_tracks),
        }

    # -------------------------------------------------------------------- batch

    def evaluate_fasta(
        self,
        fasta_path: str,
        metadata_path: str,
        reference_fasta_path: str,
        output_report_path: str = "logs/evaluation_results.json",
    ) -> Dict[str, Any]:
        """Batch evaluate a FASTA of synthetic sequences and save a JSON report.

        Every sequence must map to a metadata row giving its cell type and the
        source locus, so it can be scored against that cell type's tracks inside
        real genomic context. All three inputs are required — an unanchored,
        cell-type-agnostic MSSI would not mean anything.

        Args:
            fasta_path: Path to input FASTA file.
            metadata_path: CSV with columns ``peak_id``, ``chrom``, ``start``,
                ``end`` and ``cell_type``.
            reference_fasta_path: Path to the indexed reference FASTA.
            output_report_path: Path to save output evaluation JSON report.

        Returns:
            Evaluation summary dictionary.

        Raises:
            FileNotFoundError: If any input file does not exist.
            ValueError: If the metadata lacks a required column or a sequence.
        """
        for label, path in (("FASTA", fasta_path), ("metadata", metadata_path),
                            ("reference FASTA", reference_fasta_path)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        meta = pd.read_csv(metadata_path)
        required = {"peak_id", "chrom", "start", "end", "cell_type"}
        missing_cols = required - set(meta.columns)
        if missing_cols:
            raise ValueError(f"Metadata {metadata_path} is missing columns: {sorted(missing_cols)}")

        locus_map = {
            str(row.peak_id): {
                "chrom": str(row.chrom),
                "center_pos": int((row.start + row.end) // 2),
                "cell_type": str(row.cell_type),
            }
            for row in meta.itertuples(index=False)
        }
        logger.info("Loaded loci and cell types for %d sequences from %s.", len(locus_map), metadata_path)

        records = list(SeqIO.parse(fasta_path, "fasta"))
        unannotated = [rec.id for rec in records if rec.id not in locus_map]
        if unannotated:
            raise ValueError(
                f"{len(unannotated)} sequences have no metadata row (first: {unannotated[0]}). "
                f"Each sequence needs a cell type and a source locus to be scored."
            )

        results: Dict[str, Any] = {
            "sequences": {},
            "mssi_scores": [],
            "model_name": self.model_name,
            "background_tracks": list(self.background_tracks),
        }

        total_seqs = len(records)
        for i, rec in enumerate(records, 1):
            locus = locus_map[rec.id]
            res = self.predict_cell_specificity(
                str(rec.seq),
                cell_type=locus["cell_type"],
                reference_fasta_path=reference_fasta_path,
                chrom=locus["chrom"],
                center_pos=locus["center_pos"],
            )
            results["sequences"][rec.id] = res
            results["mssi_scores"].append(res["mssi_score"])

            if i % 10 == 0 or i == total_seqs or i <= 3:
                curr_mean = float(np.mean(results["mssi_scores"]))
                logger.info(
                    "[%3d/%d - %3d%%] Scored %s (%s) -> MSSI: %+.4f (Running Mean MSSI: %+.4f)",
                    i, total_seqs, (i * 100) // total_seqs, rec.id, locus["cell_type"], res["mssi_score"], curr_mean
                )

        results["mean_mssi"] = float(np.mean(results["mssi_scores"])) if results["mssi_scores"] else 0.0

        os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            "Evaluated %d sequences from %s. Mean MSSI: %.4f",
            len(records), fasta_path, results["mean_mssi"],
        )
        return results

    # ----------------------------------------------------------------- encoding

    @staticmethod
    def _one_hot_encode(sequence: str) -> torch.Tensor:
        """One-hot encode a nucleotide sequence.

        Args:
            sequence: DNA sequence string.

        Returns:
            Tensor of shape ``(len(sequence), 4)``. Models differ in whether they
            want this or its transpose; each subclass orients it in
            :meth:`_predict_track_means` rather than assuming a convention here.
        """
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        encoded = np.zeros((len(sequence), 4), dtype=np.float32)
        for i, b in enumerate(sequence.upper()):
            if b in mapping:
                encoded[i, mapping[b]] = 1.0
            else:
                encoded[i, :] = 0.25
        return torch.from_numpy(encoded)
