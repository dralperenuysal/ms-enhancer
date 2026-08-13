"""Motif Analyzer module for MS-ENHANCER-GEN.

Scans synthetic DNA sequences for transcription factor binding sites using real
position weight matrices from JASPAR.

The TF list is read from ``configs/model_config.yaml`` and is the single source
of truth: every named TF must resolve to a JASPAR matrix or initialisation fails.
There is no built-in fallback pattern table, because a consensus regex answers a
different question than a PWM does — it reports an exact-string hit with no
score, so a strong site and a barely-tolerated one are indistinguishable, and
nothing can be ranked or thresholded.

``pyjaspar`` ships the JASPAR database locally, so scanning performs no network
access.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from Bio import SeqIO

logger = logging.getLogger(__name__)


class MotifDatabaseError(RuntimeError):
    """Raised when a configured TF cannot be resolved to a JASPAR matrix."""


class MotifAnalyzer:
    """Scans DNA sequences for TF binding sites using JASPAR position weight matrices."""

    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        tf_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize MotifAnalyzer from the model configuration.

        Args:
            config_path: Path to model configuration YAML file.
            tf_names: Overrides the configured factor list. The single-element
                list ``["ALL"]`` loads the entire JASPAR collection instead of a
                named panel, which is what an unbiased scan needs; a named panel
                can only find what it was pointed at.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            ValueError: If no TFs are configured or the threshold is out of range.
            MotifDatabaseError: If ``pyjaspar`` is unavailable or a configured TF
                has no matrix in the requested JASPAR release.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        motif_cfg = config.get("evaluation", {}).get("motif_analysis", {})
        self.tf_names: List[str] = tf_names if tf_names is not None else motif_cfg.get("tfs", [])
        self.jaspar_release: str = motif_cfg.get("jaspar_release", "JASPAR2024")
        self.collection: str = motif_cfg.get("jaspar_collection", "CORE")
        self.species: Optional[str] = motif_cfg.get("jaspar_species", "9606")
        self.relative_threshold: float = float(motif_cfg.get("relative_score_threshold", 0.8))

        if not self.tf_names:
            raise ValueError(
                f"No transcription factors configured. Populate "
                f"evaluation.motif_analysis.tfs in {config_path}."
            )
        if not 0.0 < self.relative_threshold <= 1.0:
            raise ValueError(
                f"relative_score_threshold must be in (0, 1], got {self.relative_threshold}."
            )

        self.pssms: Dict[str, Any] = {}
        self.matrix_ids: Dict[str, str] = {}
        self._thresholds: Dict[str, float] = {}
        self._load_matrices()

    def _load_matrices(self) -> None:
        """Fetch a JASPAR PWM for every configured TF and convert it to a PSSM.

        Raises:
            MotifDatabaseError: If ``pyjaspar`` is missing or a TF has no matrix.
        """
        try:
            from Bio.motifs.jaspar import calculate_pseudocounts
            from pyjaspar import jaspardb
        except ImportError as error:
            raise MotifDatabaseError(
                "pyjaspar is required for motif analysis (`pip install pyjaspar`). "
                "Scanning with consensus regexes instead is not supported."
            ) from error

        database = jaspardb(release=self.jaspar_release)

        if self.tf_names == ["ALL"]:
            collection = database.fetch_motifs(
                collection=self.collection, species=self.species, all_versions=False
            )
            if not collection:
                raise MotifDatabaseError(
                    f"{self.jaspar_release} {self.collection} returned no matrices for "
                    f"species '{self.species}'."
                )
            # Names repeat across matrices (dimers, variants), so the matrix id
            # disambiguates; the plain name stays first for readable output.
            named = [(f"{m.name}|{m.matrix_id}", m) for m in collection]
        else:
            named = []
            for tf_name in self.tf_names:
                matrices = database.fetch_motifs_by_name(tf_name)
                if not matrices:
                    raise MotifDatabaseError(
                        f"No {self.jaspar_release} matrix found for transcription factor "
                        f"'{tf_name}'. Check the spelling in evaluation.motif_analysis.tfs "
                        f"against JASPAR, or remove it."
                    )
                # fetch_motifs_by_name returns versions newest-first; take the current one.
                named.append((tf_name, matrices[0]))

        for tf_name, matrix in named:
            matrix.pseudocounts = calculate_pseudocounts(matrix)
            pssm = matrix.pssm

            self.pssms[tf_name] = pssm
            self.matrix_ids[tf_name] = matrix.matrix_id
            # A log-odds score is only interpretable relative to what this matrix
            # can produce, so the cutoff is placed on the normalised 0-1 scale.
            self._thresholds[tf_name] = pssm.min + self.relative_threshold * (pssm.max - pssm.min)

        if self.tf_names == ["ALL"]:
            # Enumerating ~900 matrices would make the log unreadable.
            logger.info(
                "Loaded %d %s %s matrices (whole collection) at relative score threshold %.2f.",
                len(self.pssms), self.jaspar_release, self.collection, self.relative_threshold,
            )
            return

        logger.info(
            "Loaded %d %s matrices (%s) at relative score threshold %.2f: %s",
            len(self.pssms),
            self.jaspar_release,
            ", ".join(f"{tf}={mid}" for tf, mid in self.matrix_ids.items()),
            self.relative_threshold,
            list(self.pssms),
        )

    def scan_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """Scan a DNA sequence on both strands for configured TF binding sites.

        Args:
            sequence: Input nucleotide sequence string.

        Returns:
            List of hits, each with ``tf``, ``matrix_id``, ``start``, ``end``,
            ``strand``, ``match_seq``, ``score`` (log-odds, in bits) and
            ``relative_score`` (0-1, where 1 is the matrix optimum).
        """
        upper = sequence.upper()
        hits: List[Dict[str, Any]] = []

        for tf_name, pssm in self.pssms.items():
            span = pssm.length
            score_range = pssm.max - pssm.min

            for position, score in pssm.search(upper, threshold=self._thresholds[tf_name], both=True):
                # Bio.motifs reports reverse-strand hits as negative offsets.
                strand = "+" if position >= 0 else "-"
                start = position if position >= 0 else len(upper) + position
                hits.append({
                    "tf": tf_name,
                    "matrix_id": self.matrix_ids[tf_name],
                    "start": int(start),
                    "end": int(start) + span,
                    "strand": strand,
                    "match_seq": upper[start:start + span],
                    "score": float(score),
                    "relative_score": float((score - pssm.min) / score_range) if score_range else 1.0,
                })

        return sorted(hits, key=lambda hit: (hit["start"], hit["tf"]))

    def analyze_fasta(self, fasta_path: str) -> Dict[str, Any]:
        """Scan all sequences in a FASTA file and compile a motif summary report.

        Args:
            fasta_path: Path to FASTA file.

        Returns:
            Dictionary with per-sequence hits, aggregate per-TF counts, the mean
            relative score per TF, and the matrix provenance.

        Raises:
            FileNotFoundError: If ``fasta_path`` does not exist.
        """
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        records = list(SeqIO.parse(fasta_path, "fasta"))
        results: Dict[str, Any] = {
            "sequences": {},
            "tf_counts": {tf: 0 for tf in self.pssms},
            "tf_mean_relative_score": {},
            "jaspar_release": self.jaspar_release,
            "matrix_ids": dict(self.matrix_ids),
            "relative_score_threshold": self.relative_threshold,
        }
        score_sums: Dict[str, float] = {tf: 0.0 for tf in self.pssms}

        for record in records:
            seq_hits = self.scan_sequence(str(record.seq))
            results["sequences"][record.id] = seq_hits

            for hit in seq_hits:
                results["tf_counts"][hit["tf"]] += 1
                score_sums[hit["tf"]] += hit["relative_score"]

        results["tf_mean_relative_score"] = {
            tf: (score_sums[tf] / count if (count := results["tf_counts"][tf]) else 0.0)
            for tf in self.pssms
        }

        logger.info(
            "Scanned %d sequences from %s. TF hit counts: %s",
            len(records), fasta_path, results["tf_counts"],
        )
        return results
