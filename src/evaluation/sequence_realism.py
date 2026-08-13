"""Sequence realism metrics for MS-ENHANCER-GEN.

Compares generated sequences against the real regulatory windows they were
trained on, using composition statistics that require no model and no GPU.

This is the cheap honesty check that belongs *before* any oracle scoring. An
oracle will happily return a number for uniformly random DNA, and that number
looks exactly like a result. These metrics say whether the generator produced
anything genome-like in the first place.

The decisive statistic is CpG observed/expected. Vertebrate genomes are strongly
CpG-depleted (roughly 0.2-0.4) through deamination of methylated cytosine, so a
generator that has learned nothing beyond base composition sits near 1.0 — the
value of a random sequence — while matching GC content perfectly.
"""

import collections
import itertools
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

ALPHABET = "ACGT"


class SequenceRealism:
    """Composition-based comparison of generated sequences against real ones."""

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        """Initialize from the model configuration.

        Args:
            config_path: Path to model configuration YAML. If the file has no
                ``evaluation.sequence_realism`` section, defaults are used.
        """
        config: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        section = config.get("evaluation", {}).get("sequence_realism", {})
        self.kmer_sizes: List[int] = section.get("kmer_sizes", [2, 4, 6])
        self.control_seed: int = int(section.get("control_seed", 0))

        if any(k < 1 for k in self.kmer_sizes):
            raise ValueError(f"kmer_sizes must all be >= 1, got {self.kmer_sizes}")

        logger.info("SequenceRealism initialised (k-mer sizes=%s).", self.kmer_sizes)

    @staticmethod
    def gc_content(sequences: Sequence[str]) -> float:
        """Fraction of G and C bases across all sequences.

        Args:
            sequences: Nucleotide strings.

        Returns:
            GC fraction in [0, 1], or 0.0 for empty input.
        """
        total = sum(len(s) for s in sequences)
        if not total:
            return 0.0
        return sum(s.count("G") + s.count("C") for s in sequences) / total

    @staticmethod
    def kmer_entropy(sequences: Sequence[str], k: int) -> float:
        """Shannon entropy of the k-mer distribution, in bits.

        A uniformly random sequence approaches the maximum of ``2 * k`` bits.
        Real genomic sequence falls below it, because some k-mers are depleted.

        Args:
            sequences: Nucleotide strings.
            k: k-mer length.

        Returns:
            Entropy in bits, or 0.0 if no k-mers could be counted.
        """
        counts: collections.Counter = collections.Counter()
        for sequence in sequences:
            counts.update(sequence[i:i + k] for i in range(len(sequence) - k + 1))

        total = sum(counts.values())
        if not total:
            return 0.0
        return -sum((v / total) * math.log2(v / total) for v in counts.values())

    @staticmethod
    def cpg_observed_expected(sequences: Sequence[str]) -> float:
        """CpG dinucleotide frequency divided by what base composition predicts.

        Args:
            sequences: Nucleotide strings.

        Returns:
            Observed/expected ratio. ~1.0 means CpG occurs as often as chance
            allows (the random-sequence value); vertebrate genomes sit far below.
            Returns 0.0 when the sequences contain no C or no G.
        """
        joined = "".join(sequences)
        if len(joined) < 2:
            return 0.0

        c_freq = joined.count("C") / len(joined)
        g_freq = joined.count("G") / len(joined)
        if c_freq == 0 or g_freq == 0:
            return 0.0

        observed = joined.count("CG") / (len(joined) - 1)
        return observed / (c_freq * g_freq)

    @staticmethod
    def median_longest_run(sequences: Sequence[str]) -> float:
        """Median over sequences of the longest single-base run.

        Catches decoder artefacts: taking an argmax over an uncertain model emits
        long homopolymer stretches that real regulatory sequence does not have.

        Args:
            sequences: Nucleotide strings.

        Returns:
            Median longest-run length, or 0.0 for empty input.
        """
        if not sequences:
            return 0.0
        runs = sorted(
            max(len(list(group)) for _, group in itertools.groupby(s)) if s else 0
            for s in sequences
        )
        return float(runs[len(runs) // 2])

    def profile(self, sequences: Sequence[str]) -> Dict[str, float]:
        """Compute every realism statistic for one set of sequences.

        Args:
            sequences: Nucleotide strings (case-insensitive).

        Returns:
            Mapping of metric name to value.

        Raises:
            ValueError: If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("Cannot profile an empty sequence set.")

        upper = [s.upper() for s in sequences]
        metrics: Dict[str, float] = {
            "n_sequences": float(len(upper)),
            "gc_content": self.gc_content(upper),
            "cpg_observed_expected": self.cpg_observed_expected(upper),
            "median_longest_run": self.median_longest_run(upper),
        }
        for k in self.kmer_sizes:
            metrics[f"kmer_entropy_k{k}"] = self.kmer_entropy(upper, k)
        return metrics

    def random_control(self, n_sequences: int, length: int, seed: Optional[int] = None) -> List[str]:
        """Generate uniformly random sequences as the "learned nothing" baseline.

        Args:
            n_sequences: How many sequences to generate.
            length: Length of each sequence.
            seed: Seed for the generator; defaults to the configured control seed.

        Returns:
            List of random nucleotide strings.
        """
        rng = random.Random(self.control_seed if seed is None else seed)
        return ["".join(rng.choice(ALPHABET) for _ in range(length)) for _ in range(n_sequences)]

    def compare(self, generated: Sequence[str], reference: Sequence[str]) -> Dict[str, Any]:
        """Score generated sequences against real ones and a random control.

        For each metric the generated value is placed on the line running from the
        random control to the reference. A ``realism`` of 1.0 means the generated
        set matches the real sequences on that statistic; 0.0 means it is no
        closer to them than random DNA is. Values can be negative when the
        generated set overshoots past random.

        Read the ratio only where the reference and the control actually differ.
        GC content is the cautionary case: real windows sit at ~49% and random DNA
        at 50%, so the denominator is about one percentage point and the resulting
        ratio swings wildly with sample size while the underlying GC values barely
        move. CpG observed/expected and the higher-order k-mer entropies separate
        real from random by a wide margin and are the metrics to trust. The raw
        profiles are returned alongside so this is always checkable.

        Args:
            generated: Generated nucleotide strings.
            reference: Real nucleotide strings from the training distribution.

        Returns:
            Dictionary with ``generated``, ``reference`` and ``random_control``
            profiles plus a per-metric ``realism`` mapping.

        Raises:
            ValueError: If either sequence set is empty.
        """
        if not generated or not reference:
            raise ValueError("Both generated and reference sequence sets must be non-empty.")

        control_length = len(generated[0])
        control = self.random_control(len(generated), control_length)

        generated_profile = self.profile(generated)
        reference_profile = self.profile(reference)
        control_profile = self.profile(control)

        realism: Dict[str, float] = {}
        for metric, reference_value in reference_profile.items():
            if metric == "n_sequences":
                continue
            span = control_profile[metric] - reference_value
            if abs(span) < 1e-12:
                # Random and real agree on this metric, so it carries no signal.
                continue
            realism[metric] = 1.0 - (generated_profile[metric] - reference_value) / span

        report = {
            "generated": generated_profile,
            "reference": reference_profile,
            "random_control": control_profile,
            "realism": realism,
        }

        logger.info(
            "Realism vs reference (1.0 = real, 0.0 = random): %s",
            {m: round(v, 3) for m, v in realism.items()},
        )
        return report
