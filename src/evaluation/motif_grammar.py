"""Motif arrangement statistics: what counting motifs cannot see.

Motif *density* says how many binding sites a sequence carries. It says nothing
about how they are arranged, and arrangement is most of what distinguishes a
regulatory element from a sequence with the same composition. An order-k Markov
chain reproduces every (k+1)-mer frequency of its training set by construction,
so it matches density almost exactly — but a chain conditioned on 6 preceding
bases cannot represent a constraint between two sites 200 bp apart.

This module measures the arrangement statistics that separate those cases:

* **Homotypic clustering** — repeated sites for the same factor packed close
  together, a well-described feature of developmental and immune enhancers that
  a short-range chain has no mechanism to produce.
* **Heterotypic co-occurrence** — how many distinct factors are represented, and
  how often two different factors sit within binding distance.
* **Positional distribution** — whether sites concentrate near the peak summit
  (the window centre) or spread uniformly.

All statistics are computed per sequence so that two sets can be compared with
the same permutation test used elsewhere in the project.
"""

import itertools
import logging
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class MotifGrammar:
    """Arrangement statistics over the motif hits of a sequence set."""

    def __init__(self, cluster_window: int = 100, pair_distance: int = 50) -> None:
        """Initialize with the distance scales the statistics are defined at.

        Args:
            cluster_window: Width in bp of the sliding window used to count
                homotypic clusters.
            pair_distance: Maximum separation in bp for two sites of different
                factors to count as co-occurring.

        Raises:
            ValueError: If either distance is not positive.
        """
        if cluster_window < 1 or pair_distance < 1:
            raise ValueError(
                f"cluster_window and pair_distance must be >= 1, got "
                f"{cluster_window} and {pair_distance}."
            )
        self.cluster_window = cluster_window
        self.pair_distance = pair_distance

    @staticmethod
    def _centres(hits: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group hit centre positions by transcription factor."""
        by_tf: Dict[str, List[int]] = {}
        for hit in hits:
            by_tf.setdefault(hit["tf"], []).append((hit["start"] + hit["end"]) // 2)
        return {tf: sorted(positions) for tf, positions in by_tf.items()}

    def max_homotypic_cluster(self, hits: Sequence[Dict[str, Any]]) -> int:
        """Largest number of same-factor sites falling in one ``cluster_window``.

        Args:
            hits: Motif hits for a single sequence, as returned by
                ``MotifAnalyzer.scan_sequence``.

        Returns:
            The maximum count over all factors and all window placements; 0 for
            a sequence with no hits.
        """
        best = 0
        for positions in self._centres(hits).values():
            # Two pointers over sorted positions: the widest run within the window.
            left = 0
            for right, position in enumerate(positions):
                while position - positions[left] > self.cluster_window:
                    left += 1
                best = max(best, right - left + 1)
        return best

    @staticmethod
    def distinct_factors(hits: Sequence[Dict[str, Any]]) -> int:
        """Number of distinct transcription factors with at least one site."""
        return len({hit["tf"] for hit in hits})

    def heterotypic_pairs(self, hits: Sequence[Dict[str, Any]]) -> int:
        """Count pairs of sites for *different* factors within ``pair_distance``.

        Args:
            hits: Motif hits for a single sequence.

        Returns:
            Number of qualifying unordered pairs.
        """
        centres = [((hit["start"] + hit["end"]) // 2, hit["tf"]) for hit in hits]
        centres.sort()
        pairs = 0
        for i, (position, tf) in enumerate(centres):
            for other_position, other_tf in centres[i + 1:]:
                if other_position - position > self.pair_distance:
                    break
                if other_tf != tf:
                    pairs += 1
        return pairs

    @staticmethod
    def mean_distance_from_centre(hits: Sequence[Dict[str, Any]], sequence_length: int) -> float:
        """Mean absolute distance of sites from the middle of the window.

        Windows are centred on the called peak summit, so sites concentrated near
        the centre indicate positional preference relative to the accessible
        region rather than uniform scattering.

        Args:
            hits: Motif hits for a single sequence.
            sequence_length: Length of the sequence in bp.

        Returns:
            Mean distance in bp, or ``nan`` when the sequence has no hits.
        """
        if not hits:
            return float("nan")
        middle = sequence_length / 2.0
        return float(np.mean([abs((h["start"] + h["end"]) / 2.0 - middle) for h in hits]))

    def profile(self, hits_per_sequence: Sequence[Sequence[Dict[str, Any]]],
                sequence_length: int) -> Dict[str, np.ndarray]:
        """Compute every arrangement statistic, one value per sequence.

        Args:
            hits_per_sequence: Motif hits for each sequence in the set.
            sequence_length: Length of the sequences in bp.

        Returns:
            Mapping of statistic name to a per-sequence array, suitable for the
            permutation tests used elsewhere.

        Raises:
            ValueError: If the set is empty.
        """
        if not hits_per_sequence:
            raise ValueError("Cannot profile an empty sequence set.")

        return {
            "max_homotypic_cluster": np.array(
                [self.max_homotypic_cluster(h) for h in hits_per_sequence], dtype=float
            ),
            "distinct_factors": np.array(
                [self.distinct_factors(h) for h in hits_per_sequence], dtype=float
            ),
            "heterotypic_pairs": np.array(
                [self.heterotypic_pairs(h) for h in hits_per_sequence], dtype=float
            ),
            "mean_distance_from_centre": np.array(
                [self.mean_distance_from_centre(h, sequence_length) for h in hits_per_sequence],
                dtype=float,
            ),
            "total_hits": np.array([len(h) for h in hits_per_sequence], dtype=float),
        }

    @staticmethod
    def permutation_p(a: np.ndarray, b: np.ndarray, n_permutations: int = 10000,
                      seed: int = 0) -> float:
        """Two-sided permutation test on a difference in means.

        Args:
            a: Per-sequence values for the first set.
            b: Per-sequence values for the second set.
            n_permutations: Number of label shuffles.
            seed: Seed, so the p-value is reproducible.

        Returns:
            The p-value, floored at ``1 / (n_permutations + 1)``.
        """
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        observed = abs(a.mean() - b.mean())
        pooled = np.concatenate([a, b])
        rng = np.random.default_rng(seed)
        extreme = sum(
            abs(shuffled[:len(a)].mean() - shuffled[len(a):].mean()) >= observed - 1e-12
            for shuffled in (rng.permutation(pooled) for _ in range(n_permutations))
        )
        return (extreme + 1) / (n_permutations + 1)
