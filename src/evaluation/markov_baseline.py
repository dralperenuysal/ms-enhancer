"""Order-k Markov chain generator: the baseline any neural generator has to beat.

A Markov chain of order k reproduces every k+1-mer frequency of its training set
by construction, so it matches GC content, CpG observed/expected and PWM hit
densities almost exactly. Those are precisely the statistics the realism and
motif checks measure. If a trained network does not beat this baseline on them,
the network has learned nothing the (k+1)-mer table does not already contain,
and the honest conclusion is that a lookup table would have done the job.

The chain is fitted per cell type, mirroring how the conditional models are
conditioned, so the comparison is like for like.
"""

import collections
import logging
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ALPHABET = "ACGT"
BASE_INDEX = {base: i for i, base in enumerate(ALPHABET)}


class MarkovBaseline:
    """Fits an order-k nucleotide Markov chain and samples sequences from it."""

    def __init__(self, order: int = 6, pseudocount: float = 1.0) -> None:
        """Initialize an unfitted chain.

        Args:
            order: Number of preceding bases conditioned on. Order 0 reproduces
                base composition only; order 6 reproduces all 7-mer frequencies.
            pseudocount: Added to every transition count, so unseen contexts fall
                back to a smoothed distribution instead of dividing by zero.

        Raises:
            ValueError: If ``order`` is negative or ``pseudocount`` is not positive.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if pseudocount <= 0:
            raise ValueError(f"pseudocount must be > 0, got {pseudocount}")

        self.order = order
        self.pseudocount = pseudocount
        self.transitions: Dict[str, np.ndarray] = {}
        self.prefixes: List[str] = []
        self._prefix_weights: np.ndarray = np.empty(0)

    def fit(self, sequences: Sequence[str]) -> "MarkovBaseline":
        """Count transitions in the training sequences.

        Args:
            sequences: Nucleotide strings. Positions containing anything outside
                ACGT break the context and are skipped, so ambiguous reference
                bases never enter the model.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If no sequence is long enough to yield a transition.
        """
        counts: Dict[str, np.ndarray] = {}
        prefix_counts: collections.Counter = collections.Counter()

        for sequence in sequences:
            upper = sequence.upper()
            for i in range(len(upper) - self.order):
                context = upper[i:i + self.order]
                nxt = upper[i + self.order]
                if nxt not in BASE_INDEX or any(b not in BASE_INDEX for b in context):
                    continue
                if context not in counts:
                    counts[context] = np.zeros(4, dtype=np.float64)
                counts[context][BASE_INDEX[nxt]] += 1.0

            # Sequence starts, so sampling begins from a context the data supports.
            head = upper[:self.order]
            if self.order == 0 or all(b in BASE_INDEX for b in head):
                prefix_counts[head] += 1

        if not counts:
            raise ValueError(
                f"No order-{self.order} transitions found. Sequences must be longer "
                f"than the chain order and contain unambiguous ACGT bases."
            )

        self.transitions = {
            context: (row + self.pseudocount) / (row.sum() + 4 * self.pseudocount)
            for context, row in counts.items()
        }
        self.prefixes = list(prefix_counts)
        weights = np.array([prefix_counts[p] for p in self.prefixes], dtype=np.float64)
        self._prefix_weights = weights / weights.sum()

        logger.info(
            "Fitted order-%d Markov chain on %d sequences: %d observed contexts of %d possible.",
            self.order, len(sequences), len(self.transitions), 4 ** self.order,
        )
        return self

    def _next_distribution(self, context: str) -> np.ndarray:
        """Transition row for a context, backing off to shorter ones when unseen."""
        for start in range(len(context) + 1):
            row = self.transitions.get(context[start:])
            if row is not None:
                return row
        return np.full(4, 0.25)

    def sample(self, num_sequences: int, length: int, seed: int) -> List[str]:
        """Draw sequences from the fitted chain.

        Args:
            num_sequences: How many sequences to generate.
            length: Length of each sequence in bp.
            seed: Seed for the sampler, so output is reproducible.

        Returns:
            List of nucleotide strings.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``length`` is shorter than the chain order.
        """
        if not self.transitions:
            raise RuntimeError("MarkovBaseline.sample() called before fit().")
        if length < self.order:
            raise ValueError(f"length {length} is shorter than chain order {self.order}")

        rng = np.random.default_rng(seed)
        sequences: List[str] = []

        for _ in range(num_sequences):
            bases = list(self.prefixes[rng.choice(len(self.prefixes), p=self._prefix_weights)])
            while len(bases) < length:
                context = "".join(bases[len(bases) - self.order:]) if self.order else ""
                bases.append(ALPHABET[rng.choice(4, p=self._next_distribution(context))])
            sequences.append("".join(bases[:length]))

        return sequences
