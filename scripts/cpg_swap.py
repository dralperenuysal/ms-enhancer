"""Change a sequence's CpG content without changing anything else.

CpG observed/expected separates the selected from the rejected candidates more
strongly than any motif statistic, but that is a correlation, and the density
correlation already failed its causal test. This builds the intervention.

Swapping a ``GC`` dinucleotide to ``CG`` adds one CpG; swapping ``CG`` to ``GC``
removes one. Either way the sequence keeps its exact length and its exact count
of every single base, so GC content, purine content and every mononucleotide
frequency are identical between an intact sequence and its swapped variants. The
control swaps ``AT``/``TA`` instead, which is the same kind of edit at the same
rate while leaving CpG untouched.

The direction of the prediction depends on the host locus: CpG depletion tracks
higher MSSI at four of the five selection loci and the reverse at chr4
(see the report), so a locus must be named rather than assumed.

Writes a FASTA plus metadata; scoring is done by ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import setup_logging
from scripts.occlusion_scan import load_ranked_sequences

logger = logging.getLogger(__name__)


def swap_dinucleotides(sequence: str, source: str, count: int, seed: int) -> Tuple[str, int]:
    """Reverse ``count`` randomly chosen occurrences of the dinucleotide ``source``.

    Occurrences are chosen without replacement and cannot overlap, so each swap
    is independent of the others.

    Args:
        sequence: Sequence to edit.
        source: Dinucleotide to reverse, e.g. ``"GC"`` to create CpGs.
        count: How many occurrences to swap; all of them if more are requested
            than exist.
        seed: Seed for the choice of occurrences.

    Returns:
        The edited sequence and the number of swaps actually made.

    Raises:
        ValueError: If ``source`` is not two distinct bases.
    """
    if len(source) != 2 or source[0] == source[1]:
        raise ValueError(f"source must be two distinct bases, got '{source}'.")

    positions = [i for i in range(len(sequence) - 1) if sequence[i:i + 2] == source]
    if not positions:
        return sequence, 0

    rng = np.random.default_rng(seed)
    rng.shuffle(positions)
    chosen: List[int] = []
    taken: set = set()
    for position in positions:
        if len(chosen) >= count:
            break
        if position in taken or position - 1 in taken:
            continue
        chosen.append(position)
        taken.add(position)

    bases = list(sequence)
    for position in chosen:
        bases[position], bases[position + 1] = bases[position + 1], bases[position]
    return "".join(bases), len(chosen)


def cpg_count(sequence: str) -> int:
    """Number of CpG dinucleotides, the quantity the swap is meant to move."""
    return sequence.count("CG")


def main() -> None:
    """Write CpG-raised, CpG-lowered and composition-matched control variants."""
    parser = argparse.ArgumentParser(description="CpG intervention over oracle-scored candidates")
    parser.add_argument("--score_report", required=True)
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--metadata", required=True, help="Metadata CSV giving the host locus")
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--top_k", type=int, default=40, help="How many top-ranked candidates to edit")
    parser.add_argument("--swaps", type=int, default=20, help="Dinucleotide swaps per variant")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    ranked = load_ranked_sequences(Path(args.score_report), Path(args.input_fasta))[: args.top_k]
    host = pd.read_csv(args.metadata).iloc[0]

    records: List[Tuple[str, str]] = []
    raised_total = lowered_total = 0
    for rank, (seq_id, sequence, _) in enumerate(ranked):
        seed = args.seed + rank
        more, n_more = swap_dinucleotides(sequence, "GC", args.swaps, seed)
        less, n_less = swap_dinucleotides(sequence, "CG", args.swaps, seed)
        # Two controls: the same edit rate on a dinucleotide that carries no CpG.
        control_a, _ = swap_dinucleotides(sequence, "AT", args.swaps, seed)
        control_b, _ = swap_dinucleotides(sequence, "TA", args.swaps, seed)

        records += [
            (f"s{rank:02d}_intact", sequence),
            (f"s{rank:02d}_cpg_up", more),
            (f"s{rank:02d}_cpg_down", less),
            (f"s{rank:02d}_ctl_at", control_a),
            (f"s{rank:02d}_ctl_ta", control_b),
        ]
        raised_total += cpg_count(more) - cpg_count(sequence)
        lowered_total += cpg_count(sequence) - cpg_count(less)
        if sorted(more) != sorted(sequence) or sorted(less) != sorted(sequence):
            raise RuntimeError(f"Swap changed the base composition of {seq_id}; this is a bug.")
        logger.info("%s: +%d / -%d CpG (%d / %d swaps).",
                    seq_id, cpg_count(more) - cpg_count(sequence),
                    cpg_count(sequence) - cpg_count(less), n_more, n_less)

    Path(args.output_fasta).write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    pd.DataFrame({
        "peak_id": [name for name, _ in records],
        "chrom": host["chrom"], "start": host["start"], "end": host["end"],
        "cell_type": host["cell_type"],
    }).to_csv(args.output_metadata, index=False)
    logger.info(
        "Wrote %d sequences: mean +%.1f CpG raised, -%.1f CpG lowered per candidate.",
        len(records), raised_total / len(ranked), lowered_total / len(ranked),
    )


if __name__ == "__main__":
    main()
