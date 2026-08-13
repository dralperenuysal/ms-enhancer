"""Build an oracle scoring set from measured MPRA elements.

Everything in this project so far is scored against an oracle. This is the one
comparison that is not: MPRA elements come with *measured* regulatory activity,
so scoring them lets the oracle be checked against something outside itself.

Two design choices matter.

**All elements are scored at one fixed locus.** MSSI variance is dominated by the
host locus roughly 45-fold, so scoring elements at their own native loci would
rank loci rather than elements. A fixed locus removes that term, and it also
matches what the MPRA measures, since an episomal reporter carries no genomic
context of its own.

**The sample is stratified but reports its random part separately.** A sample
enriched for extremes gives power for a group contrast but inflates a
correlation, so a plain random subset is drawn alongside and is the only part
used for correlation estimates.

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
from scripts.occlusion_scan import read_fasta

logger = logging.getLogger(__name__)


def stratified_sample(measurements: pd.DataFrame, column: str, n_random: int,
                      n_extreme: int, seed: int) -> pd.DataFrame:
    """Draw a random subset plus both extremes of ``column``.

    Args:
        measurements: One row per element, with ``column`` present.
        column: The measured quantity to stratify on.
        n_random: Size of the unbiased random subset.
        n_extreme: How many to take from each tail.
        seed: Seed for the random subset.

    Returns:
        The sampled rows with a ``stratum`` column of ``random``, ``top`` or
        ``bottom``. Elements are never duplicated across strata.

    Raises:
        ValueError: If the request exceeds the available rows.
    """
    if n_random + 2 * n_extreme > len(measurements):
        raise ValueError(
            f"Asked for {n_random + 2 * n_extreme} elements but only {len(measurements)} exist."
        )

    ordered = measurements.sort_values(column)
    bottom = ordered.head(n_extreme).assign(stratum="bottom")
    top = ordered.tail(n_extreme).assign(stratum="top")
    remaining = measurements.drop(index=bottom.index.union(top.index))
    random = remaining.sample(n=n_random, random_state=seed).assign(stratum="random")
    return pd.concat([random, top, bottom]).reset_index(drop=True)


def main() -> None:
    """Write the FASTA and metadata for scoring MPRA elements at a fixed locus."""
    parser = argparse.ArgumentParser(description="Build an MPRA element scoring set")
    parser.add_argument("--measurements", required=True, help="CSV with ID and measured activity")
    parser.add_argument("--oligos_fasta", required=True, help="FASTA of the tested oligo sequences")
    parser.add_argument("--host_metadata", required=True, help="Metadata CSV supplying the fixed locus")
    parser.add_argument("--column", default="spec", help="Measured column to stratify on")
    parser.add_argument("--n_random", type=int, default=1000)
    parser.add_argument("--n_extreme", type=int, default=500)
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--output_key", required=True, help="Where to write id -> measurement -> stratum")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    measurements = pd.read_csv(args.measurements)
    sequences = read_fasta(Path(args.oligos_fasta))
    measurements = measurements[measurements.ID.isin(sequences)]
    if measurements.empty:
        raise ValueError(
            f"No element in {args.measurements} has a sequence in {args.oligos_fasta}."
        )

    sample = stratified_sample(measurements, args.column, args.n_random, args.n_extreme, args.seed)
    host = pd.read_csv(args.host_metadata).iloc[0]

    records: List[Tuple[str, str]] = [(row.ID, sequences[row.ID]) for row in sample.itertuples()]
    Path(args.output_fasta).write_text("".join(f">{n}\n{s}\n" for n, s in records))
    pd.DataFrame({
        "peak_id": [n for n, _ in records],
        "chrom": host["chrom"], "start": host["start"], "end": host["end"],
        "cell_type": host["cell_type"],
    }).to_csv(args.output_metadata, index=False)
    sample.to_csv(args.output_key, index=False)

    lengths = {len(s) for _, s in records}
    logger.info(
        "Wrote %d elements (%s) at %s:%s-%s, lengths %s. Strata: %s.",
        len(records), args.column, host["chrom"], host["start"], host["end"],
        f"{min(lengths)}-{max(lengths)} bp", sample.stratum.value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
