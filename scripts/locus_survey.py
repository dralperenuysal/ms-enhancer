"""Run one insert-level intervention across many host loci at once.

The CpG intervention changes MSSI in opposite directions at different loci, so
the sign is a property of the host rather than of the insert. That result rests
on three loci. This builds the survey version: **one fixed set of candidate
inserts**, each carrying the same intervention, placed at many different loci.

Holding the inserts fixed is the point. Any difference in the measured effect
between two loci is then attributable to the host alone, with no contribution
from which sequences happened to be drawn.

Each sequence carries its own locus row in the metadata, so the whole survey is
a single ``evaluate.py`` run. Analysis regresses the per-locus effect on host
properties, which is what turns the anomaly into a design rule or refutes it.

Usage:
    python scripts/locus_survey.py --windows_fasta data/fasta/ms_windows_1000bp.fasta \
        --metadata data/fasta/ms_windows_metadata.csv --cell_type CD4_T_cell \
        --candidates_fasta data/fasta/sel0.fasta --n_loci 24 --n_inserts 20 \
        --output_fasta data/fasta/survey.fasta --output_metadata data/fasta/survey_metadata.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import setup_logging
from scripts.cpg_swap import cpg_count, swap_dinucleotides
from scripts.occlusion_scan import read_fasta

logger = logging.getLogger(__name__)


def cpg_observed_expected(sequence: str) -> float:
    """CpG observed/expected, the host property the survey regresses against.

    Args:
        sequence: DNA sequence.

    Returns:
        The ratio, or ``nan`` when the sequence contains no C or no G.
    """
    c, g = sequence.count("C"), sequence.count("G")
    if not c or not g:
        return float("nan")
    return cpg_count(sequence) * len(sequence) / (c * g)


def choose_loci(windows: pd.DataFrame, n_loci: int, seed: int) -> pd.DataFrame:
    """Pick loci spread across the peak-score range rather than at random.

    Peak score is the only pre-scoring proxy available for how accessible a locus
    is, and the three loci tested so far differ enormously in baseline MSSI, so
    spreading over the range is more informative than sampling uniformly.

    Args:
        windows: Candidate host windows for one cell type.
        n_loci: How many loci to pick.
        seed: Seed for the within-stratum draw.

    Returns:
        The chosen windows.

    Raises:
        ValueError: If fewer windows are available than requested.
    """
    if len(windows) < n_loci:
        raise ValueError(f"Asked for {n_loci} loci but only {len(windows)} windows are available.")

    ordered = windows.sort_values("peak_score").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(ordered), n_loci + 1).astype(int)
    picks = [int(rng.integers(start, end)) for start, end in zip(edges[:-1], edges[1:]) if end > start]
    return ordered.iloc[picks].reset_index(drop=True)


def main() -> None:
    """Write the survey FASTA and its per-sequence locus metadata."""
    parser = argparse.ArgumentParser(description="CpG intervention across many host loci")
    parser.add_argument("--windows_fasta", required=True, help="Real windows, used as host loci")
    parser.add_argument("--metadata", required=True, help="Metadata for those windows")
    parser.add_argument("--candidates_fasta", required=True, help="Inserts to place at every locus")
    parser.add_argument("--cell_type", default="CD4_T_cell")
    parser.add_argument("--n_loci", type=int, default=24)
    parser.add_argument("--n_inserts", type=int, default=20)
    parser.add_argument("--swaps", type=int, default=40, help="GC->CG reversals per insert")
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--output_hosts", required=True, help="Per-locus host properties CSV")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    metadata = pd.read_csv(args.metadata)
    windows = metadata[metadata.cell_type == args.cell_type]
    if windows.empty:
        raise ValueError(f"No windows for cell type '{args.cell_type}' in {args.metadata}.")
    loci = choose_loci(windows, args.n_loci, args.seed)

    window_seqs = read_fasta(Path(args.windows_fasta))
    candidates = read_fasta(Path(args.candidates_fasta))
    inserts = [
        (name, candidates[name])
        for name in sorted(candidates)
        if name != "real"
    ][: args.n_inserts]
    if len(inserts) < args.n_inserts:
        raise ValueError(
            f"Asked for {args.n_inserts} inserts but {args.candidates_fasta} holds {len(inserts)}."
        )

    # Every locus receives the identical insert set, so locus is the only variable.
    variants: List[Tuple[str, str]] = []
    for index, (name, sequence) in enumerate(inserts):
        raised, _ = swap_dinucleotides(sequence, "GC", args.swaps, args.seed + index)
        control, _ = swap_dinucleotides(sequence, "AT", args.swaps, args.seed + index)
        if sorted(raised) != sorted(sequence) or sorted(control) != sorted(sequence):
            raise RuntimeError(f"Swap changed the base composition of {name}; this is a bug.")
        variants += [
            (f"i{index:02d}_intact", sequence),
            (f"i{index:02d}_cpg_up", raised),
            (f"i{index:02d}_ctl", control),
        ]

    records: List[Tuple[str, str]] = []
    rows: List[Dict[str, object]] = []
    host_rows: List[Dict[str, object]] = []
    for locus_index, host in loci.iterrows():
        host_seq = window_seqs.get(host["peak_id"])
        if host_seq is None:
            logger.warning("No sequence for host %s; skipping.", host["peak_id"])
            continue
        for name, sequence in variants:
            seq_id = f"L{locus_index:02d}_{name}"
            records.append((seq_id, sequence))
            rows.append({
                "peak_id": seq_id, "chrom": host["chrom"], "start": host["start"],
                "end": host["end"], "cell_type": args.cell_type,
            })
        host_rows.append({
            "locus": f"L{locus_index:02d}",
            "peak_id": host["peak_id"],
            "chrom": host["chrom"],
            "start": host["start"],
            "peak_score": host["peak_score"],
            "fold_enrichment": host["fold_enrichment"],
            "host_cpg_oe": cpg_observed_expected(host_seq),
            "host_gc": (host_seq.count("G") + host_seq.count("C")) / len(host_seq),
        })

    if not records:
        raise ValueError("No host locus had a sequence; nothing to write.")

    Path(args.output_fasta).write_text("".join(f">{n}\n{s}\n" for n, s in records))
    pd.DataFrame(rows).to_csv(args.output_metadata, index=False)
    pd.DataFrame(host_rows).to_csv(args.output_hosts, index=False)
    logger.info(
        "Wrote %d sequences: %d loci x %d inserts x 3 variants. Mean +%.1f CpG per raised insert.",
        len(records), len(host_rows), len(inserts),
        np.mean([cpg_count(s) for n, s in variants if n.endswith("cpg_up")])
        - np.mean([cpg_count(s) for n, s in variants if n.endswith("intact")]),
    )


if __name__ == "__main__":
    main()
