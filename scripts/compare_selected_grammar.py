"""Ask what distinguishes oracle-selected candidates from rejected ones.

The selection experiment showed the oracle's preference partly transfers to an
unseen locus, so *something* about the sequence is being preferred. This script
asks what: it splits each locus's scored candidates into the highest- and
lowest-ranked fractions and compares them on motif arrangement, per-factor hit
counts and GC content.

Candidates within a locus come from one generator and are scored in one genomic
context, so a difference between the two tails is a property of the insert.
Ranks are taken within each locus and then pooled, because MSSI is comparable
between sequences at the same locus and not across loci.

Usage:
    python scripts/compare_selected_grammar.py --reports logs/sel0.json ... \
        --fastas data/fasta/sel0.fasta ... --fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation.motif_analyzer import MotifAnalyzer
from src.evaluation.motif_grammar import MotifGrammar
from src.utils.helpers import setup_logging
from scripts.occlusion_scan import read_fasta

logger = logging.getLogger(__name__)


def split_tails(report_path: Path, fasta_path: Path, fraction: float) -> Tuple[List[str], List[str]]:
    """Return (selected, rejected) sequences for one locus.

    Args:
        report_path: Oracle score report for the locus.
        fasta_path: FASTA the report scored.
        fraction: Fraction of the pool taken from each tail.

    Returns:
        The top and bottom fraction of sequences, ranked by MSSI.

    Raises:
        ValueError: If ``fraction`` is not in (0, 0.5], or a tail would be empty.
    """
    if not 0 < fraction <= 0.5:
        raise ValueError(f"--fraction must be in (0, 0.5], got {fraction}.")

    scores = json.loads(report_path.read_text())["sequences"]
    sequences = read_fasta(fasta_path)
    # 'real' is the natural element at this locus, not a candidate: it did not
    # come from the generator and would confound a generator-internal contrast.
    ranked = sorted(
        ((seq_id, entry["mssi_score"]) for seq_id, entry in scores.items() if seq_id != "real"),
        key=lambda item: item[1],
        reverse=True,
    )
    size = int(len(ranked) * fraction)
    if size < 1:
        raise ValueError(
            f"fraction={fraction} of {len(ranked)} candidates rounds to zero sequences per tail."
        )
    return (
        [sequences[seq_id] for seq_id, _ in ranked[:size]],
        [sequences[seq_id] for seq_id, _ in ranked[-size:]],
    )


def factor_counts(hits_per_sequence: List[List[Dict]], factors: List[str]) -> Dict[str, np.ndarray]:
    """Per-sequence hit count for each transcription factor."""
    return {
        f"hits:{factor}": np.array(
            [sum(1 for h in hits if h["tf"] == factor) for hits in hits_per_sequence], dtype=float
        )
        for factor in factors
    }


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Return BH-FDR q-values for an ascending list of p-values.

    With hundreds of correlated motif matrices, controlling the false discovery
    rate is the appropriate correction. Unlike Bonferroni it also stays reachable
    when the permutation p-value floor sits above the Bonferroni threshold.

    Args:
        p_values: p-values sorted ascending.

    Returns:
        q-values in the same order.

    Raises:
        ValueError: If the input is empty or not sorted ascending.
    """
    if not p_values:
        raise ValueError("Cannot compute q-values for an empty list of tests.")
    if any(later < earlier for earlier, later in zip(p_values, p_values[1:])):
        raise ValueError("p-values must be sorted ascending before BH correction.")

    total = len(p_values)
    q_values: List[float] = []
    running_min = 1.0
    for rank in range(total, 0, -1):
        running_min = min(running_min, p_values[rank - 1] * total / rank)
        q_values.append(running_min)
    q_values.reverse()
    return q_values


def gc_fraction(sequences: List[str]) -> np.ndarray:
    """Per-sequence GC fraction, the composition control for any motif effect."""
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in sequences], dtype=float)


def main() -> None:
    """Compare selected against rejected candidates and print the contrast."""
    parser = argparse.ArgumentParser(description="Selected vs rejected candidate comparison")
    parser.add_argument("--reports", nargs="+", required=True, help="Oracle score reports, one per locus")
    parser.add_argument("--fastas", nargs="+", required=True, help="FASTAs matching --reports, in order")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--fraction", type=float, default=0.15, help="Fraction taken from each tail")
    parser.add_argument(
        "--all_matrices", action="store_true",
        help="Scan the whole JASPAR collection instead of the configured 16-factor panel: "
             "a named panel can only find what it was pointed at",
    )
    parser.add_argument("--report_top", type=int, default=30, help="How many statistics to print")
    parser.add_argument(
        "--n_permutations", type=int, default=10000,
        help="Permutations per test. The p-value cannot go below 1/(n+1), so with many "
             "statistics this floor must sit below the correction threshold or no test can pass",
    )
    parser.add_argument("--output_csv", help="Write every statistic's result here")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging(log_file="logs/selected_grammar.log")

    if len(args.reports) != len(args.fastas):
        raise ValueError(
            f"Got {len(args.reports)} reports and {len(args.fastas)} FASTAs; they must pair up."
        )

    selected: List[str] = []
    rejected: List[str] = []
    for report, fasta in zip(args.reports, args.fastas):
        top, bottom = split_tails(Path(report), Path(fasta), args.fraction)
        selected.extend(top)
        rejected.extend(bottom)
        logger.info("%s: %d selected, %d rejected.", report, len(top), len(bottom))

    analyzer = MotifAnalyzer(
        config_path=args.config, tf_names=["ALL"] if args.all_matrices else None
    )
    grammar = MotifGrammar()
    length = len(selected[0])

    profiles = {}
    for name, sequences in (("selected", selected), ("rejected", rejected)):
        logger.info("Scanning %s (n=%d)...", name, len(sequences))
        hits = [analyzer.scan_sequence(s) for s in sequences]
        profiles[name] = grammar.profile(hits, sequence_length=length)
        # ``pssms`` holds the matrices actually loaded; ``tf_names`` may be the
        # ["ALL"] sentinel rather than a factor list.
        profiles[name].update(factor_counts(hits, list(analyzer.pssms)))
        profiles[name]["gc_fraction"] = gc_fraction(sequences)

    statistics = list(profiles["selected"])
    floor = 1.0 / (args.n_permutations + 1)
    bonferroni = 0.05 / len(statistics)
    if floor > bonferroni:
        logger.warning(
            "The p-value floor (%.2e, from %d permutations) sits above the Bonferroni threshold "
            "(%.2e for %d tests): no test can reach Bonferroni significance. Reporting "
            "Benjamini-Hochberg FDR as the primary correction.",
            floor, args.n_permutations, bonferroni, len(statistics),
        )

    logger.info("Testing %d statistics...", len(statistics))
    results = []
    for statistic in statistics:
        a, b = profiles["selected"][statistic], profiles["rejected"][statistic]
        p = grammar.permutation_p(a, b, n_permutations=args.n_permutations, seed=args.seed)
        results.append((p, statistic, float(np.nanmean(a)), float(np.nanmean(b))))
    results.sort()

    total = len(results)
    q_values = benjamini_hochberg([p for p, *_ in results])

    print(f"\nSelected vs rejected, {len(selected)} per group, top/bottom {args.fraction:.0%} of each locus")
    print(f"{total} statistics tested, {args.n_permutations} permutations (p floor {floor:.2e})")
    print(f"Bonferroni threshold p < {bonferroni:.2e}"
          f"{'  [unreachable: below the p floor]' if floor > bonferroni else ''}")
    print(f"Passing BH-FDR q < 0.05: {sum(1 for q in q_values if q < 0.05)}\n")
    header = f"{'statistic':40s}{'selected':>12s}{'rejected':>12s}{'delta':>10s}{'p':>10s}{'q':>10s}"
    print(header)
    print("-" * len(header))
    for (p, statistic, mean_a, mean_b), q in list(zip(results, q_values))[: args.report_top]:
        print(f"{statistic:40s}{mean_a:12.3f}{mean_b:12.3f}{mean_a - mean_b:10.3f}{p:10.4f}{q:10.4f}")

    if args.output_csv:
        pd.DataFrame(
            [
                {"statistic": s, "selected": a, "rejected": b, "delta": a - b, "p": p, "q": q}
                for (p, s, a, b), q in zip(results, q_values)
            ]
        ).to_csv(args.output_csv, index=False)
        logger.info("Wrote all %d results to %s.", total, args.output_csv)


if __name__ == "__main__":
    main()
