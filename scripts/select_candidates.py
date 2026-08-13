"""Rank scored candidate sequences by MSSI and keep the best ones.

This is the selection half of the design pipeline. The generator (an order-k
Markov chain fitted per cell type, see configs/model_config.yaml) produces
sequences that are realistic and carry the right cell-type composition but are
not optimised for anything. The oracle scores them, and this script keeps the
top of that ranking.

The real windows the candidates were generated from are scored the same way and
reported alongside, because "better than the element nature put at this locus"
is the only comparison that makes the ranking mean anything. A high MSSI on its
own is not evidence of anything: the oracle returns a number for random DNA too.

Usage:
    python scripts/select_candidates.py --report logs/eval_markov_CD4.json \
        --fasta data/fasta/markov_CD4.fasta --top_k 200 \
        --out_fasta data/fasta/selected_CD4.fasta
"""

import argparse
import json
import logging
import os
from typing import Dict

import pandas as pd
from Bio import SeqIO

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Select top-MSSI candidates from an oracle report")
    parser.add_argument("--report", required=True, help="JSON written by EnformerOracle.evaluate_fasta")
    parser.add_argument("--fasta", required=True, help="Candidate FASTA that was scored")
    parser.add_argument("--metadata", default=None, help="Candidate metadata CSV (default: <fasta>_metadata.csv)")
    parser.add_argument("--reference_report", default=None, help="Oracle report for the real windows, used as the comparison baseline")
    parser.add_argument("--top_k", type=int, default=200, help="How many candidates to keep")
    parser.add_argument("--out_fasta", required=True, help="Where to write the selected sequences")
    return parser.parse_args()


def load_scores(report_path: str) -> Dict[str, float]:
    """Read per-sequence MSSI scores from an oracle report.

    Args:
        report_path: Path to the JSON report.

    Returns:
        Mapping of sequence id to MSSI score.

    Raises:
        FileNotFoundError: If the report does not exist.
        ValueError: If the report contains no scored sequences.
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Oracle report not found: {report_path}. Run evaluate.py --oracle enformer first.")

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    scores = {seq_id: float(res["mssi_score"]) for seq_id, res in report.get("sequences", {}).items()}
    if not scores:
        raise ValueError(f"{report_path} contains no scored sequences.")
    return scores


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/select_candidates.log")

    scores = load_scores(args.report)
    records = {rec.id: rec for rec in SeqIO.parse(args.fasta, "fasta")}

    unscored = set(records) - set(scores)
    if unscored:
        raise ValueError(
            f"{len(unscored)} sequences in {args.fasta} have no score in {args.report} "
            f"(first: {sorted(unscored)[0]}). Score the same FASTA that is being selected from."
        )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    kept = ranked[:args.top_k]

    os.makedirs(os.path.dirname(args.out_fasta) or ".", exist_ok=True)
    SeqIO.write([records[seq_id] for seq_id, _ in kept], args.out_fasta, "fasta")

    metadata_path = args.metadata or os.path.splitext(args.fasta)[0] + "_metadata.csv"
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        selected_ids = [seq_id for seq_id, _ in kept]
        out_meta = meta[meta["peak_id"].isin(selected_ids)].copy()
        out_meta["mssi_score"] = out_meta["peak_id"].map(dict(kept))
        out_meta = out_meta.sort_values("mssi_score", ascending=False)
        out_meta.to_csv(os.path.splitext(args.out_fasta)[0] + "_metadata.csv", index=False)

    selected_scores = [score for _, score in kept]
    logger.info(
        "Selected %d of %d candidates. MSSI: best %.4f, worst kept %.4f, candidate pool mean %.4f.",
        len(kept), len(ranked), selected_scores[0], selected_scores[-1],
        sum(scores.values()) / len(scores),
    )

    if args.reference_report:
        reference = load_scores(args.reference_report)
        reference_mean = sum(reference.values()) / len(reference)
        above = sum(1 for score in selected_scores if score > reference_mean)
        logger.info(
            "Real windows (n=%d) mean MSSI %.4f; %d of %d selected candidates score above it.",
            len(reference), reference_mean, above, len(selected_scores),
        )
    else:
        logger.warning(
            "No --reference_report given. The MSSI values above are uncalibrated: without the "
            "real windows scored the same way, a high score is not evidence of anything."
        )


if __name__ == "__main__":
    main()
