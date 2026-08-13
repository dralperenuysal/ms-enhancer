"""Read Enformer's own input attributions over a designed insert.

The motif analyses ask what *we* can name in the selected sequences. This asks
the oracle directly: differentiate MSSI with respect to the one-hot input and
take gradient x input, which gives a per-base contribution to the very quantity
selection optimised.

Only the 1000 bp insert is reported. The surrounding ~196 kb of real genome
dominates the absolute prediction (see the variance decomposition in the report),
but it is identical across candidates, so it cannot explain why one candidate
outranks another.

Attribution is not proof of mechanism: it says where the model's sensitivity
lies, not that the sequence there does anything in a cell. It is used here to
generate a hypothesis the motif scans failed to produce.

Usage:
    python scripts/attribution_scan.py --score_report logs/sel0.json \
        --input_fasta data/fasta/sel0.fasta --metadata data/fasta/sel0_metadata.csv \
        --reference_fasta data/hg38.fa --top_k 10 --bottom_k 10 \
        --output logs/attribution.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluation.enformer_oracle import EnformerOracle
from src.utils.helpers import setup_logging
from scripts.occlusion_scan import load_ranked_sequences

logger = logging.getLogger(__name__)


def attribute(oracle: EnformerOracle, full_seq: str, target_tracks: List[int],
              insert_start: int, insert_length: int) -> np.ndarray:
    """Return per-base gradient x input attribution over the insert.

    The scalar differentiated is MSSI itself — mean target track signal minus
    mean background track signal — so the attribution answers the question
    selection was optimising, not some proxy of it.

    Args:
        oracle: A loaded Enformer oracle.
        full_seq: The full context window.
        target_tracks: Track indices for the cell type being scored.
        insert_start: Offset of the insert within the context window.
        insert_length: Length of the insert in bp.

    Returns:
        Array of length ``insert_length`` holding the attribution of each base.
    """
    oracle._load_model()
    one_hot = oracle._one_hot_encode(full_seq).unsqueeze(0).to(oracle.device)
    one_hot.requires_grad_(True)

    preds = oracle.model(one_hot)
    human = preds["human"] if isinstance(preds, dict) else preds
    track_means = human.mean(dim=1).squeeze(0)
    mssi = track_means[target_tracks].mean() - track_means[oracle.background_tracks].mean()

    mssi.backward()
    if one_hot.grad is None:
        raise RuntimeError("Enformer returned no input gradient; the graph was not retained.")

    # gradient x input, summed over channels: at a one-hot position this keeps
    # only the gradient of the base that is actually present.
    contribution = (one_hot.grad * one_hot).sum(dim=-1).squeeze(0)
    return contribution[insert_start:insert_start + insert_length].detach().float().cpu().numpy()


def main() -> None:
    """Attribute MSSI over the inserts of top- and bottom-ranked candidates."""
    parser = argparse.ArgumentParser(description="Enformer input attribution over designed inserts")
    parser.add_argument("--score_report", required=True)
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reference_fasta", required=True)
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bottom_k", type=int, default=10)
    parser.add_argument("--output", required=True, help="Where to write the per-base attributions")
    args = parser.parse_args()

    setup_logging()

    ranked = load_ranked_sequences(Path(args.score_report), Path(args.input_fasta))
    if args.top_k + args.bottom_k > len(ranked):
        raise ValueError(
            f"Asked for {args.top_k} + {args.bottom_k} candidates but only {len(ranked)} were scored."
        )
    chosen: List[Tuple[str, str, float, str]] = (
        [(seq_id, seq, score, "selected") for seq_id, seq, score in ranked[: args.top_k]]
        + [(seq_id, seq, score, "rejected") for seq_id, seq, score in ranked[-args.bottom_k:]]
    )

    host = pd.read_csv(args.metadata).iloc[0]
    centre = int((host["start"] + host["end"]) // 2)
    oracle = EnformerOracle(config_path=args.config)
    target_tracks = oracle.target_tracks_by_cell_type[host["cell_type"]]

    attributions: Dict[str, Dict[str, object]] = {}
    for index, (seq_id, sequence, score, group) in enumerate(chosen, start=1):
        full_seq = oracle.construct_context_window(
            sequence,
            reference_fasta_path=args.reference_fasta,
            chrom=host["chrom"],
            center_pos=centre,
        )
        insert_start = (len(full_seq) - len(sequence)) // 2
        values = attribute(oracle, full_seq, target_tracks, insert_start, len(sequence))
        attributions[seq_id] = {
            "group": group,
            "mssi_score": score,
            "attribution": [float(v) for v in values],
        }
        logger.info(
            "[%d/%d] %s (%s): mean |attr| %.3e, max %.3e at %d bp.",
            index, len(chosen), seq_id, group,
            float(np.abs(values).mean()), float(np.abs(values).max()), int(np.argmax(np.abs(values))),
        )

    Path(args.output).write_text(json.dumps({
        "locus": f"{host['chrom']}:{host['start']}-{host['end']}",
        "cell_type": host["cell_type"],
        "sequences": attributions,
    }))
    logger.info("Wrote attributions for %d sequences to %s.", len(attributions), args.output)


if __name__ == "__main__":
    main()
