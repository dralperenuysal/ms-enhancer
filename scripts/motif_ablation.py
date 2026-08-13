"""Test whether a specific factor's sites carry the oracle's preference.

The selected/rejected contrast is correlational: selected candidates happen to
carry more sites for some factor. This turns it into an intervention. Every hit
for the named factor is shuffled in place, so the sequence keeps its length,
composition and every other motif's position while that factor's sites are
destroyed.

The control matters more than the ablation: an equal number of equally long
windows are shuffled at positions that carry no hit for the factor. If the
factor-targeted ablation and the position-matched control cost the same MSSI,
the effect is "some DNA was scrambled", not "these sites were removed".

With ``--tf ANY`` and several ``--doses``, the same machinery becomes a
dose-response curve on motif density: sites are drawn at random across all
factors, so the only thing varying is how many are destroyed. A density
hypothesis predicts MSSI falling monotonically with dose while the
position-matched controls stay flat.

Writes a FASTA plus metadata; scoring is done by ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.evaluation.motif_analyzer import MotifAnalyzer
from src.utils.helpers import setup_logging
from scripts.occlusion_scan import load_ranked_sequences

logger = logging.getLogger(__name__)


def shuffle_spans(sequence: str, spans: List[Tuple[int, int]], seed: int) -> str:
    """Return ``sequence`` with the bases inside each span permuted in place.

    Shuffling rather than replacing keeps the local base composition identical,
    so a change in score cannot be attributed to a shift in GC content.

    Args:
        sequence: Sequence to modify.
        spans: (start, end) half-open intervals to shuffle.
        seed: Seed for the permutation.

    Returns:
        The modified sequence, the same length as the input.
    """
    rng = np.random.default_rng(seed)
    bases = list(sequence)
    for start, end in spans:
        window = bases[start:end]
        rng.shuffle(window)
        bases[start:end] = window
    return "".join(bases)


def matched_control_spans(hit_spans: List[Tuple[int, int]], length: int, seed: int) -> List[Tuple[int, int]]:
    """Pick spans of the same widths at positions that overlap no hit.

    Args:
        hit_spans: The spans being ablated, whose widths are matched.
        length: Sequence length in bp.
        seed: Seed for the placement draw.

    Returns:
        The same number of spans, of the same widths, placed off the hits.
        Fewer spans are returned if the sequence has no room left.
    """
    rng = np.random.default_rng(seed)
    occupied = np.zeros(length, dtype=bool)
    for start, end in hit_spans:
        occupied[start:end] = True

    control: List[Tuple[int, int]] = []
    for start, end in hit_spans:
        width = end - start
        for _ in range(100):  # ponytail: rejection sampling; fine at these densities.
            candidate = int(rng.integers(0, max(1, length - width)))
            if not occupied[candidate:candidate + width].any():
                occupied[candidate:candidate + width] = True
                control.append((candidate, candidate + width))
                break
    return control


def draw_dose(spans: List[Tuple[int, int]], dose: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Draw ``dose`` spans without replacement, or all of them when dose is 0.

    Args:
        spans: Available hit spans.
        dose: How many to draw; 0 means every span.
        rng: Seeded generator, so a dose series is reproducible.

    Returns:
        The drawn spans.

    Raises:
        ValueError: If ``dose`` exceeds the number of available spans.
    """
    if not dose:
        return spans
    if dose > len(spans):
        raise ValueError(f"Cannot draw {dose} spans from {len(spans)}.")
    return [spans[i] for i in rng.choice(len(spans), size=dose, replace=False)]


def main() -> None:
    """Write the ablation FASTA and metadata for the top-ranked candidates."""
    parser = argparse.ArgumentParser(description="Motif ablation over oracle-selected candidates")
    parser.add_argument("--score_report", required=True)
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--metadata", required=True, help="Metadata CSV giving the host locus")
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument(
        "--tf", required=True,
        help="Factor whose sites are ablated, or 'ANY' to draw from every configured factor",
    )
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_controls", type=int, default=3, help="Position-matched control draws")
    parser.add_argument(
        "--doses", nargs="*", type=int, default=[],
        help="Ablate this many randomly chosen sites, one variant per value; "
             "default ablates every site of the factor",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    analyzer = MotifAnalyzer(config_path=args.config)
    if args.tf != "ANY" and args.tf not in analyzer.tf_names:
        raise ValueError(
            f"'{args.tf}' is not configured; available: ANY, {', '.join(analyzer.tf_names)}."
        )
    if any(dose < 1 for dose in args.doses):
        raise ValueError(f"--doses must all be >= 1, got {args.doses}.")

    ranked = load_ranked_sequences(Path(args.score_report), Path(args.input_fasta))[: args.top_k]
    host = pd.read_csv(args.metadata).iloc[0]

    records: List[Tuple[str, str]] = []
    for rank, (seq_id, sequence, _) in enumerate(ranked):
        hits = analyzer.scan_sequence(sequence)
        spans = [(h["start"], h["end"]) for h in hits if args.tf in ("ANY", h["tf"])]
        if not spans:
            logger.warning("Candidate %s has no %s site; skipping.", seq_id, args.tf)
            continue
        records.append((f"a{rank:02d}_intact", sequence))

        rng = np.random.default_rng(args.seed + rank)
        # A dose of 0 means "every site", which is the single-dose default.
        for dose in args.doses or [0]:
            if dose and dose > len(spans):
                logger.warning(
                    "Candidate %s has %d sites, fewer than dose %d; skipping that dose.",
                    seq_id, len(spans), dose,
                )
                continue
            drawn = draw_dose(spans, dose, rng)
            tag = "all" if not dose else f"{dose:03d}"
            records.append((f"a{rank:02d}_d{tag}_abl", shuffle_spans(sequence, drawn, args.seed + rank)))
            for draw in range(args.n_controls):
                control = matched_control_spans(drawn, len(sequence), args.seed + 1000 * draw + rank)
                records.append(
                    (f"a{rank:02d}_d{tag}_ctl{draw}", shuffle_spans(sequence, control, args.seed + rank))
                )
        logger.info("Candidate %s: %d %s sites available.", seq_id, len(spans), args.tf)

    if not records:
        raise ValueError(f"No candidate carried a {args.tf} site; nothing to ablate.")

    Path(args.output_fasta).write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    pd.DataFrame(
        {
            "peak_id": [name for name, _ in records],
            "chrom": host["chrom"],
            "start": host["start"],
            "end": host["end"],
            "cell_type": host["cell_type"],
        }
    ).to_csv(args.output_metadata, index=False)
    logger.info("Wrote %d sequences to %s.", len(records), args.output_fasta)


if __name__ == "__main__":
    main()
