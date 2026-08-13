"""Build an occlusion scan for oracle-selected candidates.

Selection told us *that* the oracle prefers some inserts over others; it did
not tell us *where* in the 1 kb insert that preference lives. This script
replaces successive fixed-width tiles with random DNA so the rescored drop in
MSSI localises the signal. Tiles are the only thing that changes: locus,
flanks and cell type are held fixed, so the scored differences are attributable
to the occluded window alone.

Writes a FASTA plus the matching metadata CSV; scoring is done by
``evaluate.py --oracle enformer`` on the produced files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def load_ranked_sequences(report_path: Path, fasta_path: Path) -> List[Tuple[str, str, float]]:
    """Return (seq_id, sequence, mssi) ordered by descending MSSI.

    Args:
        report_path: JSON report written by ``evaluate.py --oracle enformer``.
        fasta_path: FASTA holding the scored sequences.

    Returns:
        Ranked list of scored sequences.

    Raises:
        FileNotFoundError: If either input is missing.
        ValueError: If the report holds no scored sequence, or a scored
            sequence is absent from the FASTA.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Score report not found: {report_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    scores = json.loads(report_path.read_text()).get("sequences", {})
    if not scores:
        raise ValueError(f"No scored sequences in {report_path}; nothing to occlude.")

    sequences = read_fasta(fasta_path)
    missing = sorted(set(scores) - set(sequences))
    if missing:
        raise ValueError(
            f"{len(missing)} scored sequences are absent from {fasta_path} "
            f"(first: {missing[0]}); the report and FASTA do not match."
        )

    ranked = [(seq_id, sequences[seq_id], float(entry["mssi_score"])) for seq_id, entry in scores.items()]
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def read_fasta(path: Path) -> Dict[str, str]:
    """Read a FASTA into an id -> sequence mapping."""
    sequences: Dict[str, str] = {}
    seq_id = None
    chunks: List[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if seq_id is not None:
                sequences[seq_id] = "".join(chunks)
            seq_id = line[1:].strip().split()[0]
            chunks = []
        elif seq_id is not None:
            chunks.append(line.strip())
    if seq_id is not None:
        sequences[seq_id] = "".join(chunks)
    return sequences


def occlude(sequence: str, tile_size: int, seed: int) -> List[Tuple[int, str]]:
    """Return (tile_index, occluded_sequence) for every tile of ``sequence``.

    Each tile is replaced with random DNA drawn at uniform base composition, so
    an occluded tile carries no motif content while keeping sequence length and
    hence the genomic context identical.

    Args:
        sequence: Insert to occlude.
        tile_size: Width in bases of each occluded window.
        seed: Seed for the replacement draw; the same seed reproduces the scan.

    Returns:
        One entry per tile, in left-to-right order.

    Raises:
        ValueError: If ``tile_size`` is not a positive number of bases.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}.")

    rng = np.random.default_rng(seed)
    variants: List[Tuple[int, str]] = []
    for index, start in enumerate(range(0, len(sequence), tile_size)):
        end = min(start + tile_size, len(sequence))
        filler = "".join(rng.choice(list("ACGT"), size=end - start))
        variants.append((index, sequence[:start] + filler + sequence[end:]))
    return variants


def main() -> None:
    """Write the occlusion FASTA and metadata for the top-ranked candidates."""
    parser = argparse.ArgumentParser(description="Occlusion scan over oracle-selected candidates")
    parser.add_argument("--score_report", required=True, help="Enformer score report for the selection set")
    parser.add_argument("--input_fasta", required=True, help="FASTA the report scored")
    parser.add_argument("--metadata", required=True, help="Metadata CSV for the selection set (host locus)")
    parser.add_argument("--output_fasta", required=True, help="Where to write the occluded sequences")
    parser.add_argument("--output_metadata", required=True, help="Where to write the matching metadata CSV")
    parser.add_argument("--top_k", type=int, default=10, help="How many top-ranked candidates to scan")
    parser.add_argument("--tile_size", type=int, default=100, help="Occluded window width in bases")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random replacement DNA")
    args = parser.parse_args()

    setup_logging()

    ranked = load_ranked_sequences(Path(args.score_report), Path(args.input_fasta))
    if args.top_k < 1:
        raise ValueError(f"--top_k must be >= 1, got {args.top_k}.")
    selected = ranked[: args.top_k]
    logger.info(
        "Scanning %d of %d scored candidates (MSSI %.4f down to %.4f).",
        len(selected), len(ranked), selected[0][2], selected[-1][2],
    )

    host = pd.read_csv(args.metadata).iloc[0]
    records: List[Tuple[str, str]] = []
    for rank, (seq_id, sequence, _) in enumerate(selected):
        records.append((f"c{rank:02d}_{seq_id}_intact", sequence))
        for tile_index, variant in occlude(sequence, args.tile_size, args.seed + rank):
            records.append((f"c{rank:02d}_{seq_id}_tile{tile_index:02d}", variant))

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

    logger.info(
        "Wrote %d sequences (%d candidates x %d variants) to %s at locus %s:%s-%s.",
        len(records), len(selected), len(records) // len(selected),
        args.output_fasta, host["chrom"], host["start"], host["end"],
    )


if __name__ == "__main__":
    main()
