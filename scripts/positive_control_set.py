"""Build the positive control for the non-peak separation test.

The non-peak experiment returned a null: the oracle does not separate
peak-derived inserts from GC-matched genomic sequence outside peaks. A null is
only worth reporting if the same setup can detect a difference it should detect,
so this builds the comparison the manuscript already reports a signal for ---
peak-derived inserts against GC-matched *synthetic* random sequence --- at the
identical host loci, with the identical inserts, scored the identical way.

If the oracle separates peak from synthetic random here but not from real
non-peak genomic DNA, the null is informative: it says the discriminative
ability does not survive a realistic negative. If it separates neither, the
setup lacks power and we must say so instead.

The random sequences use the same composition-only null as the rest of the
project (``scripts/compare_motif_grammar.gc_matched_random``): i.i.d. bases
drawn at the reference set's GC fraction.

Usage:
    python scripts/positive_control_set.py \
        --nonpeak_fasta data/fasta/nonpeak.fasta \
        --nonpeak_meta data/fasta/nonpeak_metadata.csv \
        --output_fasta data/fasta/poscontrol.fasta \
        --output_metadata data/fasta/poscontrol_metadata.csv
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_fasta(path: Path) -> dict:
    """Read a FASTA into {id: sequence}."""
    seqs, name, chunks = {}, None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line.upper())
    if name:
        seqs[name] = "".join(chunks)
    return seqs


def gc_matched_random(reference: list, seed: int) -> list:
    """Random DNA at the reference set's GC fraction: the composition-only null.

    Kept identical to ``scripts/compare_motif_grammar.gc_matched_random`` so the
    control means the same thing here as everywhere else in the project.
    """
    joined = "".join(reference)
    gc = (joined.count("G") + joined.count("C")) / len(joined)
    weights = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", weights=weights, k=len(reference[0]))) for _ in reference]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonpeak_fasta", required=True)
    parser.add_argument("--nonpeak_meta", required=True)
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    seqs = read_fasta(Path(args.nonpeak_fasta))
    meta = pd.read_csv(args.nonpeak_meta)

    # One row per (locus, insert) carrying the peak-derived sequence. The host
    # loci and insert identities are inherited rather than re-chosen, so the
    # control differs from the non-peak test in exactly one respect: what the
    # peak inserts are being compared against.
    peaks = meta[meta["insert_class"] == "peak"].copy()
    inserts = sorted(peaks["insert"].unique())
    # peaks["insert"], never peaks.insert: the latter is DataFrame.insert, the
    # method, and comparing it to a string silently yields False.
    per_insert = {
        i: seqs[f'{peaks.loc[peaks["insert"] == i, "locus"].iloc[0]}_{i}_peak']
        for i in inserts
    }

    randoms = dict(zip(inserts, gc_matched_random([per_insert[i] for i in inserts], args.seed)))

    records, rows = [], []
    for row in peaks.itertuples(index=False):
        for cls, seq in (("peak", per_insert[row.insert]), ("random", randoms[row.insert])):
            sid = f"{row.locus}_{row.insert}_{cls}"
            records.append((sid, seq))
            rows.append({
                "peak_id": sid, "chrom": row.chrom, "start": row.start, "end": row.end,
                "cell_type": row.cell_type, "locus": row.locus, "insert": row.insert,
                "insert_class": cls, "host_peak_id": row.host_peak_id,
                "gc": round((seq.count("G") + seq.count("C")) / len(seq), 4),
            })

    Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_fasta, "w") as handle:
        for sid, seq in records:
            handle.write(f">{sid}\n{seq}\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_metadata, index=False)

    gc_p = frame[frame.insert_class == "peak"]["gc"].mean()
    gc_r = frame[frame.insert_class == "random"]["gc"].mean()
    logger.info(
        "Wrote %d sequences: %d loci x %d inserts x 2 classes. GC peak %.4f vs random %.4f (difference %+.4f).",
        len(records), peaks["locus"].nunique(), len(inserts), gc_p, gc_r, gc_p - gc_r,
    )


def demo() -> None:
    """Self-check: the random set must match GC and must not copy the input."""
    ref = ["ACGT" * 250, "GGCC" * 250]
    out = gc_matched_random(ref, seed=0)
    assert len(out) == 2 and all(len(s) == 1000 for s in out)
    assert out[0] not in ref, "random sequence must not be a copy of the reference"

    def gc(s):
        return (s.count("G") + s.count("C")) / len(s)

    target = gc("".join(ref))
    observed = gc("".join(out))
    # i.i.d. draws, so 2,000 bases give a standard error near 0.011.
    assert abs(observed - target) < 0.04, (observed, target)

    # The same seed must reproduce, or a rerun would silently change the control.
    assert gc_matched_random(ref, seed=0) == out
    assert gc_matched_random(ref, seed=1) != out
    print("demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
