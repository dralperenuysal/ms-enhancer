"""Build GC-matched non-peak inserts and place them at many host loci.

Reviewer 1's central objection to the host-dominance result is that it may be an
artefact of the candidate pool: real inserts and Markov inserts are both drawn
from ATAC-seq peaks, so the range of regulatory activity on offer is narrow, and
GC-matched random sequence is not an experimentally established negative. The
test they ask for is a genuine negative class — genomic sequence that is *not*
accessible in the target cell type — matched on GC so the contrast is not
composition, placed at several host loci so that "does the oracle separate
peak from non-peak" can be asked separately at each host.

If the oracle separates the two classes consistently at every host, then
host-dependent variation in the score does not prevent it from making the
distinction that matters, and the paper's limitation claim must be narrowed. If
the separation itself flips between hosts, the claim stands on stronger ground
than it did. Either answer is worth having; this script produces the sequences
that decide it.

A non-peak window here means: overlapping no called peak from any sample in any
of the three cell types, with a margin, and carrying no N. Excluding peaks from
*every* cell type (not just the target) is deliberate — a window accessible in B
cells is not a clean negative for "not regulatory", only for "not CD4-specific",
and the weaker definition would make the negative class easier to separate for
the wrong reason.

Usage:
    python scripts/nonpeak_negatives.py \
        --reference data/hg38.fa --peaks_dir data/raw/peaks \
        --windows_fasta data/fasta/ms_windows_1000bp.fasta \
        --windows_meta data/fasta/ms_windows_metadata.csv \
        --cell_type CD4_T_cell --n_inserts 40 --n_loci 12 \
        --output_fasta data/fasta/nonpeak.fasta \
        --output_metadata data/fasta/nonpeak_metadata.csv
"""

from __future__ import annotations

import argparse
import gzip
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INSERT_LEN = 1000
#: Keep this far from any called peak, so a "non-peak" window is not merely a
#: peak whose summit sat a few bases outside the interval.
PEAK_MARGIN = 5000


def load_peak_intervals(peaks_dir: Path) -> Dict[str, np.ndarray]:
    """Read every narrowPeak/HOMER file into per-chromosome sorted intervals.

    Args:
        peaks_dir: Directory of per-GSE subdirectories of peak files.

    Returns:
        ``{chrom: array of shape (n, 2)}``, start-sorted, margin already applied.
    """
    spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    files = sorted(peaks_dir.rglob("*"))
    n_files = 0
    for path in files:
        if path.is_dir():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        n_files += 1
        with opener(path, "rt") as handle:
            for line in handle:
                if not line or line.startswith(("#", "track", "chr\t")):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start, end = int(parts[1]), int(parts[2])
                except ValueError:
                    continue  # HOMER files carry a header block with text columns.
                spans[chrom].append((start - PEAK_MARGIN, end + PEAK_MARGIN))

    merged = {}
    for chrom, items in spans.items():
        arr = np.array(sorted(items), dtype=np.int64)
        merged[chrom] = _merge(arr)
    logger.info(
        "Read %d peak files: %d chromosomes, %d merged excluded intervals.",
        n_files, len(merged), sum(len(v) for v in merged.values()),
    )
    return merged


def _merge(intervals: np.ndarray) -> np.ndarray:
    """Merge overlapping sorted intervals, so exclusion is one binary search."""
    if len(intervals) == 0:
        return intervals
    out = [intervals[0].tolist()]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return np.array(out, dtype=np.int64)


def overlaps_peak(chrom: str, start: int, end: int, peaks: Dict[str, np.ndarray]) -> bool:
    """True if ``[start, end)`` touches any excluded interval on ``chrom``."""
    arr = peaks.get(chrom)
    if arr is None or len(arr) == 0:
        return False
    idx = np.searchsorted(arr[:, 0], end)
    # Only the interval starting before `end` can overlap; check it and its
    # predecessor, since merging guarantees the rest start later.
    for j in (idx - 1, idx):
        if 0 <= j < len(arr) and arr[j, 0] < end and start < arr[j, 1]:
            return True
    return False


def gc_fraction(seq: str) -> float:
    """G+C as a fraction of non-N bases; 0.0 for an all-N sequence."""
    upper = seq.upper()
    usable = len(upper) - upper.count("N")
    if usable == 0:
        return 0.0
    return (upper.count("G") + upper.count("C")) / usable


def read_fasta(path: Path) -> Dict[str, str]:
    """Read a FASTA into {id: sequence}."""
    seqs: Dict[str, str] = {}
    name, chunks = None, []
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


def sample_nonpeak(
    fasta,
    peaks: Dict[str, np.ndarray],
    targets: List[float],
    rng: np.random.Generator,
    tolerance: float = 0.02,
    max_tries: int = 400000,
) -> List[Tuple[str, int, str]]:
    """Draw one non-peak window per target GC value.

    Rejection sampling against a GC target rather than drawing freely: an
    unmatched negative class would differ from the positives in composition, and
    the oracle's score tracks GC, so any separation found would be uninformative.

    Args:
        fasta: Indexed reference (``pyfaidx.Fasta``).
        peaks: Excluded intervals from :func:`load_peak_intervals`.
        targets: GC fractions to match, one per requested insert.
        rng: Seeded generator.
        tolerance: Absolute GC difference allowed.
        max_tries: Give up rather than spin forever on an impossible target.

    Returns:
        List of ``(chrom, start, sequence)``, one per target, in target order.

    Raises:
        RuntimeError: If a target cannot be matched within ``max_tries``.
    """
    usable = [c for c in fasta.keys() if "_" not in c and c not in ("chrM", "chrEBV")]
    lengths = {c: len(fasta[c]) for c in usable}
    weights = np.array([lengths[c] for c in usable], dtype=float)
    weights /= weights.sum()

    remaining = sorted(range(len(targets)), key=lambda i: targets[i])
    found: Dict[int, Tuple[str, int, str]] = {}
    taken: List[Tuple[str, int]] = []
    tries = 0

    while remaining and tries < max_tries:
        tries += 1
        chrom = usable[rng.choice(len(usable), p=weights)]
        start = int(rng.integers(0, lengths[chrom] - INSERT_LEN))
        end = start + INSERT_LEN
        if overlaps_peak(chrom, start, end, peaks):
            continue
        if any(c == chrom and abs(s - start) < INSERT_LEN for c, s in taken):
            continue
        seq = str(fasta[chrom][start:end]).upper()
        if "N" in seq:
            continue
        gc = gc_fraction(seq)
        for i in list(remaining):
            if abs(gc - targets[i]) <= tolerance:
                found[i] = (chrom, start, seq)
                taken.append((chrom, start))
                remaining.remove(i)
                break

    if remaining:
        raise RuntimeError(
            f"Could not GC-match {len(remaining)} of {len(targets)} inserts in {tries} draws "
            f"(unmatched targets: {[round(targets[i], 3) for i in remaining[:5]]}). "
            f"Widen --tolerance or lower --n_inserts."
        )
    logger.info("Matched %d non-peak windows in %d draws.", len(found), tries)
    return [found[i] for i in range(len(targets))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--peaks_dir", required=True)
    parser.add_argument("--windows_fasta", required=True)
    parser.add_argument("--windows_meta", required=True)
    parser.add_argument("--cell_type", default="CD4_T_cell")
    parser.add_argument("--n_inserts", type=int, default=40,
                        help="Peak-derived inserts, each matched by one non-peak insert")
    parser.add_argument("--n_loci", type=int, default=12, help="Host loci to place both classes at")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import pyfaidx

    rng = np.random.default_rng(args.seed)
    fasta = pyfaidx.Fasta(args.reference)

    meta = pd.read_csv(args.windows_meta)
    meta = meta[meta["cell_type"] == args.cell_type].reset_index(drop=True)
    if len(meta) < args.n_inserts + args.n_loci:
        raise ValueError(f"Only {len(meta)} {args.cell_type} windows available.")

    seqs = read_fasta(Path(args.windows_fasta))

    # Hosts are spread across the peak-score range, matching how the 24-locus
    # survey chose its own panel, so the two are comparable.
    ranked = meta.sort_values("peak_score").reset_index(drop=True)
    host_idx = np.linspace(0, len(ranked) - 1, args.n_loci).astype(int)
    hosts = ranked.iloc[host_idx]

    # Positives are drawn from windows not used as hosts.
    host_ids = set(hosts["peak_id"])
    pool = meta[~meta["peak_id"].isin(host_ids)]
    pos = pool.iloc[rng.choice(len(pool), size=args.n_inserts, replace=False)]
    pos_seqs = [seqs[pid] for pid in pos["peak_id"]]
    targets = [gc_fraction(s) for s in pos_seqs]

    peaks = load_peak_intervals(Path(args.peaks_dir))
    negatives = sample_nonpeak(fasta, peaks, targets, rng, tolerance=args.tolerance)

    records, rows = [], []
    for li, host in enumerate(hosts.itertuples(index=False)):
        centre = (host.start + host.end) // 2
        for ii, (seq, (chrom, start, neg_seq)) in enumerate(zip(pos_seqs, negatives)):
            for cls, s in (("peak", seq), ("nonpeak", neg_seq)):
                sid = f"L{li:02d}_i{ii:02d}_{cls}"
                records.append((sid, s))
                rows.append({
                    "peak_id": sid, "chrom": host.chrom,
                    "start": centre - INSERT_LEN // 2, "end": centre + INSERT_LEN // 2,
                    "cell_type": args.cell_type, "locus": f"L{li:02d}",
                    "insert": f"i{ii:02d}", "insert_class": cls,
                    "host_peak_id": host.peak_id, "host_peak_score": host.peak_score,
                    "source_chrom": chrom if cls == "nonpeak" else "",
                    "source_start": start if cls == "nonpeak" else "",
                    "gc": round(gc_fraction(s), 4),
                })

    Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_fasta, "w") as handle:
        for sid, seq in records:
            handle.write(f">{sid}\n{seq}\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_metadata, index=False)

    gc_pos = frame[frame.insert_class == "peak"]["gc"]
    gc_neg = frame[frame.insert_class == "nonpeak"]["gc"]
    logger.info(
        "Wrote %d sequences: %d loci x %d inserts x 2 classes. "
        "GC peak %.4f vs non-peak %.4f (difference %.4f).",
        len(records), args.n_loci, args.n_inserts,
        gc_pos.mean(), gc_neg.mean(), gc_pos.mean() - gc_neg.mean(),
    )


def demo() -> None:
    """Self-check on the interval and GC logic, which is where errors hide."""
    merged = _merge(np.array([[10, 20], [15, 30], [100, 110]], dtype=np.int64))
    assert merged.tolist() == [[10, 30], [100, 110]], merged

    peaks = {"chr1": merged}
    assert overlaps_peak("chr1", 25, 35, peaks)       # straddles the first
    assert overlaps_peak("chr1", 12, 14, peaks)       # inside the first
    assert not overlaps_peak("chr1", 40, 90, peaks)   # in the gap
    assert not overlaps_peak("chr1", 30, 40, peaks)   # half-open: 30 is the end
    assert overlaps_peak("chr1", 105, 120, peaks)     # straddles the last
    assert not overlaps_peak("chr2", 0, 1000, peaks)  # chromosome with no peaks

    assert gc_fraction("GCGC") == 1.0
    assert gc_fraction("ATAT") == 0.0
    assert gc_fraction("GCATNN") == 0.5   # N excluded from the denominator
    assert gc_fraction("NNNN") == 0.0

    # A target that cannot be met must raise rather than return a mismatched set.
    class OneChrom:
        def __init__(self): self._s = "AT" * 1000
        def keys(self): return ["chr1"]
        def __getitem__(self, k): return self
        def __len__(self): return len(self._s)
        def __getitem__(self, k):  # noqa: F811 - slicing returns the string
            return self._s if isinstance(k, str) else self._s[k]

    try:
        sample_nonpeak(OneChrom(), {}, [0.9], np.random.default_rng(0), max_tries=500)
    except RuntimeError:
        pass
    else:
        raise AssertionError("impossible GC target should raise")

    print("demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
