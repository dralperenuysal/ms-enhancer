"""Peak processing for MS-ENHANCER-GEN.

Turns per-sample ATAC-seq peak calls into a clean set of uniform 1000 bp genomic
windows that (a) lie inside an MS GWAS risk locus and (b) carry the cell-type and
signal metadata the conditional generator is trained on.

Interval intersection is implemented directly on sorted NumPy arrays rather than
via ``pybedtools``: the container has no ``bedtools`` binary, and the operation
needed here (is this peak summit inside any risk locus?) is a one-dimensional
containment test that is exactly reproducible without an external dependency.
"""

from __future__ import annotations

import bisect
import gzip
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Column layouts of the supported peak-call formats.
NARROWPEAK_COLUMNS = [
    "chrom", "start", "end", "name", "score", "strand",
    "signal_value", "p_value", "q_value", "summit_offset",
]


class BEDProcessor:
    """Builds MS-relevant 1000 bp regulatory windows from peak calls."""

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """Initialize from the data configuration file.

        Args:
            config_path: Path to ``data_config.yaml``.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle)

        ref = self.config.get("reference_genome", {})
        self.primary_chromosomes: List[str] = ref.get(
            "primary_chromosomes", [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
        )
        self.genome_version: str = ref.get("version", "hg38")
        self.reference_fasta: str = ref.get("local_fasta", "data/hg38.fa")

        seq = self.config.get("sequence_encoding", {})
        self.target_window_len: int = seq.get("sequence_length", 1000)
        self.max_n_fraction: float = float(seq.get("max_n_fraction", 0.05))
        self.min_window_separation: int = int(
            seq.get("min_window_separation_bp", self.target_window_len // 2)
        )
        self.cell_type_exclusive: bool = bool(seq.get("cell_type_exclusive_windows", True))

        paths = self.config.get("paths", {})
        self.bed_dir: str = paths.get("bed_dir", "data/bed")
        self.fasta_dir: str = paths.get("fasta_dir", "data/fasta")
        self.raw_dir: str = paths.get("raw_dir", "data/raw")
        os.makedirs(self.bed_dir, exist_ok=True)
        os.makedirs(self.fasta_dir, exist_ok=True)

        self.gwas_bed: str = self.config.get("gwas", {}).get("output_bed", "data/bed/ms_gwas_loci_hg38.bed")

        logger.info(
            "BEDProcessor initialised (genome=%s, window=%d bp, max_N=%.2f).",
            self.genome_version,
            self.target_window_len,
            self.max_n_fraction,
        )

    # -------------------------------------------------------------- peak loading

    def load_peak_file(self, path: str, peak_format: str = "narrowPeak") -> pd.DataFrame:
        """Read one peak-call file into a normalized DataFrame.

        Args:
            path: Path to a (optionally gzipped) peak file.
            peak_format: ``"narrowPeak"`` (MACS) or ``"homer"`` (HOMER findPeaks).

        Returns:
            DataFrame with columns ``chrom``, ``start``, ``end``, ``peak_score``,
            ``fold_enrichment`` and ``summit_offset`` (NaN when unavailable).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the format is unknown or the file yields no peaks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Peak file not found: {path}")

        opener = gzip.open if path.endswith(".gz") else open

        if peak_format == "narrowPeak":
            with opener(path, "rt") as handle:  # type: ignore[operator]
                frame = pd.read_csv(handle, sep="\t", header=None, comment="#")
            frame = frame.iloc[:, : len(NARROWPEAK_COLUMNS)]
            frame.columns = NARROWPEAK_COLUMNS[: frame.shape[1]]
            frame = frame.rename(columns={"score": "peak_score", "signal_value": "fold_enrichment"})
            if "summit_offset" not in frame.columns:
                frame["summit_offset"] = np.nan

        elif peak_format == "homer":
            with opener(path, "rt") as handle:  # type: ignore[operator]
                frame = pd.read_csv(handle, sep="\t", comment=None, header=0)
            frame.columns = [str(c).strip().lstrip("#") for c in frame.columns]
            rename = {
                "chr": "chrom",
                "findPeaks Score": "peak_score",
                "Fold Change vs Local": "fold_enrichment",
            }
            frame = frame.rename(columns=rename)
            required = {"chrom", "start", "end", "peak_score", "fold_enrichment"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"HOMER peak file {path} is missing columns: {sorted(missing)}")
            # HOMER emits fixed-width peaks with no summit column.
            frame["summit_offset"] = np.nan

        else:
            if "summit_offset" not in frame.columns:
                frame["summit_offset"] = np.nan

        # Fallback for missing fold_enrichment (e.g., simple 5-column BED files)
        if "fold_enrichment" not in frame.columns:
            frame["fold_enrichment"] = 1.0

        frame = frame[["chrom", "start", "end", "peak_score", "fold_enrichment", "summit_offset"]].copy()
        frame["chrom"] = frame["chrom"].astype(str)
        for column in ("start", "end", "peak_score", "fold_enrichment", "summit_offset"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["start", "end", "peak_score", "fold_enrichment"])

        if frame.empty:
            raise ValueError(f"No usable peaks parsed from {path} (format={peak_format}).")
        return frame

    def filter_primary_chromosomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop alt/patch/unplaced contigs.

        Args:
            frame: DataFrame with a ``chrom`` column.

        Returns:
            Filtered copy.

        Raises:
            KeyError: If ``chrom`` is absent.
        """
        if "chrom" not in frame.columns:
            raise KeyError("DataFrame must contain a 'chrom' column.")
        filtered = frame[frame["chrom"].isin(self.primary_chromosomes)].copy()
        removed = len(frame) - len(filtered)
        if removed:
            logger.debug("Filtered out %d non-primary contig entries.", removed)
        return filtered

    # ------------------------------------------------------------- intersection

    @staticmethod
    def load_loci_bed(path: str) -> pd.DataFrame:
        """Read a risk-locus BED file.

        Args:
            path: Path to a 3+ column BED file.

        Returns:
            DataFrame with ``chrom``, ``start``, ``end``.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If the file is empty.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GWAS risk-locus BED not found: {path}. Run MSGWASLoci.build() first."
            )
        frame = pd.read_csv(path, sep="\t", header=None, comment="#")
        if frame.empty:
            raise ValueError(f"GWAS risk-locus BED is empty: {path}")
        frame = frame.iloc[:, :3]
        frame.columns = ["chrom", "start", "end"]
        frame["chrom"] = frame["chrom"].astype(str)
        return frame

    @staticmethod
    def intersect_with_loci(peaks: pd.DataFrame, loci: pd.DataFrame) -> pd.DataFrame:
        """Keep peaks whose summit position falls inside any risk locus.

        Uses a per-chromosome ``searchsorted`` over locus starts, which is exact for
        the merged, non-overlapping intervals produced by :class:`MSGWASLoci`.

        Args:
            peaks: DataFrame with ``chrom`` and a ``summit_pos`` column.
            loci: DataFrame with ``chrom``, ``start``, ``end``.

        Returns:
            The subset of ``peaks`` inside a locus.

        Raises:
            KeyError: If ``summit_pos`` is absent from ``peaks``.
        """
        if "summit_pos" not in peaks.columns:
            raise KeyError("peaks DataFrame must contain a 'summit_pos' column.")

        kept: List[pd.DataFrame] = []
        for chrom, chrom_peaks in peaks.groupby("chrom", sort=False):
            chrom_loci = loci[loci["chrom"] == chrom]
            if chrom_loci.empty:
                continue
            starts = np.sort(chrom_loci["start"].to_numpy())
            order = np.argsort(chrom_loci["start"].to_numpy())
            ends = chrom_loci["end"].to_numpy()[order]

            positions = chrom_peaks["summit_pos"].to_numpy()
            idx = np.searchsorted(starts, positions, side="right") - 1
            inside = (idx >= 0) & (positions < ends[np.clip(idx, 0, len(ends) - 1)])
            if inside.any():
                kept.append(chrom_peaks[inside])

        if not kept:
            return peaks.iloc[0:0].copy()
        return pd.concat(kept, ignore_index=True)

    # ------------------------------------------------------------------ resizing

    def resize_peak_to_window(
        self,
        start: int,
        end: int,
        summit_offset: Optional[int] = None,
        target_len: int = 1000,
    ) -> Tuple[int, int, str]:
        """Centre a uniform window on the peak summit, falling back to the midpoint.

        Args:
            start: 0-based peak start.
            end: 0-based exclusive peak end.
            summit_offset: narrowPeak column 10 (offset from ``start`` to summit).
            target_len: Window width.

        Returns:
            Tuple of ``(new_start, new_end, method)`` where method is
            ``"summit"`` or ``"midpoint"``.
        """
        if summit_offset is not None and 0 <= summit_offset <= (end - start):
            center = start + int(summit_offset)
            method = "summit"
        else:
            center = (start + end) // 2
            method = "midpoint"

        new_start = max(0, center - target_len // 2)
        return new_start, new_start + target_len, method

    @staticmethod
    def collapse_redundant_windows(windows: pd.DataFrame, min_separation: int) -> pd.DataFrame:
        """Drop near-duplicate windows that describe the same regulatory element.

        The same element is called in every donor of a cell type, producing windows
        whose starts differ by only a few tens of bp. Keeping them all inflates the
        dataset with near-identical sequences and lets the same element appear in
        both the training and validation split. Within each ``(cell_type, chrom)``
        group this keeps the highest-scoring window and rejects any later window
        whose start lies within ``min_separation`` bp of an accepted one.

        Args:
            windows: Pooled windows with ``cell_type``, ``chrom``, ``win_start``
                and ``peak_score`` columns.
            min_separation: Minimum distance in bp between the starts of two kept
                windows. Must be positive; values below 1 disable the filter.

        Returns:
            The retained subset of ``windows``, index reset.
        """
        if min_separation < 1 or windows.empty:
            return windows.reset_index(drop=True)

        keep_positions: List[int] = []
        ordered = windows.sort_values("peak_score", ascending=False)
        for _, group in ordered.groupby(["cell_type", "chrom"], sort=False):
            accepted: List[int] = []  # sorted starts of windows kept in this group
            for position, start in zip(group.index, group["win_start"].to_numpy()):
                insert_at = bisect.bisect_left(accepted, start)
                left_ok = insert_at == 0 or start - accepted[insert_at - 1] >= min_separation
                right_ok = insert_at == len(accepted) or accepted[insert_at] - start >= min_separation
                if left_ok and right_ok:
                    accepted.insert(insert_at, int(start))
                    keep_positions.append(position)

        collapsed = windows.loc[keep_positions].reset_index(drop=True)
        logger.info(
            "Collapsed %d redundant windows (>=%d bp separation enforced); %d remain.",
            len(windows) - len(collapsed),
            min_separation,
            len(collapsed),
        )
        return collapsed

    @staticmethod
    def drop_shared_windows(windows: pd.DataFrame, min_separation: int) -> pd.DataFrame:
        """Keep only windows whose locus is called in exactly one cell type.

        Roughly half of the pooled windows are the same regulatory element called
        in two or three cell types. Keeping them puts near-identical sequences in
        the dataset under conflicting condition vectors, so the cell-type label
        carries no learnable information: measured on the pooled windows, no
        lineage-defining transcription factor (SPI1, PAX5, TCF7, ...) separates
        the cell types, while on the exclusive subset SPI1 is enriched 1.9x in
        microglia over CD4+ T cells. Dropping shared windows is what makes the
        conditioning target exist at all.

        Two windows count as the same element when their starts lie within
        ``min_separation`` bp, matching :meth:`collapse_redundant_windows`.

        Args:
            windows: Windows with ``cell_type``, ``chrom`` and ``win_start``.
            min_separation: Distance in bp below which windows of different cell
                types are treated as the same locus. Values below 1 disable the
                filter, keeping every window.

        Returns:
            The cell-type-exclusive subset of ``windows``, index reset.
        """
        if min_separation < 1 or windows.empty:
            return windows.reset_index(drop=True)

        # Per (chrom, cell_type), the sorted starts, so "is there a window of
        # another cell type near this one?" is a binary search rather than a scan.
        starts_by_group: Dict[Any, np.ndarray] = {
            key: np.sort(group["win_start"].to_numpy())
            for key, group in windows.groupby(["chrom", "cell_type"], sort=False)
        }

        shared = np.zeros(len(windows), dtype=bool)
        for position, (chrom, cell_type, start) in enumerate(
            zip(windows["chrom"], windows["cell_type"], windows["win_start"])
        ):
            for (other_chrom, other_type), others in starts_by_group.items():
                if other_chrom != chrom or other_type == cell_type:
                    continue
                insert_at = np.searchsorted(others, start)
                neighbours = others[max(0, insert_at - 1):insert_at + 1]
                if neighbours.size and np.min(np.abs(neighbours - start)) < min_separation:
                    shared[position] = True
                    break

        exclusive = windows.loc[~shared].reset_index(drop=True)
        logger.info(
            "Dropped %d windows shared across cell types (<%d bp apart); %d cell-type-exclusive windows remain: %s",
            int(shared.sum()),
            min_separation,
            len(exclusive),
            exclusive["cell_type"].value_counts().to_dict(),
        )
        return exclusive

    def build_windows(
        self,
        manifest: List[Dict[str, Any]],
        loci_bed: Optional[str] = None,
        max_windows_per_cell_type: Optional[int] = None,
    ) -> pd.DataFrame:
        """Turn a download manifest into deduplicated, MS-restricted 1000 bp windows.

        Args:
            manifest: Records from ``GEODownloader.download_all()``; each needs
                ``local_path``, ``peak_format``, ``cell_type`` and ``gsm``.
            loci_bed: Risk-locus BED path; defaults to ``gwas.output_bed``.
            max_windows_per_cell_type: Optional cap, applied by descending peak
                score, to keep the training set balanced and tractable.

        Returns:
            DataFrame with ``chrom``, ``start``, ``end``, ``name``, ``cell_type``,
            ``peak_score``, ``fold_enrichment``, ``center_method``.

        Raises:
            ValueError: If ``manifest`` is empty or nothing survives intersection.
        """
        if not manifest:
            raise ValueError("Empty manifest: nothing to process.")

        loci = self.load_loci_bed(loci_bed or self.gwas_bed)
        method_counts: Dict[str, int] = {"summit": 0, "midpoint": 0}
        collected: List[pd.DataFrame] = []

        for record in manifest:
            frame = self.load_peak_file(record["local_path"], record.get("peak_format", "narrowPeak"))
            frame = self.filter_primary_chromosomes(frame)
            if frame.empty:
                continue

            has_summit = frame["summit_offset"].notna()
            summit_pos = np.where(
                has_summit,
                frame["start"] + frame["summit_offset"].fillna(0),
                (frame["start"] + frame["end"]) // 2,
            ).astype(np.int64)
            frame = frame.assign(summit_pos=summit_pos)
            method_counts["summit"] += int(has_summit.sum())
            method_counts["midpoint"] += int((~has_summit).sum())

            frame = self.intersect_with_loci(frame, loci)
            if frame.empty:
                logger.warning("Sample %s contributed no peaks inside MS risk loci.", record["gsm"])
                continue

            half = self.target_window_len // 2
            frame = frame.assign(
                cell_type=record["cell_type"],
                gsm=record["gsm"],
                win_start=(frame["summit_pos"] - half).clip(lower=0),
            )
            frame["win_end"] = frame["win_start"] + self.target_window_len
            collected.append(
                frame[["chrom", "win_start", "win_end", "cell_type", "gsm", "peak_score", "fold_enrichment"]]
            )

        if not collected:
            raise ValueError(
                "No peaks intersected the MS GWAS risk loci. Check that peak files and the "
                "risk-locus BED use the same genome build."
            )

        windows = pd.concat(collected, ignore_index=True)
        logger.info(
            "Pooled %d MS-locus peaks from %d samples (centering: %s).",
            len(windows),
            len(manifest),
            method_counts,
        )

        # Collapse windows recurrent across donors: one row per (cell_type, locus),
        # keeping the strongest observation so the condition vector stays meaningful.
        windows = (
            windows.sort_values("peak_score", ascending=False)
            .drop_duplicates(subset=["cell_type", "chrom", "win_start"], keep="first")
            .reset_index(drop=True)
        )
        windows = self.collapse_redundant_windows(windows, self.min_window_separation)
        if self.cell_type_exclusive:
            windows = self.drop_shared_windows(windows, self.min_window_separation)

        if max_windows_per_cell_type:
            windows = (
                windows.sort_values("peak_score", ascending=False)
                .groupby("cell_type", group_keys=False)
                .head(max_windows_per_cell_type)
                .reset_index(drop=True)
            )

        windows = windows.sort_values(["cell_type", "chrom", "win_start"]).reset_index(drop=True)
        windows["name"] = [
            f"{row.cell_type}|{row.chrom}:{row.win_start}-{row.win_end}" for row in windows.itertuples()
        ]
        windows = windows.rename(columns={"win_start": "start", "win_end": "end"})
        windows["center_method"] = "summit" if method_counts["summit"] >= method_counts["midpoint"] else "midpoint"

        counts = windows["cell_type"].value_counts().to_dict()
        logger.info("Built %d unique windows after deduplication: %s", len(windows), counts)
        return windows

    # ------------------------------------------------------------------- outputs

    def write_bed(self, windows: pd.DataFrame, filename: str = "ms_windows_1000bp.bed") -> str:
        """Write windows to a BED file under ``bed_dir``.

        Args:
            windows: DataFrame from :meth:`build_windows`.
            filename: Output file name.

        Returns:
            Path to the written BED file.
        """
        path = os.path.join(self.bed_dir, filename)
        windows[["chrom", "start", "end", "name", "peak_score"]].to_csv(
            path, sep="\t", header=False, index=False
        )
        logger.info("Wrote %d windows to %s.", len(windows), path)
        return path

    def extract_fasta(
        self,
        windows: pd.DataFrame,
        reference_fasta_path: Optional[str] = None,
        fasta_filename: str = "ms_windows_1000bp.fasta",
        metadata_filename: str = "ms_windows_metadata.csv",
    ) -> Tuple[str, str]:
        """Extract window sequences with ``pyfaidx`` and write FASTA + metadata.

        ``pyfaidx`` performs indexed random access, so hg38 is never loaded into
        memory. Windows that run off the end of a contig, or that exceed
        ``max_n_fraction`` ambiguous bases, are dropped and reported.

        Args:
            windows: DataFrame from :meth:`build_windows`.
            reference_fasta_path: Reference FASTA; defaults to ``reference_genome.local_fasta``.
            fasta_filename: Output FASTA name under ``fasta_dir``.
            metadata_filename: Output metadata CSV name under ``fasta_dir``.

        Returns:
            Tuple of ``(fasta_path, metadata_path)``. The metadata rows are aligned
            one-to-one, in order, with the FASTA records.

        Raises:
            FileNotFoundError: If the reference FASTA is absent.
            ValueError: If no window survives extraction.
        """
        from pyfaidx import Fasta  # imported lazily so tests can run without hg38

        reference = reference_fasta_path or self.reference_fasta
        if not os.path.exists(reference):
            raise FileNotFoundError(
                f"Reference genome FASTA not found: {reference}. "
                f"Download {self.config.get('reference_genome', {}).get('url')} and gunzip it."
            )

        genome = Fasta(reference, sequence_always_upper=True, rebuild=False)
        fasta_path = os.path.join(self.fasta_dir, fasta_filename)
        metadata_path = os.path.join(self.fasta_dir, metadata_filename)

        kept_rows: List[Dict[str, Any]] = []
        skipped = {"missing_contig": 0, "out_of_bounds": 0, "too_many_n": 0}

        with open(fasta_path, "w", encoding="utf-8") as handle:
            for row in windows.itertuples(index=False):
                if row.chrom not in genome:
                    skipped["missing_contig"] += 1
                    continue
                sequence = str(genome[row.chrom][int(row.start) : int(row.end)])
                if len(sequence) != self.target_window_len:
                    skipped["out_of_bounds"] += 1
                    continue
                if sequence.count("N") / len(sequence) > self.max_n_fraction:
                    skipped["too_many_n"] += 1
                    continue

                handle.write(f">{row.name}\n{sequence}\n")
                kept_rows.append(
                    {
                        "peak_id": row.name,
                        "chrom": row.chrom,
                        "start": int(row.start),
                        "end": int(row.end),
                        "cell_type": row.cell_type,
                        "peak_score": float(row.peak_score),
                        "fold_enrichment": float(row.fold_enrichment),
                        "source_gsm": row.gsm,
                    }
                )

        if not kept_rows:
            raise ValueError("No window sequences could be extracted; check the reference genome build.")

        pd.DataFrame(kept_rows).to_csv(metadata_path, index=False)
        logger.info(
            "Extracted %d sequences to %s (skipped %s); metadata -> %s.",
            len(kept_rows),
            fasta_path,
            skipped,
            metadata_path,
        )
        return fasta_path, metadata_path
