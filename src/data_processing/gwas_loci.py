"""MS GWAS risk locus retrieval for MS-ENHANCER-GEN.

Fetches multiple-sclerosis-associated variants from the EBI GWAS Catalog REST API
and writes them out as a BED file of risk-locus windows on GRCh38/hg38.

No coordinates are hardcoded anywhere: the trait identifier lives in
``configs/data_config.yaml`` and every position comes from the live API.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class GWASCatalogError(RuntimeError):
    """Raised when the GWAS Catalog API is unusable or returns unexpected content."""


class MSGWASLoci:
    """Retrieves MS risk loci from the GWAS Catalog and materialises them as BED."""

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """Initialize from the data configuration file.

        Args:
            config_path: Path to ``data_config.yaml``.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the GWAS build does not match the reference genome build.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle)

        gwas = self.config.get("gwas", {})
        self.trait_id: str = gwas.get("trait_id", "MONDO_0005301")
        self.trait_label: str = gwas.get("trait_label", "multiple sclerosis")
        self.api_base: str = gwas.get("api_base", "https://www.ebi.ac.uk/gwas/rest/api")
        self.flank_bp: int = int(gwas.get("locus_flank_bp", 100000))
        self.page_size: int = int(gwas.get("page_size", 500))
        self.output_bed: str = gwas.get("output_bed", "data/bed/ms_gwas_loci_hg38.bed")
        self.declared_build: str = gwas.get("genome_build", "GRCh38")

        ref = self.config.get("reference_genome", {})
        accepted = [b.lower() for b in ref.get("equivalent_builds", [ref.get("version", "hg38")])]
        if self.declared_build.lower() not in accepted:
            raise ValueError(
                f"gwas.genome_build='{self.declared_build}' is not compatible with "
                f"reference_genome builds {accepted}. Liftover is not implemented."
            )

        self.primary_chromosomes: List[str] = ref.get(
            "primary_chromosomes", [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
        )

        logger.info(
            "MSGWASLoci initialised (trait=%s '%s', flank=%d bp, build=%s).",
            self.trait_id,
            self.trait_label,
            self.flank_bp,
            self.declared_build,
        )

    @staticmethod
    def _get_json(url: str, timeout: int = 120) -> Dict[str, Any]:
        """GET a URL and parse the response as JSON.

        Args:
            url: URL to fetch.
            timeout: Socket timeout in seconds.

        Returns:
            Parsed JSON object.

        Raises:
            GWASCatalogError: If the request fails or the body is not JSON.
        """
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - re-raised as a domain error
            raise GWASCatalogError(f"GWAS Catalog request failed for {url}: {error}") from error

    def fetch_variants(self, max_pages: int = 40) -> pd.DataFrame:
        """Fetch every MS-associated variant with a mapped genomic position.

        Args:
            max_pages: Safety cap on pagination.

        Returns:
            DataFrame with columns ``rs_id``, ``chrom``, ``pos``, ``functional_class``.

        Raises:
            GWASCatalogError: If the API returns no variants for the trait.
        """
        url = (
            f"{self.api_base}/singleNucleotidePolymorphisms/search/findByEfoTrait?"
            + urllib.parse.urlencode({"efoTrait": self.trait_label, "size": self.page_size})
        )

        rows: List[Dict[str, Any]] = []
        seen: set = set()
        pages = 0

        while url and pages < max_pages:
            payload = self._get_json(url)
            variants = payload.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
            for variant in variants:
                rs_id = variant.get("rsId")
                for location in variant.get("locations", []):
                    chrom_name = location.get("chromosomeName")
                    position = location.get("chromosomePosition")
                    if chrom_name is None or position is None:
                        continue
                    chrom = f"chr{chrom_name}" if not str(chrom_name).startswith("chr") else str(chrom_name)
                    key = (rs_id, chrom, int(position))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "rs_id": rs_id,
                            "chrom": chrom,
                            "pos": int(position),
                            "functional_class": variant.get("functionalClass"),
                        }
                    )
            pages += 1
            url = payload.get("_links", {}).get("next", {}).get("href")

        if not rows:
            raise GWASCatalogError(
                f"GWAS Catalog returned no mapped variants for trait '{self.trait_label}' "
                f"({self.trait_id}). Check the trait identifier in configs/data_config.yaml."
            )

        frame = pd.DataFrame(rows)
        logger.info("Fetched %d unique MS-associated variants across %d API pages.", len(frame), pages)
        return frame

    def to_loci_bed(self, variants: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Expand variants into flanked risk-locus windows and write a BED file.

        Overlapping windows are merged so that a dense LD region contributes one
        locus rather than many near-duplicates.

        Args:
            variants: DataFrame from :meth:`fetch_variants`.
            output_path: Destination BED path; defaults to ``gwas.output_bed``.

        Returns:
            Path to the written BED file.

        Raises:
            ValueError: If ``variants`` is empty or missing required columns.
        """
        required = {"rs_id", "chrom", "pos"}
        missing = required - set(variants.columns)
        if missing:
            raise ValueError(f"variants DataFrame is missing columns: {sorted(missing)}")
        if variants.empty:
            raise ValueError("variants DataFrame is empty; refusing to write an empty GWAS BED file.")

        frame = variants[variants["chrom"].isin(self.primary_chromosomes)].copy()
        dropped = len(variants) - len(frame)
        if dropped:
            logger.info("Dropped %d variants on non-primary contigs.", dropped)

        frame["start"] = (frame["pos"] - 1 - self.flank_bp).clip(lower=0)
        frame["end"] = frame["pos"] + self.flank_bp
        frame = frame.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

        merged: List[Dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            if merged and merged[-1]["chrom"] == row.chrom and row.start <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], row.end)
                merged[-1]["variants"].append(row.rs_id)
            else:
                merged.append(
                    {"chrom": row.chrom, "start": int(row.start), "end": int(row.end), "variants": [row.rs_id]}
                )

        out_frame = pd.DataFrame(
            {
                "chrom": [m["chrom"] for m in merged],
                "start": [m["start"] for m in merged],
                "end": [m["end"] for m in merged],
                "name": [",".join(m["variants"][:5]) + ("..." if len(m["variants"]) > 5 else "") for m in merged],
                "score": [len(m["variants"]) for m in merged],
            }
        )

        destination = output_path or self.output_bed
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        out_frame.to_csv(destination, sep="\t", header=False, index=False)

        total_bp = int((out_frame["end"] - out_frame["start"]).sum())
        logger.info(
            "Wrote %d merged MS risk loci (%d variants, %.1f Mb) to %s.",
            len(out_frame),
            len(frame),
            total_bp / 1e6,
            destination,
        )
        return destination

    def build(self, output_path: Optional[str] = None) -> str:
        """Fetch variants and write the risk-locus BED in one call.

        Args:
            output_path: Destination BED path; defaults to ``gwas.output_bed``.

        Returns:
            Path to the written BED file.
        """
        return self.to_loci_bed(self.fetch_variants(), output_path=output_path)
