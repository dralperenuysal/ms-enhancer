"""Unit tests for src/data_processing/gwas_loci.py.

All GWAS Catalog API calls are mocked via ``MSGWASLoci._get_json``.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_processing.gwas_loci import GWASCatalogError, MSGWASLoci

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = """\
gwas:
  trait_id: "MONDO_0005301"
  trait_label: "multiple sclerosis"
  api_base: "https://www.ebi.ac.uk/gwas/rest/api"
  locus_flank_bp: 100000
  page_size: 500
  output_bed: "{bed_path}"
  genome_build: "GRCh38"

reference_genome:
  version: "hg38"
  equivalent_builds: ["hg38", "GRCh38"]
  primary_chromosomes:
    - chr1
    - chr2
    - chr6
    - chrX
"""


def _make_api_response(variants, next_url=None):
    """Build a GWAS Catalog-style paginated JSON response."""
    snps = []
    for v in variants:
        snps.append(
            {
                "rsId": v["rs_id"],
                "functionalClass": v.get("functional_class", "intergenic_variant"),
                "locations": [
                    {
                        "chromosomeName": v["chrom"].replace("chr", ""),
                        "chromosomePosition": v["pos"],
                    }
                ],
            }
        )
    payload = {"_embedded": {"singleNucleotidePolymorphisms": snps}, "_links": {}}
    if next_url:
        payload["_links"]["next"] = {"href": next_url}
    return payload


@pytest.fixture
def temp_cfg():
    """Temp directory with a valid data_config.yaml and bed output path."""
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = os.path.join(tmp, "ms_gwas_loci.bed")
        cfg_path = os.path.join(tmp, "data_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_CONFIG.format(bed_path=bed_path))
        yield cfg_path, bed_path, tmp


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def test_init_missing_config() -> None:
    """FileNotFoundError if config does not exist."""
    with pytest.raises(FileNotFoundError):
        MSGWASLoci(config_path="/nonexistent/path.yaml")


def test_init_valid(temp_cfg) -> None:
    """Successful init parses all config fields."""
    cfg_path, _, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)
    assert loci.trait_id == "MONDO_0005301"
    assert loci.flank_bp == 100000
    assert "chr6" in loci.primary_chromosomes


def test_init_build_mismatch() -> None:
    """ValueError when GWAS build doesn't match reference genome builds."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "bad.yaml")
        with open(cfg, "w") as fh:
            fh.write(
                "gwas:\n  genome_build: hg19\n"
                "reference_genome:\n  version: hg38\n  equivalent_builds: [hg38, GRCh38]\n"
            )
        with pytest.raises(ValueError, match="not compatible"):
            MSGWASLoci(config_path=cfg)


# ---------------------------------------------------------------------------
# fetch_variants
# ---------------------------------------------------------------------------


@patch.object(MSGWASLoci, "_get_json")
def test_fetch_variants(mock_json, temp_cfg) -> None:
    """Fetches and deduplicates variants from paginated API."""
    cfg_path, _, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)

    mock_json.return_value = _make_api_response(
        [
            {"rs_id": "rs123", "chrom": "chr6", "pos": 32000000},
            {"rs_id": "rs456", "chrom": "chr1", "pos": 10000000},
            {"rs_id": "rs123", "chrom": "chr6", "pos": 32000000},  # duplicate
        ]
    )

    df = loci.fetch_variants(max_pages=1)
    assert len(df) == 2
    assert set(df["rs_id"]) == {"rs123", "rs456"}
    assert all(df["chrom"].str.startswith("chr"))


@patch.object(MSGWASLoci, "_get_json")
def test_fetch_variants_empty_raises(mock_json, temp_cfg) -> None:
    """GWASCatalogError when API returns no variants."""
    cfg_path, _, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)
    mock_json.return_value = {"_embedded": {"singleNucleotidePolymorphisms": []}, "_links": {}}

    with pytest.raises(GWASCatalogError, match="no mapped variants"):
        loci.fetch_variants()


# ---------------------------------------------------------------------------
# to_loci_bed
# ---------------------------------------------------------------------------


def test_to_loci_bed(temp_cfg) -> None:
    """Writes a valid BED file from a variants DataFrame."""
    cfg_path, bed_path, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)

    variants = pd.DataFrame(
        {
            "rs_id": ["rs100", "rs200"],
            "chrom": ["chr6", "chr1"],
            "pos": [32000000, 10000000],
            "functional_class": ["intergenic_variant", "intron_variant"],
        }
    )
    path = loci.to_loci_bed(variants, output_path=bed_path)
    assert os.path.exists(path)

    bed = pd.read_csv(path, sep="\t", header=None)
    assert len(bed) == 2
    # Each locus: start = pos-1-flank, end = pos+flank → width = 2*flank+1
    widths = bed[2] - bed[1]
    assert all(widths == 200001)


def test_to_loci_bed_merge_overlapping(temp_cfg) -> None:
    """Overlapping windows on the same chromosome are merged."""
    cfg_path, bed_path, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)

    # Two variants 50k apart on chr6 — with 100k flanks they overlap
    variants = pd.DataFrame(
        {
            "rs_id": ["rs1", "rs2"],
            "chrom": ["chr6", "chr6"],
            "pos": [32000000, 32050000],
            "functional_class": [None, None],
        }
    )
    path = loci.to_loci_bed(variants, output_path=bed_path)
    bed = pd.read_csv(path, sep="\t", header=None)
    assert len(bed) == 1  # merged into one locus


def test_to_loci_bed_empty_raises(temp_cfg) -> None:
    """Empty variants DataFrame raises ValueError."""
    cfg_path, _, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)
    with pytest.raises(ValueError, match="empty"):
        loci.to_loci_bed(pd.DataFrame(columns=["rs_id", "chrom", "pos"]))


def test_to_loci_bed_missing_columns(temp_cfg) -> None:
    """Missing columns raise ValueError."""
    cfg_path, _, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)
    with pytest.raises(ValueError, match="missing columns"):
        loci.to_loci_bed(pd.DataFrame({"rs_id": ["rs1"], "chrom": ["chr1"]}))


def test_to_loci_bed_drops_non_primary(temp_cfg) -> None:
    """Variants on non-primary chromosomes are filtered out."""
    cfg_path, bed_path, _ = temp_cfg
    loci = MSGWASLoci(config_path=cfg_path)

    variants = pd.DataFrame(
        {
            "rs_id": ["rs1", "rs2"],
            "chrom": ["chr6", "chrUn_gl000220"],
            "pos": [32000000, 50000],
            "functional_class": [None, None],
        }
    )
    path = loci.to_loci_bed(variants, output_path=bed_path)
    bed = pd.read_csv(path, sep="\t", header=None)
    assert len(bed) == 1
    assert bed.iloc[0, 0] == "chr6"
