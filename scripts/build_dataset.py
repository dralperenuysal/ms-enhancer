"""Build or rebuild the training dataset from GWAS loci and GEO peak files.

Supports both:
1. Local rebuild mode (default, no network access if files already exist).
2. Autonomous download mode (fetches missing GWAS loci from EBI Catalog and
   peak files from NCBI GEO when custom accessions are supplied).

Usage:
    # Default MS rebuild (from local cache):
    python scripts/build_dataset.py

    # Dynamic multi-disease fetch & build:
    python scripts/build_dataset.py \
        --gwas_id EFO_0000729 \
        --gwas_label "ulcerative colitis" \
        --gse GSE282442 \
        --cell_type epithelial_inflamed \
        --suffix uc
"""

import argparse
import json
import logging
import os
import re
import yaml

from src.data_processing.bed_processor import BEDProcessor
from src.data_processing.geo_downloader import GEODownloader
from src.data_processing.gwas_loci import MSGWASLoci
from src.data_processing.sequence_encoder import SequenceEncoder
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build or rebuild dataset from GWAS and GEO data")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Path to data configuration YAML")
    parser.add_argument("--manifest", default=None, help="Path to peak_manifest.json (default: derived from raw_dir)")
    parser.add_argument("--suffix", default="ms", help="Experiment suffix identifier (e.g. 'uc', 'ms', 't1d')")
    parser.add_argument("--gwas_id", default=None, help="EFO/MONDO accession or GWAS Catalog URL")
    parser.add_argument("--gwas_label", default=None, help="GWAS Catalog trait label (e.g. 'ulcerative colitis')")
    parser.add_argument("--gse", default=None, help="GEO GSE accession number (e.g. 'GSE282442')")
    parser.add_argument("--cell_type", default=None, help="Default cell type label for custom GSE")
    parser.add_argument("--download", action="store_true", help="Force download even if manifest exists")
    return parser.parse_args()


def extract_clean_accession(val: str) -> str:
    """Extract canonical accession from raw string or URL (e.g., https://.../EFO_0000729 -> EFO_0000729)."""
    if not val:
        return val
    match = re.search(r"(EFO_\d+|MONDO_\d+|HP_\d+|GSE\d+)", val)
    return match.group(1) if match else val.strip()


def ensure_reference_genome(config: dict) -> str:
    """Ensure the local reference genome FASTA exists, checking local project and downloading if needed."""
    import gzip
    import shutil
    import subprocess
    import urllib.request

    ref = config.get("reference_genome", {})
    local_fasta = ref.get("local_fasta", "data/hg38.fa")
    
    # Check if already present in workdir or parent repo
    if os.path.exists(local_fasta):
        return local_fasta

    # Check if present in parent repo directory
    for candidate in [os.path.join("..", "..", local_fasta), os.path.abspath(local_fasta)]:
        if os.path.exists(candidate):
            os.makedirs(os.path.dirname(local_fasta) or ".", exist_ok=True)
            try:
                os.symlink(os.path.abspath(candidate), local_fasta)
                logger.info("Symlinked existing reference genome from %s -> %s", candidate, local_fasta)
                return local_fasta
            except OSError:
                pass

    url = ref.get("url", "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz")
    logger.info("Reference genome FASTA not found at %s. Downloading from %s...", local_fasta, url)
    os.makedirs(os.path.dirname(local_fasta) or ".", exist_ok=True)
    gz_path = local_fasta + ".gz"

    # Try fast wget/curl if available, fallback to urllib
    downloaded = False
    if shutil.which("wget"):
        res = subprocess.run(["wget", "-c", url, "-O", gz_path], capture_output=False)
        downloaded = (res.returncode == 0 and os.path.exists(gz_path))
    elif shutil.which("curl"):
        res = subprocess.run(["curl", "-L", url, "-o", gz_path], capture_output=False)
        downloaded = (res.returncode == 0 and os.path.exists(gz_path))

    if not downloaded:
        logger.info("Downloading via python urllib...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(gz_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f, length=16 * 1024 * 1024)

    logger.info("Extracting %s -> %s ...", gz_path, local_fasta)
    with gzip.open(gz_path, "rb") as f_in:
        with open(local_fasta, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)

    if os.path.exists(gz_path):
        os.remove(gz_path)

    logger.info("Reference genome prepared successfully at %s.", local_fasta)
    return local_fasta


def main() -> None:
    args = parse_args()
    setup_logging(log_file="logs/build_dataset.log")

    # 1. Load base configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Dynamic override if user passed CLI flags
    if args.gwas_id:
        clean_gwas_id = extract_clean_accession(args.gwas_id)
        config.setdefault("gwas", {})
        config["gwas"]["trait_id"] = clean_gwas_id
        if args.gwas_label:
            config["gwas"]["trait_label"] = args.gwas_label
        config["gwas"]["output_bed"] = f"data/bed/{args.suffix}_gwas_loci_hg38.bed"

    if args.gse:
        clean_gse = extract_clean_accession(args.gse)
        cell_type = args.cell_type or f"{args.suffix}_cell"
        logger.warning(
            "No 'expected' metadata block for GSE %s (CLI-provided GSE skips organism/"
            "library_strategy/n_samples verification). Add one to configs/data_config.yaml "
            "to enable GEODownloader.verify_dataset() checks.",
            clean_gse,
        )
        config.setdefault("geo_datasets", {})
        config["geo_datasets"]["verified_datasets"] = [
            {
                "accession": clean_gse,
                "cell_type_default": cell_type,
                "peak_format": "narrowPeak",
            }
        ]
        config.setdefault("condition_encoding", {})
        config["condition_encoding"]["cell_types"] = [cell_type]

    # Write effective config to temp runtime file if overridden
    effective_config_path = args.config
    if args.gwas_id or args.gse:
        os.makedirs("configs", exist_ok=True)
        effective_config_path = f"configs/.runtime_data_config_{args.suffix}.yaml"
        with open(effective_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

    # 3. Ensure GWAS Risk Loci BED exists (fetch if missing or explicitly provided)
    gwas_bed = config.get("gwas", {}).get("output_bed", f"data/bed/{args.suffix}_gwas_loci_hg38.bed")
    if not os.path.exists(gwas_bed) or args.gwas_id:
        logger.info("GWAS risk loci BED not found at %s. Fetching from EBI API...", gwas_bed)
        gwas_fetcher = MSGWASLoci(config_path=effective_config_path)
        gwas_fetcher.build(output_path=gwas_bed)

    # 4. Ensure GEO Data & Manifest exist (download if missing or custom GSE requested)
    raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
    manifest_path = args.manifest or os.path.join(raw_dir, f"{args.suffix}_peak_manifest.json" if args.gse else "peak_manifest.json")

    if not os.path.exists(manifest_path) or args.gse or args.download:
        logger.info("Downloading peaks for configured datasets via GEODownloader...")
        downloader = GEODownloader(config_path=effective_config_path)
        manifest = downloader.download_all()
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    else:
        logger.info("Loading existing manifest from %s", manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # 5. Build 1000 bp windows and extract FASTA
    ensure_reference_genome(config)
    processor = BEDProcessor(config_path=effective_config_path)
    windows = processor.build_windows(manifest)
    processor.write_bed(windows)
    fasta_path, metadata_path = processor.extract_fasta(windows)

    # 6. One-hot encode sequences and build condition tensors
    encoder = SequenceEncoder(config_path=effective_config_path)
    encoder.process_and_save_dataset(fasta_path, metadata_path)
    logger.info("Dataset construction complete from %s.", manifest_path)


if __name__ == "__main__":
    main()
