"""Rebuild the training dataset from downloaded peak files.

Reads the manifest written by GEODownloader, rebuilds the windows, extracts their
sequences from the reference genome and writes the encoded tensor dataset.
No network access: everything it needs is already under data/raw.

Usage:
    python scripts/build_dataset.py
"""

import argparse
import logging

from src.data_processing.bed_processor import BEDProcessor
from src.data_processing.sequence_encoder import SequenceEncoder
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild windows, FASTA and encoded dataset")
    parser.add_argument("--manifest", default="data/raw/peak_manifest.json")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    setup_logging(log_file="logs/build_dataset.log")

    import json
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    processor = BEDProcessor(config_path=args.config)
    windows = processor.build_windows(manifest)
    processor.write_bed(windows)
    fasta_path, metadata_path = processor.extract_fasta(windows)

    encoder = SequenceEncoder(config_path=args.config)
    encoder.process_and_save_dataset(fasta_path, metadata_path)
    logger.info("Dataset rebuilt from %s.", args.manifest)


if __name__ == "__main__":
    main()
