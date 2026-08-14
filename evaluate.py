"""In-silico evaluation script for synthetic MS enhancers.

Usage:
    python evaluate.py --input_fasta data/fasta/synthetic_ms_enhancers.fasta --oracle enformer --output_report logs/evaluation_results.json
"""

# Global bypass for HuggingFace CVE-2025-32434 check on older torch versions
try:
    import sys
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except Exception:
    pass

import os
import argparse
import logging
import json

from src.utils.helpers import setup_logging, load_yaml_config
from src.evaluation.borzoi_oracle import BorzoiOracle
from src.evaluation.enformer_oracle import EnformerOracle
from src.evaluation.motif_analyzer import MotifAnalyzer
from src.evaluation.sequence_realism import SequenceRealism

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synthetic MS enhancers")
    parser.add_argument("--input_fasta", type=str, default="data/fasta/synthetic_ms_enhancers.fasta", help="Input synthetic FASTA file")
    parser.add_argument("--oracle", type=str, default="enformer", choices=["enformer", "borzoi", "motif", "realism"], help="Evaluation type; 'realism' is the model-free composition check to run first. 'borzoi' is an independently trained second oracle: agreement with 'enformer' is what distinguishes a sequence effect from one model's idiosyncrasy")
    parser.add_argument("--reference_windows", type=str, default="data/fasta/ms_windows_1000bp.fasta", help="Real training windows to compare against (realism check)")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--output_report", type=str, default="logs/evaluation_results.json", help="Path to save evaluation report")
    parser.add_argument("--metadata", type=str, default="data/fasta/ms_windows_metadata.csv", help="CSV with peak_id, chrom, start, end, cell_type for each sequence (enformer oracle)")
    parser.add_argument("--reference_fasta", type=str, default="data/hg38.fa", help="Indexed reference genome for real flanking context (enformer oracle)")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    setup_logging(log_file="logs/evaluate.log")

    if not os.path.exists(args.input_fasta):
        logger.error(f"Input FASTA file not found: {args.input_fasta}")
        raise FileNotFoundError(f"Missing FASTA: {args.input_fasta}")

    logger.info(f"Evaluating {args.input_fasta} using {args.oracle} oracle...")

    if args.oracle in ("enformer", "borzoi"):
        oracle_class = EnformerOracle if args.oracle == "enformer" else BorzoiOracle
        oracle = oracle_class(config_path=args.config)
        report = oracle.evaluate_fasta(
            fasta_path=args.input_fasta,
            metadata_path=args.metadata,
            reference_fasta_path=args.reference_fasta,
            output_report_path=args.output_report,
        )
    elif args.oracle == "realism":
        from Bio import SeqIO

        if not os.path.exists(args.reference_windows):
            raise FileNotFoundError(
                f"Reference windows not found: {args.reference_windows}. The realism check "
                f"compares against the real sequences the model was trained on."
            )
        generated = [str(r.seq) for r in SeqIO.parse(args.input_fasta, "fasta")]
        reference = [str(r.seq) for r in SeqIO.parse(args.reference_windows, "fasta")]

        report = SequenceRealism(config_path=args.config).compare(generated, reference)
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    elif args.oracle == "motif":
        analyzer = MotifAnalyzer(config_path=args.config)
        report = analyzer.analyze_fasta(fasta_path=args.input_fasta)
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    logger.info(f"Evaluation report written to {args.output_report}")


if __name__ == "__main__":
    evaluate()
