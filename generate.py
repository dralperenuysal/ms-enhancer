"""Sequence generation script for MS-ENHANCER-GEN.

Usage:
    python generate.py --checkpoint models/generator/cvae_best.pt --cell_type CD4_T_cell \
        --num_samples 1000 --out_fasta data/fasta/synthetic_cd4.fasta
"""

import os
import argparse
import logging
from typing import List, Optional

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.utils.helpers import setup_logging, set_seed, get_device, seeded_generator, load_yaml_config
from src.models.cvae_generator import CVAEGenerator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic MS enhancers")
    parser.add_argument("--generator", type=str, default="checkpoint", choices=["checkpoint", "markov"], help="'checkpoint' samples a trained network; 'markov' fits an order-k chain on the real windows of the target cell type at run time")
    parser.add_argument("--markov_order", type=int, default=None, help="Chain order for --generator markov (default: the configured order)")
    parser.add_argument("--windows_fasta", type=str, default="data/fasta/ms_windows_1000bp.fasta", help="Real windows the Markov chain is fitted on")
    parser.add_argument("--checkpoint", type=str, default="models/generator/cvae_best.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Model config, used only for architecture fields absent from older checkpoints")
    parser.add_argument("--cell_type", type=str, required=True, help="Target cell type; must be one of the cell types the model was trained on")
    parser.add_argument("--peak_score", type=float, default=None, help="Desired peak score in raw units (default: the training mean)")
    parser.add_argument("--fold_enrichment", type=float, default=None, help="Desired fold enrichment in raw units (default: the training mean)")
    parser.add_argument("--dataset", type=str, default="data/processed/processed_dataset.pt", help="Processed dataset supplying cell-type order and normalization constants")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic sequences to sample")
    parser.add_argument("--out_fasta", type=str, default="data/fasta/synthetic_ms_enhancers.fasta", help="Output FASTA filepath")
    parser.add_argument("--host_loci", type=str, default="data/fasta/ms_windows_metadata.csv", help="Real windows of this cell type; each generated sequence is assigned one as its host locus for in-silico scoring")
    parser.add_argument("--host_locus_id", type=str, default=None, help="Assign every candidate to this one peak_id instead of sampling loci. Required for selection: the host locus explains ~45x more MSSI variance than the insert, so candidates scored at different loci cannot be ranked against each other")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic kernels so output matches across machines (slower)")
    parser.add_argument("--sample_batch_size", type=int, default=100, help="Sequences sampled per autoregressive batch. The transformer has no KV cache, so peak attention memory scales with batch_size x sequence_length^2; chunking num_samples into batches of this size avoids GPU OOM on smaller cards (e.g. 1000 samples in one batch exceeds a T4's 15 GB). Does not change which cell type/condition is sampled, only how many rows go through forward() at once.")
    return parser.parse_args()


def build_condition_vector(args, condition_dim: int) -> torch.Tensor:
    """Build the condition vector for the requested cell type and signal level.

    The continuous features are normalized with the constants stored at training
    time. Feeding raw peak scores to a model trained on z-scores would place the
    request several standard deviations away from anything the model ever saw.

    Args:
        args: Parsed command-line arguments.
        condition_dim: Condition width recorded in the checkpoint.

    Returns:
        Condition tensor of shape ``(condition_dim,)``.

    Raises:
        FileNotFoundError: If the processed dataset is missing.
        ValueError: If the cell type is unknown, the dataset lacks normalization
            constants, or the widths disagree with the checkpoint.
    """
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Processed dataset not found: {args.dataset}. It supplies the cell-type "
            f"order and the normalization constants the checkpoint was trained with."
        )

    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    cell_types = dataset["cell_types"]
    stats = dataset.get("normalization_stats")

    if args.cell_type not in cell_types:
        raise ValueError(
            f"Unknown cell type '{args.cell_type}'. The model was trained on: {cell_types}."
        )
    if not stats:
        raise ValueError(
            f"{args.dataset} has no normalization_stats. Re-run "
            f"SequenceEncoder.process_and_save_dataset() to record them."
        )

    features = stats["features"]
    expected_dim = len(cell_types) + len(features)
    if expected_dim != condition_dim:
        raise ValueError(
            f"Checkpoint expects condition_dim={condition_dim} but the dataset describes "
            f"{len(cell_types)} cell types plus {len(features)} features ({expected_dim})."
        )

    cond = torch.zeros(condition_dim)
    cond[cell_types.index(args.cell_type)] = 1.0

    # Default to the training mean, i.e. a typical peak rather than an extreme one.
    requested = {"peak_score": args.peak_score, "fold_enrichment": args.fold_enrichment}
    for i, feature in enumerate(features):
        raw = requested.get(feature)
        center, scale = stats["center"][i], stats["scale"][i]
        cond[len(cell_types) + i] = 0.0 if raw is None else (raw - center) / scale

    return cond


def assign_host_loci(
    host_loci_path: str,
    cell_type: str,
    num_samples: int,
    seed: int,
    locus_id: Optional[str] = None,
) -> pd.DataFrame:
    """Assign each generated sequence a real genomic locus of the target cell type.

    A designed enhancer has no coordinates of its own, but Enformer scores a
    196 kb window, so the insert has to be placed somewhere real to be scored at
    all. Each sequence is therefore paired with an observed regulatory window of
    the same cell type, and the resulting MSSI answers "if this sequence replaced
    that element, what would the oracle predict?".

    Args:
        host_loci_path: CSV of real windows (``BEDProcessor.extract_fasta`` output).
        cell_type: Cell type to draw host loci from.
        num_samples: Number of loci to assign.
        seed: Seed for the sampling, so the assignment is reproducible.
        locus_id: Optional ``peak_id``; when given, every sequence is placed at
            that one locus so the oracle scores differ only by the insert.

    Returns:
        DataFrame with ``chrom``, ``start``, ``end`` and ``cell_type`` columns,
        one row per generated sequence.

    Raises:
        FileNotFoundError: If ``host_loci_path`` does not exist.
        ValueError: If the file has no windows for ``cell_type``, or if
            ``locus_id`` is not one of them.
    """
    if not os.path.exists(host_loci_path):
        raise FileNotFoundError(
            f"Host loci file not found: {host_loci_path}. Run BEDProcessor.extract_fasta() first, "
            f"or pass --host_loci."
        )

    windows = pd.read_csv(host_loci_path)
    candidates = windows[windows["cell_type"] == cell_type]
    if candidates.empty:
        raise ValueError(
            f"{host_loci_path} contains no windows for cell type '{cell_type}' "
            f"(available: {sorted(windows['cell_type'].unique())})."
        )

    if locus_id is not None:
        chosen = candidates[candidates["peak_id"] == locus_id]
        if chosen.empty:
            raise ValueError(
                f"Host locus '{locus_id}' is not a '{cell_type}' window in {host_loci_path}."
            )
        sampled = chosen.iloc[[0] * num_samples]
    else:
        # Sampling with replacement: there are usually fewer real windows than
        # requested sequences, and reusing a host locus is harmless here.
        sampled = candidates.sample(n=num_samples, replace=True, random_state=seed)

    return sampled[["chrom", "start", "end", "cell_type"]].reset_index(drop=True)


def sample_markov(args, sequence_length: int) -> List[str]:
    """Fit an order-k Markov chain on the target cell type's windows and sample from it.

    The chain is conditioned by construction rather than by a condition vector:
    it only ever sees windows of the requested cell type, so its output carries
    that cell type's k-mer composition. Fitting takes seconds, so there is no
    checkpoint to load and nothing to keep in sync with a training run.

    Args:
        args: Parsed command-line arguments.
        sequence_length: Length of the sequences to generate, in bp.

    Returns:
        List of ``args.num_samples`` nucleotide strings.

    Raises:
        FileNotFoundError: If the windows FASTA or its metadata is missing.
        ValueError: If no window of the requested cell type is available.
    """
    from src.evaluation.markov_baseline import MarkovBaseline

    for path in (args.windows_fasta, args.host_loci):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Markov generation needs the real windows: {path} not found. "
                f"Run scripts/build_dataset.py first."
            )

    metadata = pd.read_csv(args.host_loci)
    wanted = set(metadata.loc[metadata["cell_type"] == args.cell_type, "peak_id"])
    if not wanted:
        raise ValueError(
            f"{args.host_loci} contains no windows for cell type '{args.cell_type}' "
            f"(available: {sorted(metadata['cell_type'].unique())})."
        )

    training = [str(rec.seq).upper() for rec in SeqIO.parse(args.windows_fasta, "fasta") if rec.id in wanted]
    if not training:
        raise ValueError(
            f"No sequence in {args.windows_fasta} matched a '{args.cell_type}' peak_id from {args.host_loci}."
        )

    config = load_yaml_config(args.config).get("markov_baseline", {})
    order = args.markov_order if args.markov_order is not None else int(config.get("order", 6))
    chain = MarkovBaseline(order=order, pseudocount=float(config.get("pseudocount", 1.0)))
    chain.fit(training)

    logger.info(
        "Fitted order-%d Markov chain on %d real %s windows.", order, len(training), args.cell_type
    )
    return chain.sample(num_sequences=args.num_samples, length=sequence_length, seed=args.seed)


def decode_by_sampling(probabilities: torch.Tensor, generator: torch.Generator) -> List[str]:
    """Draw nucleotide sequences from the decoder's per-position distributions.

    Taking the argmax instead collapses each position onto its modal base. Where
    the decoder is uncertain — which is most of the sequence — neighbouring
    positions share the same modal base, so argmax emits long homopolymer runs
    that the model never actually assigned high probability to. Sampling
    reproduces the distribution the model represents.

    Args:
        probabilities: Tensor of shape ``(N, 4, L)`` of per-position probabilities.
        generator: Seeded generator, so decoding is reproducible.

    Returns:
        List of ``N`` nucleotide strings of length ``L``.
    """
    bases = "ACGT"
    total_samples = probabilities.shape[0]
    for i in range(total_samples):
        # multinomial expects (L, 4): one categorical distribution per position.
        indices = torch.multinomial(
            probabilities[i].transpose(0, 1), num_samples=1, generator=generator
        ).squeeze(1)
        sequences.append("".join(bases[j] for j in indices.tolist()))
        if (i + 1) % 250 == 0 or (i + 1) == total_samples:
            logger.info("[%d/%d - %d%%] Sampled synthetic regulatory sequences...", i + 1, total_samples, ((i + 1) * 100) // total_samples)

    return sequences


def write_output(args, sequences: List[str]) -> None:
    """Write the generated sequences as FASTA plus the metadata the oracle needs.

    Args:
        args: Parsed command-line arguments.
        sequences: Generated nucleotide strings, one per requested sample.
    """
    hosts = assign_host_loci(args.host_loci, args.cell_type, len(sequences), args.seed, args.host_locus_id)

    peak_ids = [f"syn_{args.cell_type}_{i + 1:04d}" for i in range(len(sequences))]
    records = [
        SeqRecord(Seq(seq), id=peak_id, description=f"synthetic {args.cell_type} regulatory region")
        for peak_id, seq in zip(peak_ids, sequences)
    ]

    os.makedirs(os.path.dirname(args.out_fasta) or ".", exist_ok=True)
    SeqIO.write(records, args.out_fasta, "fasta")

    # Written alongside the FASTA so the sequences can be scored: the oracle needs
    # a cell type and a host locus for every record.
    hosts.insert(0, "peak_id", peak_ids)
    metadata_path = os.path.splitext(args.out_fasta)[0] + "_metadata.csv"
    hosts.to_csv(metadata_path, index=False)

    logger.info(
        f"Successfully saved {len(records)} synthetic sequences to {args.out_fasta} "
        f"and their host loci to {metadata_path}"
    )


def generate():
    args = parse_args()
    setup_logging(log_file="logs/generate.log")
    set_seed(args.seed, deterministic=args.deterministic)

    if args.generator == "markov":
        sequence_length = load_yaml_config(args.config).get("genomic_transformer", {}).get("sequence_length", 1000)
        sequences = sample_markov(args, sequence_length)
        write_output(args, sequences)
        return

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = get_device()
    logger.info(f"Loading generator model from checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if "d_model" in ckpt:
        from src.models.genomic_transformer import GenomicTransformer

        # Architecture comes from the checkpoint. Older checkpoints predate these
        # fields, so fall back to the config the run was trained with rather than
        # to literals buried here.
        trans_cfg = load_yaml_config(args.config).get("genomic_transformer", {})
        model = GenomicTransformer(
            sequence_length=ckpt.get("sequence_length", 1000),
            num_tokens=ckpt.get("num_tokens", 4),
            condition_dim=ckpt.get("condition_dim", 5),
            d_model=ckpt.get("d_model", 128),
            nhead=ckpt.get("nhead", trans_cfg.get("nhead", 4)),
            num_layers=ckpt.get("num_layers", trans_cfg.get("num_layers", 4)),
            dim_feedforward=ckpt.get("dim_feedforward", trans_cfg.get("dim_feedforward", 256)),
            seed=args.seed
        ).to(device)
    else:
        model = CVAEGenerator(
            in_channels=ckpt["in_channels"],
            sequence_length=ckpt["sequence_length"],
            condition_dim=ckpt["condition_dim"],
            latent_dim=ckpt["latent_dim"],
            hidden_dims=ckpt["hidden_dims"],
            seed=args.seed
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cond = build_condition_vector(args, ckpt["condition_dim"])

    logger.info(
        f"Sampling {args.num_samples} sequences for cell type '{args.cell_type}' "
        f"with condition vector: {[round(v, 4) for v in cond.tolist()]}..."
    )
    if args.num_samples > args.sample_batch_size:
        # Chunked to bound peak attention memory (no KV cache: cost scales with
        # batch_size x sequence_length^2). The run is seeded once above via
        # set_seed(), so only the first chunk reseeds explicitly; later chunks
        # draw from the already-seeded RNG instead of repeating the same batch.
        chunks = []
        remaining = args.num_samples
        first = True
        while remaining > 0:
            batch = min(args.sample_batch_size, remaining)
            chunks.append(model.sample(condition=cond, num_samples=batch, seed=args.seed if first else None))
            remaining -= batch
            first = False
        prob_tensors = torch.cat(chunks, dim=0)
    else:
        prob_tensors = model.sample(condition=cond, num_samples=args.num_samples, seed=args.seed)

    sequences = decode_by_sampling(prob_tensors.cpu(), seeded_generator(args.seed))
    write_output(args, sequences)


if __name__ == "__main__":
    generate()
