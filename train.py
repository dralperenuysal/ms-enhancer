"""Training script for MS-ENHANCER-GEN cVAE generator.

Usage:
    python train.py --config configs/model_config.yaml --batch_size 64 --epochs 100 --gpu_id 0
"""

import os
import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.helpers import setup_logging, set_seed, load_yaml_config, get_device, seeded_generator
from src.models.cvae_generator import CVAEGenerator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MS-ENHANCER-GEN generator model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--model_type", type=str, default="cvae", choices=["cvae", "transformer"], help="Model architecture type")
    parser.add_argument("--data_path", type=str, default="data/processed/processed_dataset.pt", help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID (negative for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of the dataset held out for validation")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic kernels so runs are reproducible across machines (slower)")
    parser.add_argument("--resume", action="store_true", help="Continue from the last checkpoint if one exists (for preemptible/spot VMs)")
    parser.add_argument("--amp", action="store_true", help="Mixed-precision training; on a T4 the FP16 tensor cores are several times faster than FP32 (CUDA only)")
    return parser.parse_args()


def capture_rng_state(loader_generator: torch.Generator) -> dict:
    """Snapshot every RNG stream the training loop draws from.

    Args:
        loader_generator: The generator driving DataLoader shuffling.

    Returns:
        Dictionary of RNG states, suitable for ``restore_rng_state``.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`.

    Resuming without this restarts every random stream from the seed, so dropout
    masks and batch order repeat epoch 1 instead of continuing — a resumed run
    would not match the uninterrupted run it is standing in for.

    Args:
        state: Dictionary from :func:`capture_rng_state`.
        loader_generator: The generator driving DataLoader shuffling.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader"])


@torch.no_grad()
def evaluate_split(model, loader, device, model_type: str, beta: float, use_amp: bool = False):
    """Compute mean loss and reconstruction loss over a held-out split.

    Args:
        model: The generator being trained.
        loader: DataLoader over the held-out split.
        device: Device the model lives on.
        model_type: Either ``"cvae"`` or ``"transformer"``.
        beta: Current KL weight, so the value is comparable to the training loss.
        use_amp: Evaluate under mixed precision, matching the training pass.

    Returns:
        Tuple of ``(mean_loss, mean_recon_loss)``.
    """
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    n_seen = 0

    for batch_x, batch_c in loader:
        batch_x, batch_c = batch_x.to(device), batch_c.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if model_type == "cvae":
                logits, mu, logvar = model(batch_x, batch_c)
                loss_dict = model.compute_loss(batch_x, logits, mu, logvar, beta=beta)
            else:
                logits = model(batch_x, batch_c)
                loss_dict = model.compute_loss(logits, batch_x)

        batch_size = len(batch_x)
        total_loss += loss_dict["loss"].item() * batch_size
        total_recon += loss_dict.get("recon_loss", loss_dict["loss"]).item() * batch_size
        n_seen += batch_size

    return total_loss / max(1, n_seen), total_recon / max(1, n_seen)


def train():
    args = parse_args()
    setup_logging(log_file=f"logs/train_{args.model_type}.log")
    set_seed(args.seed, deterministic=args.deterministic)

    config = load_yaml_config(args.config)
    device = get_device(args.gpu_id)

    if not os.path.exists(args.data_path):
        logger.error(
            f"Processed dataset file not found at '{args.data_path}'. "
            "Please run data processing (bed_processor.py & sequence_encoder.py) first."
        )
        raise FileNotFoundError(f"Dataset missing: {args.data_path}")

    logger.info(f"Loading processed dataset from {args.data_path}...")
    dataset_dict = torch.load(args.data_path)
    seqs = dataset_dict["sequences"]
    conds = dataset_dict["conditions"]

    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError(f"--val_fraction must be in [0, 1), got {args.val_fraction}")

    dataset = TensorDataset(seqs, conds)
    n_val = int(len(dataset) * args.val_fraction)
    n_train = len(dataset) - n_val
    if n_val == 0:
        raise ValueError(
            f"--val_fraction={args.val_fraction} yields an empty validation set for "
            f"{len(dataset)} samples. Model selection without held-out data is not supported."
        )

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=seeded_generator(args.seed)
    )
    # Shuffling draws from its own generator, so batch order does not depend on
    # how much of the global RNG stream model construction consumed.
    loader_generator = seeded_generator(args.seed)
    dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Split dataset: {n_train} training / {n_val} validation sequences.")

    if args.model_type == "cvae":
        cvae_cfg = config.get("cvae", {})
        model = CVAEGenerator(
            in_channels=cvae_cfg.get("in_channels", 4),
            sequence_length=cvae_cfg.get("sequence_length", 1000),
            condition_dim=conds.shape[1],
            latent_dim=cvae_cfg.get("latent_dim", 128),
            hidden_dims=cvae_cfg.get("hidden_dims", [64, 128, 256]),
            seed=args.seed
        ).to(device)
    else:
        from src.models.genomic_transformer import GenomicTransformer
        trans_cfg = config.get("genomic_transformer", {})
        model = GenomicTransformer(
            sequence_length=trans_cfg.get("sequence_length", 1000),
            num_tokens=trans_cfg.get("num_tokens", 4),
            condition_dim=conds.shape[1],
            d_model=trans_cfg.get("d_model", 128),
            nhead=trans_cfg.get("nhead", 4),
            num_layers=trans_cfg.get("num_layers", 4),
            dim_feedforward=trans_cfg.get("dim_feedforward", 256),
            seed=args.seed
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    cvae_cfg = config.get("cvae", {})
    kl_warmup_epochs = cvae_cfg.get("kl_warmup_epochs", 10)
    best_loss = float("inf")
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "models/generator")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Named after the architecture: a transformer run must never overwrite the
    # cVAE checkpoint, which is what a shared filename previously caused.
    best_ckpt_path = os.path.join(checkpoint_dir, f"{args.model_type}_best.pt")
    last_ckpt_path = os.path.join(checkpoint_dir, f"{args.model_type}_last.pt")

    start_epoch = 1
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        if ckpt["condition_dim"] != conds.shape[1]:
            raise ValueError(
                f"Checkpoint {last_ckpt_path} was trained with condition_dim="
                f"{ckpt['condition_dim']} but the dataset has {conds.shape[1]}. "
                f"Delete the checkpoint or point --data_path at the matching dataset."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        restore_rng_state(ckpt["rng_state"], loader_generator)
        logger.info(
            f"Resumed from {last_ckpt_path} at epoch {ckpt['epoch']} "
            f"(best validation loss so far: {best_loss:.4f})."
        )
    elif args.resume:
        logger.warning(f"--resume given but {last_ckpt_path} does not exist; starting from scratch.")

    if start_epoch > args.epochs:
        logger.info(f"Checkpoint is already at epoch {start_epoch - 1}; nothing to do.")
        return

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        logger.warning("--amp requested but no CUDA device is in use; training in FP32.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    logger.info(
        f"Starting {args.model_type} training for {args.epochs} epochs on {device} "
        f"(mixed precision: {use_amp})..."
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        # Linear KL annealing
        beta = min(1.0, epoch / max(1, kl_warmup_epochs)) * cvae_cfg.get("beta", 1.0)

        for batch_x, batch_c in dataloader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if args.model_type == "cvae":
                    logits, mu, logvar = model(batch_x, batch_c)
                    loss_dict = model.compute_loss(batch_x, logits, mu, logvar, beta=beta)
                else:
                    logits = model(batch_x, batch_c)
                    loss_dict = model.compute_loss(logits, batch_x)

            loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            # Gradients must be unscaled before clipping, or the clip threshold
            # would be applied to FP16-scaled values.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(batch_x)
            total_recon += loss_dict.get("recon_loss", loss).item() * len(batch_x)
            total_kl += loss_dict.get("kl_loss", torch.tensor(0.0)).item() * len(batch_x)

        n_samples = len(train_set)
        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_kl = total_kl / n_samples

        val_loss, val_recon = evaluate_split(model, val_loader, device, args.model_type, beta, use_amp)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs:03d} | Beta: {beta:.2f} | "
            f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
            f"Val loss: {val_loss:.4f} | Val recon: {val_recon:.4f}"
        )

        # Model selection is on held-out data: the training loss keeps falling even
        # when the model is only memorising, and these windows are genomic regions
        # that recur across donors.
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_checkpoint(best_ckpt_path, epoch=epoch, optimizer=optimizer, loss=best_loss)
            logger.info(f"New best validation loss {best_loss:.4f}; checkpoint written to {best_ckpt_path}.")

        # Written every epoch, best or not: on a preemptible VM this is the only
        # thing standing between a preemption and losing the run.
        model.save_checkpoint(
            last_ckpt_path,
            epoch=epoch,
            optimizer=optimizer,
            loss=val_loss,
            extra={"best_loss": best_loss, "rng_state": capture_rng_state(loader_generator)},
        )

    logger.info("Training finished successfully.")


if __name__ == "__main__":
    train()
