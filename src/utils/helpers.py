"""Helper utilities module for MS-ENHANCER-GEN.

Provides functions for logging setup, random seed initialization, YAML configuration loading,
and hardware device detection.
"""

import os
import random
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup structured logging for MS-ENHANCER-GEN pipeline.

    Args:
        level: Logging level (default: logging.INFO).
        log_file: Optional path to log file under logs/.

    Returns:
        Configured Logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')

    # Remove existing handlers to prevent duplication
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch.

    Seeding alone makes a run repeatable on the same machine. It does not make it
    repeatable across machines: cuDNN picks convolution algorithms by benchmarking
    the local hardware, and several CUDA kernels accumulate in nondeterministic
    order. ``deterministic=True`` pins those choices, at a cost in speed, and is
    what a CPU run and a GPU run need in order to be comparable.

    Args:
        seed: Fixed integer seed value.
        deterministic: Force deterministic kernels. PyTorch raises
            ``RuntimeError`` from any op that has no deterministic implementation,
            which is intentional — a silent nondeterministic op is what makes a
            "reproducible" run irreproducible on other hardware.
    """
    # Affects hash randomisation only in subprocesses; set for completeness so a
    # spawned DataLoader worker inherits the same ordering behaviour.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # No-op without CUDA.

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        # Required by cuBLAS (CUDA >= 10.2) for reproducible GEMMs; must be set
        # before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    logger.info("Global seed set to %d (deterministic=%s).", seed, deterministic)


def seeded_generator(seed: int) -> torch.Generator:
    """Return a torch Generator for data shuffling, independent of global RNG.

    A DataLoader with ``shuffle=True`` and no explicit generator draws from the
    global RNG, whose state at that moment depends on how many random numbers
    model construction happened to consume. That makes batch order differ between
    the cVAE and the transformer, and between any two runs whose setup differs.

    Args:
        seed: Fixed integer seed value.

    Returns:
        A seeded ``torch.Generator``.
    """
    return torch.Generator().manual_seed(seed)


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file.

    Args:
        filepath: Path to YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If filepath does not exist.
        yaml.YAMLError: If YAML content is malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML content in {filepath}. Expected dict, got {type(config)}.")

    return config


def get_device(gpu_id: Optional[int] = 0) -> torch.device:
    """Detect hardware device (GPU vs CPU) and return PyTorch device object.

    Args:
        gpu_id: Optional CUDA GPU index.

    Returns:
        torch.device object.
    """
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    if cuda_avail and gpu_id is not None and gpu_id >= 0 and device_count > 0:
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda}, {device_count} GPUs available).")
    else:
        device = torch.device("cpu")
        logger.info(f"CUDA not available or disabled (is_available={cuda_avail}, count={device_count}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}). Running on CPU.")

    return device
