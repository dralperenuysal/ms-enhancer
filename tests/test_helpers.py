"""Unit tests for src/utils/helpers.py."""

import os
import tempfile
import pytest
import torch
import numpy as np

from src.utils.helpers import setup_logging, set_seed, load_yaml_config, get_device, seeded_generator


def test_setup_logging():
    """Test setting up logger with file handler."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, "test.log")
        logger = setup_logging(log_file=log_path)

        logger.info("Test log message")
        assert os.path.exists(log_path)

        with open(log_path, "r") as f:
            content = f.read()
            assert "Test log message" in content


def test_set_seed():
    """Test global seed setting for reproducibility."""
    set_seed(1234)
    val1 = torch.randn(2, 2)
    np_val1 = np.random.rand(2, 2)

    set_seed(1234)
    val2 = torch.randn(2, 2)
    np_val2 = np.random.rand(2, 2)

    assert torch.allclose(val1, val2)
    assert np.allclose(np_val1, np_val2)


def test_load_yaml_config():
    """Test loading YAML configuration file."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("param1: 100\nparam2: 'test'\n")
        path = f.name

    try:
        config = load_yaml_config(path)
        assert config["param1"] == 100
        assert config["param2"] == "test"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_load_yaml_config_missing():
    """Test loading missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("non_existent_config_file.yaml")


def test_get_device():
    """Test device selection (CPU or CUDA)."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_set_seed_is_reproducible():
    """The same seed reproduces the same stream from all three RNGs."""
    import random
    import numpy as np
    import torch

    set_seed(1234)
    first = (random.random(), np.random.rand(), torch.randn(3).tolist())

    set_seed(1234)
    second = (random.random(), np.random.rand(), torch.randn(3).tolist())

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_set_seed_deterministic_flag_sets_backend_state():
    """The deterministic flag pins cuDNN and PyTorch algorithm selection."""
    import torch

    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    # Restore the default so the flag does not leak into other tests.
    set_seed(0, deterministic=False)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_seeded_generator_is_independent_of_global_rng():
    """Shuffling order must not depend on prior global RNG consumption."""
    import torch

    set_seed(7)
    baseline = torch.randperm(10, generator=seeded_generator(7)).tolist()

    set_seed(7)
    torch.randn(1000)  # Simulate model construction consuming the global stream.
    after = torch.randperm(10, generator=seeded_generator(7)).tolist()

    assert baseline == after
