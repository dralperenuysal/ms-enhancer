"""Unit tests for train.py."""

import random

import numpy as np
import torch

from train import capture_rng_state, restore_rng_state
from src.utils.helpers import seeded_generator, set_seed


def test_rng_state_roundtrip_continues_every_stream():
    """Restoring a snapshot continues each stream exactly where it left off."""
    set_seed(99)
    generator = seeded_generator(99)

    # Burn some randomness so the snapshot is not the seeded starting state.
    random.random()
    np.random.rand()
    torch.randn(5)
    torch.randperm(10, generator=generator)

    snapshot = capture_rng_state(generator)
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.randn(3).tolist(),
        torch.randperm(10, generator=generator).tolist(),
    )

    # Simulate a fresh process: reseed, then restore the snapshot.
    set_seed(99)
    resumed_generator = seeded_generator(99)
    restore_rng_state(snapshot, resumed_generator)
    actual = (
        random.random(),
        float(np.random.rand()),
        torch.randn(3).tolist(),
        torch.randperm(10, generator=resumed_generator).tolist(),
    )

    assert actual == expected


def test_reseeding_alone_does_not_continue_the_stream():
    """Without restoration a resumed run replays the beginning of the stream."""
    set_seed(99)
    torch.randn(5)
    after_burn = torch.randn(3).tolist()

    set_seed(99)
    from_scratch = torch.randn(3).tolist()

    # This is the bug restore_rng_state exists to prevent.
    assert after_burn != from_scratch
