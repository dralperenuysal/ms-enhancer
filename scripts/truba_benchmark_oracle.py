"""Time one Enformer/Borzoi forward pass on whichever GPU this lands on.

TRUBA's two open GPU queues differ by far more than their spec sheets suggest:
a P100 (barbun-cuda, no tensor cores) has been measured 35-40x slower than a
V100 (akya-cuda) on attention-heavy work, enough to turn a feasible job into an
impossible one. The campaign plan depends on knowing whether that gap applies
here, so this is run on both node types before anything long is submitted.

It also verifies that the driver actually supports the container's CUDA build:
a mismatch leaves torch.cuda.is_available() False, the job silently falls back
to CPU, and the GPU hours are wasted in the queue for nothing.

Usage:
    srun -p debug --gres=gpu:1 -w <node> --time=00:20:00 \
        apptainer exec --nv --bind /arf ms-enhancer.sif \
        python3 scripts/truba_benchmark_oracle.py --oracle enformer --reps 3
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", choices=["enformer", "borzoi"], default="enformer")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    import torch

    report = {
        "host": platform.node(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        # Not a warning to be noted and worked around: every downstream timing
        # would describe a CPU run that the real job would never do.
        report["error"] = "CUDA unavailable — driver/runtime mismatch, do not submit the campaign"
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))

    if args.oracle == "enformer":
        from enformer_pytorch import Enformer

        model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
        context = 196608
        channels_first = False
    else:
        from borzoi_pytorch import Borzoi

        model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
        context = 524288
        channels_first = True

    model = model.to("cuda").eval()

    one_hot = torch.zeros(context, 4)
    one_hot[torch.arange(context), torch.randint(0, 4, (context,))] = 1.0
    batch = one_hot.transpose(0, 1).unsqueeze(0) if channels_first else one_hot.unsqueeze(0)
    batch = batch.to("cuda")

    timings = []
    with torch.no_grad():
        for rep in range(args.reps + 1):
            torch.cuda.synchronize()
            start = time.time()
            out = model(batch)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            # The first pass carries cuDNN autotuning and lazy init, so it is
            # recorded but excluded from the median.
            timings.append(elapsed)
            if rep == 0:
                report["warmup_s"] = round(elapsed, 3)

    steady = sorted(timings[1:])
    report["median_s"] = round(steady[len(steady) // 2], 3)
    report["all_s"] = [round(t, 3) for t in timings]
    report["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    tensor = out["human"] if isinstance(out, dict) else out
    report["output_shape"] = list(tensor.shape)

    # What the campaign actually needs to know, in its own units.
    report["seconds_per_1000_seqs"] = round(report["median_s"] * 1000, 1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
