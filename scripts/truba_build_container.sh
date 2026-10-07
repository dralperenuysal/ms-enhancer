#!/bin/bash
# Build the Apptainer container for oracle scoring on TRUBA.
#
# Conda/pip must never be installed onto /arf directly (site rule: hundreds of
# thousands of small files cripple the shared Lustre filesystem). The approved
# route is a sandbox that is packed into a single .sif and then deleted, which
# is what this script does end to end.
#
# Driver note: akya-cuda (V100) and barbun-cuda (P100) report driver 565.57.01,
# which tops out around CUDA 12.7. The NGC pytorch images in
# /arf/sw/containers/truba-ai ship CUDA 13.x and would fail at runtime, so we
# install a cu124 wheel from PyPI instead.
#
# The base image already carries Python 3.11 and pip at /opt/conda, so there is
# no micromamba step: an earlier version installed one and hit a libmamba
# "Cannot find a valid extracted directory cache" failure, which is the package
# cache misbehaving on Lustre. pip alone avoids that entirely.
#
# Usage (from a debug-queue allocation, not the login node):
#   srun -p debug --time=03:00:00 --ntasks=1 --cpus-per-task=4 \
#       bash scripts/truba_build_container.sh

set -euo pipefail

ROOT=/arf/scratch/$USER/ms-enhancer
BASE=/arf/sw/containers/miniconda3/miniconda3-container.sif
SIF="$ROOT/ms-enhancer.sif"

cd "$ROOT"

command -v apptainer >/dev/null || { echo "apptainer missing on $(hostname)" >&2; exit 1; }

[ -f base.sif ] || cp "$BASE" base.sif

rm -rf sandbox
apptainer build --sandbox sandbox base.sif

# pip's build and cache directories go to the node-local /tmp rather than the
# sandbox's Lustre path: the wheels are large, the writes are many and small,
# and nothing here needs to survive the build.
apptainer exec --writable --fakeroot sandbox bash -c '
  set -eux
  export TMPDIR=/tmp/pipbuild
  mkdir -p "$TMPDIR"
  PIP="/opt/conda/bin/pip install --no-cache-dir"
  $PIP --upgrade pip
  $PIP torch --index-url https://download.pytorch.org/whl/cu124
  $PIP numpy pandas pyyaml biopython pyfaidx pyjaspar scipy
  $PIP enformer-pytorch borzoi-pytorch
'

rm -f "$SIF"
apptainer build "$SIF" sandbox
rm -rf sandbox base.sif

# The build proves itself: a .sif with a missing package fails silently in the
# queue hours later, so verify imports here rather than in the first real job.
apptainer exec --bind /arf "$SIF" python3 -c "
import torch, numpy, pandas, yaml, Bio, pyfaidx, pyjaspar, scipy
from enformer_pytorch import Enformer
from borzoi_pytorch import Borzoi
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
print('imports OK')
"
echo "Built $SIF"
