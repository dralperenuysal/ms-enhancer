FROM continuumio/miniconda3:24.9.2-0

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "ms_enhancer", "/bin/bash", "-c"]

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/huggingface \
    PATH=/opt/conda/envs/ms_enhancer/bin:$PATH

# Nextflow's docker profile runs the container as the host's UID:GID (not root,
# not a user in /etc/passwd), so $HOME resolves to `/` and any library that
# defaults to `~/.cache` (huggingface_hub included, despite HF_HOME above --
# some of its internals still probe $HOME) fails with a permission error on
# `/.cache`. Give every UID a writable home instead of relying on one landing
# in /etc/passwd.
RUN mkdir -p /tmp/home && chmod 777 /tmp/home
ENV HOME=/tmp/home

COPY . .

# No ENTRYPOINT wrapper: PATH above already points at the ms_enhancer env's
# bin/, so `python`, `pip`, etc. work directly. A `conda run -n ms_enhancer`
# ENTRYPOINT was tried and rejected -- Nextflow's docker executor runs its own
# command as the container's CMD, and nesting that inside `conda run` reset
# PATH and broke env activation (`python: command not found` even though the
# image PATH was correct), so plain CMD is both simpler and the only form that
# works under Nextflow.
CMD ["bash"]
