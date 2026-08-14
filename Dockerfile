FROM continuumio/miniconda3:24.9.2-0

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "ms_enhancer", "/bin/bash", "-c"]

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    PATH=/opt/conda/envs/ms_enhancer/bin:$PATH

COPY . .

# No ENTRYPOINT wrapper: PATH above already points at the ms_enhancer env's
# bin/, so `python`, `pip`, etc. work directly. A `conda run -n ms_enhancer`
# ENTRYPOINT was tried and rejected -- Nextflow's docker executor runs its own
# command as the container's CMD, and nesting that inside `conda run` reset
# PATH and broke env activation (`python: command not found` even though the
# image PATH was correct), so plain CMD is both simpler and the only form that
# works under Nextflow.
CMD ["bash"]
