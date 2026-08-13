FROM continuumio/miniconda3:24.9.2-0

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "ms_enhancer", "/bin/bash", "-c"]

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ms_enhancer"]
CMD ["bash"]
