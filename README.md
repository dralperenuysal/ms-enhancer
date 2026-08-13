# MS-ENHANCER-GEN

A computational research and benchmarking pipeline that generates candidate cell-type-specific cis-regulatory DNA sequences for multiple sclerosis (MS) risk loci, evaluates candidates with genomic deep-learning oracles (Enformer, cross-checked against Borzoi), and systematically audits what these oracles reward via in-silico genetic interventions.

> **Scope Note:** This is a research/prototype project, not a clinical tool. No claims of therapeutic or diagnostic validity appear anywhere in this repository.

---

## Quickstart (Nextflow & Docker)

The fastest and most reproducible way to run the entire pipeline on any GPU-enabled machine (no local dependency setup required):

```bash
# 1. Install Nextflow (requires Java 11+)
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# 2. Run the full MS pipeline (automatically pulls the pre-built Docker Hub image)
nextflow run dralperenuysal/ms-enhancer -profile docker --suffix "ms"
```

All outputs, models, Enformer scores, and HTML execution reports are automatically placed in `results_ms/`.

> **Cloud GPU Instances (Vast.ai, RunPod, Lambda Labs):** Since cloud instances are already isolated containers without nested Docker, run Nextflow natively: `nextflow run main.nf --suffix "ms"`. See [**`docs/REPRODUCING.md`**](docs/REPRODUCING.md#4-running-on-cloud-gpu-platforms-vastai-runpod-lambda-labs) for details.

To run this pipeline on a **different disease** (e.g. Ulcerative Colitis, Alzheimer's) or custom GEO dataset, see [**`docs/REPRODUCING.md`**](docs/REPRODUCING.md).

---

## Local Setup (Conda / Pip)

```bash
conda env create -f environment.yml
conda activate ms_enhancer
# or: pip install -r requirements.txt
```

* **Hardware:** Requires a CUDA GPU with ≥24 GB VRAM for Enformer/Borzoi inference and Transformer training.
* **Configuration:** All paths, GEO accessions, oracle track indices, and hyperparameters live in `configs/*.yaml`; nothing is hardcoded in source.
* **Pre-built Container Image:** `docker pull dralperenuysal/ms-enhancer:latest`

---

## Step-by-Step CLI Pipeline

If running commands individually:

```bash
# 1. Build 1000 bp windows and one-hot encoded tensors from GWAS/GEO data
python scripts/build_dataset.py --config configs/data_config.yaml

# 2. Train the genomic transformer generator
python train.py --config configs/model_config.yaml --model_type transformer

# 3. Generate candidate sequences (or fit order-6 Markov baseline)
python generate.py --generator checkpoint --checkpoint models/generator/transformer_best.pt --cell_type CD4_T_cell

# 4. In-silico evaluation with DeepMind Enformer
python evaluate.py --oracle enformer --input_fasta data/fasta/synthetic_ms_enhancers.fasta

# 5. Rank and select top candidates
python scripts/select_candidates.py --report logs/evaluation_results.json --fasta data/fasta/synthetic_ms_enhancers.fasta --top_k 50 --out_fasta data/fasta/top_selected_candidates.fasta

# 6. In-silico interventions & mechanistic auditing
python scripts/compare_selected_grammar.py
python scripts/occlusion_scan.py
python scripts/motif_ablation.py
python scripts/cpg_swap.py
python scripts/locus_survey.py
python scripts/mpra_scoring_set.py
```

Each script's `--help` lists its arguments; defaults match the paths and parameters in `configs/model_config.yaml` and `configs/data_config.yaml`.

---

## Repository Layout

```text
configs/        data_config.yaml, model_config.yaml (all paths/accessions/hyperparameters)
src/            data_processing/, models/ (cVAE, transformer), evaluation/ (oracles, motif,
                Markov baseline, sequence realism), utils/
scripts/        build_dataset.py, candidate selection, interventions, survey, MPRA scoring
tests/          pytest suite, one file per src/ module (185 tests, 100% pass)
main.nf         Nextflow DSL2 automated multi-disease workflow
nextflow.config Nextflow profiles (docker, slurm, singularity, awsbatch)
Dockerfile      Reproducible container definition (dralperenuysal/ms-enhancer:latest)
```

`data/`, `models/`, `logs/`, and `results_*/` are git-ignored: they hold reference genomes, downloaded/generated data, model checkpoints, run logs, and pipeline deliverables.

---

## Testing

```bash
# In local conda env:
pytest

# Or via Docker:
docker run --rm dralperenuysal/ms-enhancer:latest pytest
```
All 185 unit tests are seeded (`seed=42`) for deterministic reproducibility across platforms.
