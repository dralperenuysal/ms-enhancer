# MS-ENHANCER-GEN

A computational research and benchmarking pipeline that generates candidate cell-type-specific cis-regulatory DNA sequences for multiple sclerosis (MS) risk loci, evaluates candidates with genomic deep-learning oracles (Enformer, cross-checked against Borzoi), and systematically audits what these oracles reward via in-silico genetic interventions.

> **Scope Note:** This is a research/prototype project, not a clinical tool. No claims of therapeutic or diagnostic validity appear anywhere in this repository.

> **Reproducibility Note:** We independently re-ran this entire pipeline end to end (Nextflow + Docker, GPU) after publication to validate it. The underlying GEO accessions (GSE202087, GSE307262) are live deposits, not a frozen snapshot, so a fresh run pulls whatever the depositors currently host; GSE307262 in particular is preprint-linked and may still be revised. Our re-run's window counts differed from the manuscript's by under 0.1%, and every qualitative conclusion (CpG causality and its host-locus-dependent sign, the absence of a motif-density effect, and the between-locus heterogeneity itself) reproduced at comparable significance and effect size. We did not update the manuscript's reported numbers to match this re-run: exact point estimates from live third-party data are expected to drift slightly between any two runs (including a reviewer's), and the biological conclusions are what the numbers are there to support. See `docs/REPRODUCING.md` for details.

---

## Quickstart (Nextflow & Docker)

The fastest and most reproducible way to run the entire pipeline on any GPU-enabled machine (no local dependency setup required):

```bash
# 1. Install Nextflow (requires Java 11+)
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# 2. Run the full MS pipeline (automatically pulls the pre-built Docker Hub image;
#    the hg38 reference genome is fetched and cached in data/ on first run)
nextflow run dralperenuysal/ms-enhancer -profile docker --suffix "ms"
```

All outputs, models, Enformer scores, and HTML execution reports are automatically placed in `results_ms/`.

> **Cloud GPU Instances (Vast.ai, RunPod, Lambda Labs):** Since cloud instances are already isolated containers without nested Docker, run Nextflow natively: `nextflow run main.nf --suffix "ms"`. See [**`docs/REPRODUCING.md`**](docs/REPRODUCING.md#4-running-on-cloud-gpu-platforms-vastai-runpod-lambda-labs) for details.

To run this pipeline on a **different disease** (e.g. Ulcerative Colitis, Alzheimer's) or custom GEO dataset, see [**`docs/REPRODUCING.md`**](docs/REPRODUCING.md).

> **No GEO access yet, or just want to check the pipeline runs?** `data/example_synthetic/` bundles a small synthetic (non-biological) peak dataset in the exact same `narrowPeak` + `peak_manifest.json` format GEO downloads produce, so you can smoke-test the full pipeline offline: `PYTHONPATH=. python scripts/build_dataset.py --manifest data/example_synthetic/peak_manifest.json`. (`PYTHONPATH=.` is only needed for local Python runs because `src/` is not a pip-installed package; the Docker/Nextflow path sets `PYTHONPATH=/app` itself.) See [`docs/REPRODUCING.md` §10](docs/REPRODUCING.md#10-offline-smoke-test-with-bundled-synthetic-data-no-geo-access-required).

### Kendi Verinizi Kullanmak

Kendi peak verinizi (GEO dışından, kendi ATAC-seq/ChIP-seq deneyinizden vb.) pipeline'a vermek isterseniz, `--manifest` ile `data/example_synthetic/` içindeki örnekle **aynı formatta** iki şey sağlamanız yeterli:

1. **Peak dosyaları** — her örnek (sample) için, tab-ayrılmış (TSV), başlıksız (headerless) bir `narrowPeak` dosyası, şu 10 sütunla (bkz. `data/example_synthetic/peaks/*.narrowPeak` örnekleri):

   | # | Sütun | Açıklama |
   |---|---|---|
   | 1 | `chrom` | Kromozom, örn. `chr1` (hg38 koordinatları) |
   | 2 | `start` | Peak başlangıcı (0-tabanlı) |
   | 3 | `end` | Peak bitişi |
   | 4 | `name` | Peak kimliği (serbest metin) |
   | 5 | `score` | 0-1000 arası tam sayı |
   | 6 | `strand` | Genelde `.` |
   | 7 | `signal_value` | Sinyal şiddeti (ondalık) |
   | 8 | `p_value` | -log10(p) |
   | 9 | `q_value` | -log10(q) |
   | 10 | `summit_offset` | Zirve (summit) konumunun peak başlangıcına göre ofseti |

   (HOMER formatı, başlık satırlı TSV olarak da desteklenir; bkz. `src/data_processing/bed_processor.py`.)

2. **`peak_manifest.json`** — her peak dosyasını bir hücre tipiyle eşleyen liste, her kayıtta şu alanlar:

   ```json
   {
     "accession": "MY_DATASET",
     "gsm": "SAMPLE_1",
     "title": "İsteğe bağlı açıklama",
     "cell_type": "CD4_T_cell",
     "url": "",
     "peak_format": "narrowPeak",
     "local_path": "data/my_dataset/peaks/SAMPLE_1.narrowPeak"
   }
   ```

   `cell_type` alanı, `configs/model_config.yaml`'daki oracle track eşleştirmeleriyle (bkz. [`docs/REPRODUCING.md` §8](docs/REPRODUCING.md#8-pointing-the-oracles-at-a-new-cell-type-configsmodel_configyaml)) uyumlu olmalı; yeni bir hücre tipi kullanıyorsanız o eşleştirmeyi elle eklemeniz gerekir.

   Sonra: `PYTHONPATH=. python scripts/build_dataset.py --manifest path/to/peak_manifest.json --suffix mydata` — `build_dataset.py` manifest'i bulduğu an GEO'ya hiç gitmeden doğrudan kullanır (bkz. `scripts/build_dataset.py`'deki manifest kısayolu).

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
main.nf         Nextflow DSL2 automated multi-disease workflow (data -> train -> generate -> score -> select -> audit)
modules/        EVALUATE_ORACLE, included multiple times (main scoring + each audit rescoring)
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
