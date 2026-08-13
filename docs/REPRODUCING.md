# Reproducing this Pipeline on Different Diseases & Datasets

This project's pipeline (locus retrieval → data construction → generation → oracle scoring) is not MS-specific by architecture; it is MS-specific by *configuration*. 

You can run this pipeline for **any disease or cell type** using either **Nextflow (recommended)** or **Docker / Python CLI**.

---

## 1. Quick Execution with Nextflow (Zero-Config Mode)

Nextflow can fetch missing GWAS risk loci from the EBI Catalog and peak files from NCBI GEO on the fly:

```bash
# Example: Run for Ulcerative Colitis
nextflow run dralperenuysal/ms-enhancer \
  -profile docker \
  --suffix "uc" \
  --gwas_id "EFO_0000729" \
  --gwas_label "ulcerative colitis" \
  --gse "GSE282442" \
  --cell_type "epithelial_inflamed"

# Example: Run for Default Multiple Sclerosis
nextflow run dralperenuysal/ms-enhancer -profile docker --suffix "ms"
```

All models, candidate FASTAs, Enformer evaluations, and HTML execution reports are automatically placed in `results_${suffix}/`.

---

## 2. Execution with Custom YAML Config

If you have complex multi-GSE series or custom cell-type mapping rules:

1. Copy the example configuration:
   ```bash
   cp configs/data_config.example.yaml configs/data_config_custom.yaml
   ```
2. Run via Nextflow:
   ```bash
   nextflow run dralperenuysal/ms-enhancer \
     -profile docker \
     --data_config configs/data_config_custom.yaml \
     --suffix "custom" \
     --cell_type "my_target_cell"
   ```

---

## 3. Manual Docker Execution

If running step-by-step with the pre-built Docker image on a standard Linux host:

```bash
docker run --gpus all --rm -it \
  --user $(id -u):$(id -g) \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  dralperenuysal/ms-enhancer:latest python scripts/build_dataset.py \
    --gwas_id EFO_0000729 \
    --gwas_label "ulcerative colitis" \
    --gse GSE282442 \
    --cell_type epithelial_inflamed \
    --suffix uc
```

Mount `configs/` and `data/` so outputs persist on the host with your user's permissions (`--user $(id -u):$(id -g)`).

---

## 4. Running on Cloud GPU Platforms (Vast.ai, RunPod, Lambda Labs)

Cloud instances (like Vast.ai or RunPod) are **already running inside an isolated Docker container** with native GPU drivers and CUDA. Because Docker-in-Docker is not present by default, running with `-profile docker` will fail with `docker: command not found`.

On these instances, run Nextflow **natively** using the host's Conda environment and GPU:

```bash
# 1. Clone repo and create environment
git clone https://github.com/dralperenuysal/ms-enhancer.git
cd ms-enhancer
conda env create -f environment.yml
conda activate ms_enhancer

# 2. Prepare the reference genome (hg38)
mkdir -p data
wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O data/hg38.fa.gz
gunzip data/hg38.fa.gz

# 3. Run Nextflow natively (using host GPU)
nextflow run main.nf --suffix "ms"

# Or for a different disease:
nextflow run main.nf \
  --suffix "uc" \
  --gwas_id "EFO_0000729" \
  --gwas_label "ulcerative colitis" \
  --gse "GSE282442" \
  --cell_type "epithelial_inflamed"
```

---

## 5. Pipeline Outputs & Deliverables

Every entry point logs to stdout and saves structured deliverables under `results_${suffix}/` (or mounted volumes):

| Stage | Process / Script | File Outputs |
|---|---|---|
| **Data** | `scripts/build_dataset.py` | `data/bed/*.bed`, `data/fasta/*.fasta`, `data/processed/*.pt` |
| **Model** | `train.py` | `models/generator/transformer_best.pt`, `..._last.pt` |
| **Generation** | `generate.py` | `candidates/synthetic_candidates_${suffix}.fasta` + metadata CSV |
| **Oracle** | `evaluate.py` | `evaluation/evaluation_results_${suffix}.json` |
| **Selection** | `scripts/select_candidates.py` | `selected/top_selected_${suffix}_${cell_type}.fasta` |
| **Reports** | Nextflow runtime info | `pipeline_info/execution_report.html`, `timeline.html`, `pipeline_dag.html` |

---

## 6. Setting the GWAS Risk-Locus Trait

`gwas.trait_id` / `gwas.trait_label` (`src/data_processing/gwas_loci.py`, `MSGWASLoci`) are the only disease-identifying fields for locus retrieval:

1. Search your trait at https://www.ebi.ac.uk/gwas/; the MONDO/EFO id is in the result URL.
2. `trait_label` must match the GWAS Catalog trait name **exactly**; a mismatch returns HTTP 404 from the API.
3. `gwas.genome_build` must be one of `reference_genome.equivalent_builds` (GRCh38 / hg38; this pipeline does not implement liftover).

---

## 7. GEO Dataset Verification

Each entry under `geo_datasets.verified_datasets` is checked at runtime by `GEODownloader.verify_dataset()` against live GEO SOFT metadata:

| `expected` field | Checked against |
|---|---|
| `organism` | every sample's `!Sample_organism_ch1` |
| `library_strategy` | every sample's `!Sample_library_strategy` |
| `n_samples` | number of GSM samples in the series |
| (implicit) | assembly parsed from `!Sample_data_processing` must be in `reference_genome.equivalent_builds` |
