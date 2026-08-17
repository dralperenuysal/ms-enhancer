# Reproducing this Pipeline on Different Diseases & Datasets

This project's pipeline (locus retrieval → data construction → generation → oracle scoring) is not MS-specific by architecture; it is MS-specific by *configuration*. 

You can run this pipeline for **any disease or cell type** using either **Nextflow (recommended)** or **Docker / Python CLI**.

---

## 1. Quick Execution with Nextflow (Zero-Config Mode)

Nextflow fetches everything on the fly: the hg38 reference genome (cached in `data/` after the first run), GWAS risk loci from the EBI Catalog, and peak files from NCBI GEO.

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

If running step-by-step with the pre-built Docker image on a standard Linux host. `build_dataset.py` is CPU-only and needs no GPU; add `--gpus all` only for `train.py` / `evaluate.py --oracle enformer|borzoi`:

```bash
docker run --rm -it \
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

## 3b. HPC Clusters without Root Docker (Singularity / SLURM)

The `singularity` and `slurm` profiles run against a local `ms-enhancer.sif` image, which must be built once from the Docker Hub image (most HPC login nodes have Singularity/Apptainer but not Docker):

```bash
singularity build ms-enhancer.sif docker://dralperenuysal/ms-enhancer:latest

# Single node with Singularity
nextflow run main.nf -profile singularity --suffix "ms"

# SLURM cluster (submits TRAIN_MODEL/GENERATE_SEQUENCES/EVALUATE_ORACLE with --gres=gpu:1)
nextflow run main.nf -profile slurm --suffix "ms"
```

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

# 2. Run Nextflow natively (using host GPU; hg38 is fetched and cached in data/ automatically)
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
| **Audit** | `scripts/occlusion_scan.py`, `motif_ablation.py`, `cpg_swap.py`, `locus_survey.py`, `compare_selected_grammar.py` | `audit/{occlusion,motif_ablation,cpg_swap,locus_survey,grammar}/*`, each intervention rescored via `evaluate.py` into `evaluation/evaluation_results_${suffix}_*.json` |
| **Reports** | Nextflow runtime info | `pipeline_info/execution_report.html`, `timeline.html`, `pipeline_dag.html` |

The audit stage runs by default (`params.run_audit = true`), so a plain `nextflow run ... --suffix uc` reproduces the full protocol end to end: candidate selection *and* the causal interventions that test what the oracle is rewarding, each with its own rescored MSSI report. Pass `--run_audit false` to stop after candidate selection (e.g. for a quick data-construction smoketest on a new disease/GEO dataset without paying for a full GPU run).

`scripts/mpra_scoring_set.py` is intentionally not wired into `main.nf`: it needs externally measured MPRA activity data (`--measurements`) that will not exist for most new diseases. Run it manually once you have a matching MPRA dataset; see its module docstring for usage.

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

---

## 8. Pointing the Oracles at a New Cell Type (`configs/model_config.yaml`)

Sections 1-7 make locus retrieval and data construction disease-agnostic. The oracle-scoring step is **not** automatic: Enformer and Borzoi don't know what "CD4_T_cell" or "epithelial_inflamed" means, they only expose numbered output tracks. Every cell type used with `--oracle enformer` or `--oracle borzoi` needs its track indices curated by hand in `configs/model_config.yaml`, in up to four places:

1. **`evaluation.enformer.target_tracks_by_cell_type`** — pick the DNase/ATAC/ChIP track indices that match your cell type from Enformer's human targets file (`targets_file_url` in the same config, `calico/basenji targets_human.txt`, 5313 tracks). Search it by cell-type name in the track description column.
2. **`evaluation.borzoi.target_tracks_by_cell_type`** — repeat independently against Borzoi's targets file (`precomputed/targets.txt` shipped with `borzoi-pytorch`, 7611 tracks). Track indices are **not shared** between the two oracles; do not reuse Enformer's numbers.
3. **`cvae.condition_dim` / `genomic_transformer.condition_dim`** — must equal (number of cell types, one-hot) + (number of continuous conditioning features). Update if your cell-type count differs from the shipped 3 (CD4_T_cell, B_cell, microglia).
4. **`evaluation.motif_analysis.tfs`** (optional, only needed for `scripts/motif_ablation.py`) — add the lineage-defining transcription factors for your new cell type, resolved by name against `jaspar_release`/`jaspar_collection`. Every name listed must exist in that JASPAR release or motif analysis fails at startup.

If your exact cell type has no track in a given oracle (e.g. Enformer has no microglia track — see the proxy-track caveat at `configs/model_config.yaml:112-117`), use the nearest available lineage as an explicit proxy and document it as such; do not report the resulting score as a direct measurement of the missing cell type.

There is no CLI flag or automated lookup for this step — it is manual curation, done once per (oracle, cell type) pair, the same way the CD4_T_cell/B_cell/microglia tracks shipped with this repo were originally selected.

---

## 9. Reproducibility of the MS Manuscript's Numbers

We re-ran the full MS pipeline (`--suffix ms`, `--generator markov`, the generator the manuscript's own selection/audit results were built on, see the comment above `params.generator` in `main.nf`) end to end via `nextflow run ... -profile docker` on a fresh GPU VM, independently of the original analysis, to validate that this repository actually reproduces the manuscript. It did, with one caveat worth stating plainly:

**GSE202087 and GSE307262 are live GEO deposits, not a frozen snapshot bundled with this repo.** A fresh `nextflow run` re-downloads whatever GEO currently hosts under those accessions. GSE307262 is explicitly preprint-linked (see the manuscript's reference list); depositors can and do revise preprint-associated deposits after publication without changing the accession number. Our re-run's cell-type-exclusive window count came out within 0.1% of the manuscript's reported figure (a handful of windows out of ~10,800), small enough to be consistent with exactly this kind of upstream revision, not with a configuration difference (the same accessions, same sample counts per cell type, and same dedup rules in `configs/data_config.yaml` were used both times).

Every qualitative conclusion reproduced despite that drift: CpG content's causal effect and its host-locus-dependent sign, the null result for motif-density ablation, and the between-locus heterogeneity in Section 2.7 (Cochran's $Q$, $I^2$, and the direction/rate of individually significant loci) all came out at comparable magnitude and significance in the independent re-run. **We did not update the manuscript's reported point estimates to match this re-run**, and do not intend to update them after future re-runs either: exact figures computed against live third-party data will drift slightly between any two pulls, including a reviewer's own re-run, and chasing bit-for-bit numeric agreement against a moving upstream target is not a meaningful reproducibility bar. The bar that matters, and the one this re-run was checked against, is whether the same biological conclusions come out, and they did.

If you re-run this pipeline and get numbers that differ from the manuscript by more than roughly this margin, or whose *qualitative* conclusions (sign, significance, direction) differ, that is worth investigating as a real discrepancy; small drift in the third or fourth significant figure of a point estimate, on its own, is not.

---

## 10. Offline Smoke Test with Bundled Synthetic Data (No GEO Access Required)

`data/example_synthetic/` ships in this repo (it's the one exception to `data/` being
git-ignored): a `peak_manifest.json` plus a handful of `narrowPeak` files at 40 peaks
each, for 3 fake samples (`CD4_T_cell`, `B_cell`, `microglia`). Peak coordinates are
drawn from the real MS GWAS risk loci in `data/bed/ms_gwas_loci_hg38.bed` so windows
land somewhere biologically meaningful, but the scores/signal/p-values are fabricated
(`scripts/make_synthetic_example_data.py`, seed 42). **This is not real ATAC-seq data
and produces no valid scientific conclusions** — it exists purely so you can verify the
pipeline mechanics (window building, encoding, generation, oracle scoring, audits) run
end to end on a machine that can't or doesn't want to hit live GEO yet, before pointing
it at a real accession.

```bash
# Regenerate the bundled files (optional, they're already committed):
PYTHONPATH=. python scripts/make_synthetic_example_data.py

# Run the pipeline against them directly, skipping GEO entirely:
PYTHONPATH=. python scripts/build_dataset.py --manifest data/example_synthetic/peak_manifest.json --suffix synthtest

# Or via Nextflow, same idea (PYTHONPATH handled by the Docker image, not needed):
nextflow run main.nf -profile docker --suffix synthtest \
  --manifest data/example_synthetic/peak_manifest.json
```

`PYTHONPATH=.` is only needed for local Python runs: `src/` is imported as a
package but there is no `pip install -e .` step, so the repo root must be on
`sys.path` (the Docker image sets `PYTHONPATH=/app` for the same reason).
