# Reproducing this pipeline on a different disease / dataset

This project's pipeline (locus retrieval → data construction → generation →
oracle scoring) is not MS-specific by architecture; it is MS-specific by
*configuration*. Everything below is driven by `configs/data_config.yaml`;
no disease, GEO accession, or cell type is hardcoded in `src/`.

## 1. Build and run the container

```bash
docker build -t ms-enhancer .
docker run --gpus all -it \
  --user $(id -u):$(id -g) \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  ms-enhancer python scripts/build_dataset.py
```

Mount `configs/` (so your edits are visible without rebuilding the image) and
`data/` (so downloaded/generated data persists on the host). Pass `--user $(id -u):$(id -g)`
so newly written files in mounted volumes match your host user permissions rather than `root`.
Drop `--gpus all` if you only need CPU-only steps (locus fetch, BED processing).

### What you'll see

Every entry point logs to stdout (visible directly in the terminal running
`docker run`) and duplicates the same log to a file under `logs/`; mount
that directory too (`-v $(pwd)/logs:/app/logs`) if you want it on the host.
The container produces no other console output (no server, no interactive
prompt); each command runs once and exits.

| Command | stdout/log | File output (under mounted `data/`, `models/`, `logs/`) |
|---|---|---|
| `scripts/build_dataset.py` | `logs/build_dataset.log` (locus/GEO fetch progress, sample counts per cell type) | `data/bed/*.bed`, `data/fasta/*.fasta`, `data/processed/*`: the windows the rest of the pipeline reads |
| `train.py` | `logs/train_<model_type>.log` (per-epoch loss) | `models/generator/<model_type>_best.pt`, `..._last.pt`: checkpoints |
| `generate.py` | `logs/generate.log` (one line confirming sequence count and output paths) | `<out_fasta>` (FASTA of generated sequences) + `<out_fasta>_metadata.csv` (host locus per sequence) |
| `evaluate.py` | `logs/evaluate.log` | `<output_report>` (JSON, default `logs/evaluation_results.json`): per-sequence oracle scores |
| `scripts/select_candidates.py` | `logs/select_candidates.log` | top-K candidate FASTA + `..._metadata.csv` |
| other `scripts/*.py` (ablation, occlusion, CpG swap, locus survey, MPRA scoring) | `logs/<script_name>.log` | a JSON or CSV report path given via `--output`/positional arg; see each script's `--help` |

Nothing renders to a GUI or opens a browser; every result is a file. To
inspect them, either read them on the host through the mounted volume, or
`docker cp` / `docker exec` into a running container.

## 2. Copy the template config

```bash
cp configs/data_config.example.yaml configs/data_config.yaml
```

Do not start from the real `configs/data_config.yaml` if it is present in your
checkout: it holds the verified MS accessions and MS trait id; copying it
risks silently mixing MS data into a different disease's run.

## 3. Set the risk-locus trait

`gwas.trait_id` / `gwas.trait_label` (`src/data_processing/gwas_loci.py`,
`MSGWASLoci`) are the only disease-identifying fields for locus retrieval;
nothing else needs to change for this step.

1. Search your trait at https://www.ebi.ac.uk/gwas/; the MONDO/EFO id is in
   the result URL.
2. `trait_label` must match the GWAS Catalog trait name **exactly**; a
   mismatch returns HTTP 404 from the API, not a partial result.
3. `gwas.genome_build` must be one of `reference_genome.equivalent_builds`
   (`MSGWASLoci.__init__` raises `ValueError` otherwise; this pipeline does
   not implement liftover, so a GRCh37-only trait cannot be mixed in as-is).

## 4. Add and verify a GEO dataset

Each entry under `geo_datasets.verified_datasets` is checked at runtime by
`GEODownloader.verify_dataset()` (`src/data_processing/geo_downloader.py`)
against live GEO SOFT metadata. It compares your `expected` block against
what GEO actually reports and raises `DatasetVerificationError` on any
mismatch:

| `expected` field    | Checked against                                          |
|----------------------|-----------------------------------------------------------|
| `organism`           | every sample's `!Sample_organism_ch1`                    |
| `library_strategy`   | every sample's `!Sample_library_strategy`                |
| `n_samples`          | number of GSM samples in the series                       |
| (implicit)           | assembly parsed from `!Sample_data_processing` must be in `reference_genome.equivalent_builds` |

Before adding an accession, fetch its metadata and check these fields
yourself; the `geo_datasets.verification.method` field in the config records
how to do this (`acc.cgi?acc=<GSE>&targ=gsm&form=text&view=brief`). This
project's own `verification.rejected` list (in the shipped `data_config.yaml`,
not the template) exists precisely because some accessions that look relevant
turn out to be the wrong organism, assay, or have no usable peak calls;
check before you trust a GSE number.

## 5. Map GEO samples to cell types

`geo_datasets.*.cell_type_map` maps each series' GEO `cell type:`
characteristic to the canonical labels used in `condition_encoding.cell_types`
(`GEODownloader.select_samples`). If a series has no `cell type:`
characteristic, use `sample_title_regex` + `cell_type_default` instead (see
the commented-out example in `configs/data_config.example.yaml`).

A sample that matches no map entry and no title regex is silently skipped
(logged, not an error); after running `select_samples`, check the log line
`Selected N samples from <GSE>: {...}` to confirm every cell type you expect
actually got samples, not just that the run didn't crash.

## 6. Run the pipeline

Same entry points as the MS pipeline, unchanged; see the root
[`README.md`](../README.md#pipeline). None of `train.py`, `generate.py`,
`evaluate.py`, or the `scripts/*.py` steps read the disease or cell types from
anywhere except the config you just edited.
