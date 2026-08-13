#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * =========================================================================================
 *  MS-ENHANCER-GEN: Nextflow Workflow
 * =========================================================================================
 *  A reproducible benchmarking and auditing pipeline for evaluating conditional
 *  generative DNA design and genomic oracle biases across diverse human traits.
 * =========================================================================================
 */

// Pipeline Parameters
params.suffix           = "ms"
params.outdir           = "${launchDir}/results_${params.suffix}"
params.data_config      = "${projectDir}/configs/data_config.yaml"
params.model_config     = "${projectDir}/configs/model_config.yaml"
params.manifest         = null
params.genome_fasta     = "${projectDir}/data/hg38.fa"

// Dynamic Disease & Locus Parameters
params.gwas_id          = null
params.gwas_label       = null
params.gse              = null
params.cell_type        = "CD4_T_cell"

// Model & Oracle Parameters
params.model_type       = "transformer"  // 'transformer' or 'cvae'
params.num_samples      = 1000
params.oracle           = "enformer"     // 'enformer', 'borzoi', 'motif', 'realism'
params.top_k            = 50
params.epochs           = 100
params.batch_size       = 64
params.seed             = 42

// Process 1: Build Genomic Windows & Tensors
process BUILD_DATASET {
    tag "Building dataset (suffix: ${params.suffix})"
    publishDir "${params.outdir}/data", mode: 'copy'

    input:
    path data_cfg
    path genome_fa

    output:
    path "processed/processed_dataset.pt", emit: processed_pt
    path "fasta/ms_windows_1000bp.fasta",  emit: windows_fasta
    path "fasta/ms_windows_metadata.csv",  emit: windows_meta
    path "bed/ms_windows_1000bp.bed",      emit: windows_bed

    script:
    def gwas_args = params.gwas_id ? "--gwas_id '${params.gwas_id}'" : ""
    def gwas_lbl  = params.gwas_label ? "--gwas_label '${params.gwas_label}'" : ""
    def gse_args  = params.gse ? "--gse '${params.gse}'" : ""
    def cell_arg  = params.cell_type ? "--cell_type '${params.cell_type}'" : ""
    def suffix_arg = "--suffix '${params.suffix}'"
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    mkdir -p data
    if [ -f "${genome_fa}" ]; then
        ln -s \$(readlink -f ${genome_fa}) data/hg38.fa
    fi

    python ${projectDir}/scripts/build_dataset.py \
        --config ${data_cfg} \
        ${suffix_arg} \
        ${gwas_args} \
        ${gwas_lbl} \
        ${gse_args} \
        ${cell_arg}

    mkdir -p processed fasta bed
    cp -r data/processed/* processed/
    cp -r data/fasta/* fasta/
    cp -r data/bed/* bed/
    """
}

// Process 2: Train Generative Model
process TRAIN_MODEL {
    tag "Training ${params.model_type} (${params.epochs} epochs)"
    publishDir "${params.outdir}/models", mode: 'copy'
    accelerator 1, type: 'nvidia-gpu'

    input:
    path model_cfg
    path dataset_pt

    output:
    path "generator/${params.model_type}_best.pt", emit: best_checkpoint
    path "generator/${params.model_type}_last.pt", emit: last_checkpoint
    path "logs/train_${params.model_type}.log",   emit: train_log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    mkdir -p models/generator logs data/processed
    ln -s \$(readlink -f ${dataset_pt}) data/processed/processed_dataset.pt

    python ${projectDir}/train.py \
        --config ${model_cfg} \
        --model_type ${params.model_type} \
        --data_path data/processed/processed_dataset.pt \
        --batch_size ${params.batch_size} \
        --epochs ${params.epochs} \
        --seed ${params.seed} \
        --amp

    mkdir -p generator
    cp models/generator/* generator/
    """
}

// Process 3: Generate Cell-Type-Specific Sequences
process GENERATE_SEQUENCES {
    tag "Sampling ${params.num_samples} sequences for ${params.cell_type}"
    publishDir "${params.outdir}/candidates", mode: 'copy'
    accelerator 1, type: 'nvidia-gpu'

    input:
    path checkpoint
    path model_cfg
    path dataset_pt
    path windows_fasta
    path windows_meta

    output:
    path "synthetic_candidates_${params.suffix}.fasta",      emit: candidates_fasta
    path "synthetic_candidates_${params.suffix}_metadata.csv", emit: candidates_meta
    path "logs/generate.log",                                  emit: gen_log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    mkdir -p data/fasta data/processed logs

    ln -s \$(readlink -f ${dataset_pt}) data/processed/processed_dataset.pt
    ln -s \$(readlink -f ${windows_fasta}) data/fasta/ms_windows_1000bp.fasta
    ln -s \$(readlink -f ${windows_meta}) data/fasta/ms_windows_metadata.csv

    python ${projectDir}/generate.py \
        --checkpoint ${checkpoint} \
        --config ${model_cfg} \
        --cell_type ${params.cell_type} \
        --dataset data/processed/processed_dataset.pt \
        --windows_fasta data/fasta/ms_windows_1000bp.fasta \
        --host_loci data/fasta/ms_windows_metadata.csv \
        --num_samples ${params.num_samples} \
        --out_fasta synthetic_candidates_${params.suffix}.fasta \
        --seed ${params.seed}
    """
}

// Process 4: Evaluate with In-Silico Oracle
process EVALUATE_ORACLE {
    tag "Scoring with ${params.oracle} oracle"
    publishDir "${params.outdir}/evaluation", mode: 'copy'
    accelerator 1, type: 'nvidia-gpu'

    input:
    path candidates_fasta
    path candidates_meta
    path model_cfg
    path genome_fa

    output:
    path "evaluation_results_${params.suffix}.json", emit: eval_report
    path "logs/evaluate.log",                         emit: eval_log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    mkdir -p logs data
    ln -s \$(readlink -f ${genome_fa}) data/hg38.fa

    python ${projectDir}/evaluate.py \
        --input_fasta ${candidates_fasta} \
        --metadata ${candidates_meta} \
        --oracle ${params.oracle} \
        --config ${model_cfg} \
        --reference_fasta data/hg38.fa \
        --output_report evaluation_results_${params.suffix}.json
    """
}

// Process 5: Select Top Candidates
process SELECT_CANDIDATES {
    tag "Selecting top ${params.top_k} candidates"
    publishDir "${params.outdir}/selected", mode: 'copy'

    input:
    path eval_report
    path candidates_fasta
    path candidates_meta

    output:
    path "top_selected_${params.suffix}_${params.cell_type}.fasta", emit: selected_fasta
    path "top_selected_${params.suffix}_${params.cell_type}_metadata.csv", optional: true
    path "logs/select_candidates.log", emit: select_log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    mkdir -p logs

    python ${projectDir}/scripts/select_candidates.py \
        --report ${eval_report} \
        --fasta ${candidates_fasta} \
        --metadata ${candidates_meta} \
        --top_k ${params.top_k} \
        --out_fasta top_selected_${params.suffix}_${params.cell_type}.fasta
    """
}

// Main Workflow Orchestration
workflow {
    log.info """\
    ================================================================================
    MS-ENHANCER-GEN : Nextflow Workflow
    ================================================================================
    Experiment Suffix : ${params.suffix}
    Output Directory  : ${params.outdir}
    Data Config       : ${params.data_config}
    Model Config      : ${params.model_config}
    GWAS Trait ID     : ${params.gwas_id ?: 'Default (Config-defined)'}
    GWAS Trait Label  : ${params.gwas_label ?: 'Default (Config-defined)'}
    GEO Accession     : ${params.gse ?: 'Default (Config-defined)'}
    Target Cell Type  : ${params.cell_type}
    Model Type        : ${params.model_type}
    Num Samples       : ${params.num_samples}
    Oracle Evaluator  : ${params.oracle}
    Top-K Selection   : ${params.top_k}
    ================================================================================
    """

    data_config_ch   = file(params.data_config)
    model_config_ch  = file(params.model_config)
    genome_fasta_ch  = file(params.genome_fasta)

    // Step 1: Build dataset
    BUILD_DATASET(data_config_ch, genome_fasta_ch)

    // Step 2: Train model
    TRAIN_MODEL(model_config_ch, BUILD_DATASET.out.processed_pt)

    // Step 3: Sample sequences
    GENERATE_SEQUENCES(
        TRAIN_MODEL.out.best_checkpoint,
        model_config_ch,
        BUILD_DATASET.out.processed_pt,
        BUILD_DATASET.out.windows_fasta,
        BUILD_DATASET.out.windows_meta
    )

    // Step 4: In-silico evaluation
    EVALUATE_ORACLE(
        GENERATE_SEQUENCES.out.candidates_fasta,
        GENERATE_SEQUENCES.out.candidates_meta,
        model_config_ch,
        genome_fasta_ch
    )

    // Step 5: Candidate selection
    SELECT_CANDIDATES(
        EVALUATE_ORACLE.out.eval_report,
        GENERATE_SEQUENCES.out.candidates_fasta,
        GENERATE_SEQUENCES.out.candidates_meta
    )
}
