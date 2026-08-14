#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// EVALUATE_ORACLE included once per call site: Nextflow forbids invoking the
// same process more than once from a single script, so the main scoring call
// and each audit-rescoring call get their own alias.
include { EVALUATE_ORACLE as EVALUATE_MAIN            } from './modules/evaluate_oracle.nf'
include { EVALUATE_ORACLE as EVALUATE_OCCLUSION       } from './modules/evaluate_oracle.nf'
include { EVALUATE_ORACLE as EVALUATE_MOTIF_ABLATION  } from './modules/evaluate_oracle.nf'
include { EVALUATE_ORACLE as EVALUATE_CPG_SWAP        } from './modules/evaluate_oracle.nf'
include { EVALUATE_ORACLE as EVALUATE_LOCUS_SURVEY    } from './modules/evaluate_oracle.nf'

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

// Mechanistic Auditing (in-silico interventions on the selected candidates)
params.run_audit        = true

// Process 0: Fetch Reference Genome (cached in data/, downloaded once)
process GENOME_PREP {
    tag "Preparing hg38 reference genome"
    storeDir "${projectDir}/data"

    output:
    path "hg38.fa", emit: fasta

    script:
    """
    if [ ! -f hg38.fa ]; then
        wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O hg38.fa.gz
        gunzip hg38.fa.gz
    fi
    """
}

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
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
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
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
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
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
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
    path "top_selected_${params.suffix}_${params.cell_type}_metadata.csv", emit: selected_meta
    path "logs/select_candidates.log", emit: select_log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs

    python ${projectDir}/scripts/select_candidates.py \
        --report ${eval_report} \
        --fasta ${candidates_fasta} \
        --metadata ${candidates_meta} \
        --top_k ${params.top_k} \
        --out_fasta top_selected_${params.suffix}_${params.cell_type}.fasta
    """
}

// ===========================================================================
// Mechanistic Auditing: in-silico interventions on the selected candidates,
// each rescored by EVALUATE_ORACLE so the causal effect on MSSI is a
// pipeline artifact, not a manual notebook step.
// ===========================================================================

// Process 6a: Occlusion Scan (localises the oracle's preference within the insert)
process AUDIT_OCCLUSION {
    tag "Occlusion scan (suffix: ${params.suffix})"
    publishDir "${params.outdir}/audit/occlusion", mode: 'copy'

    input:
    path eval_report
    path selected_fasta
    path selected_meta

    output:
    path "occluded_${params.suffix}.fasta",          emit: fasta
    path "occluded_${params.suffix}_metadata.csv",   emit: meta
    path "logs/occlusion_scan.log",                  emit: log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs
    python ${projectDir}/scripts/occlusion_scan.py \
        --score_report ${eval_report} \
        --input_fasta ${selected_fasta} \
        --metadata ${selected_meta} \
        --output_fasta occluded_${params.suffix}.fasta \
        --output_metadata occluded_${params.suffix}_metadata.csv \
        --seed ${params.seed} \
        2>&1 | tee logs/occlusion_scan.log
    """
}

// Process 6b: Motif Ablation (tests whether a specific factor's sites carry the preference)
process AUDIT_MOTIF_ABLATION {
    tag "Motif ablation (suffix: ${params.suffix})"
    publishDir "${params.outdir}/audit/motif_ablation", mode: 'copy'

    input:
    path eval_report
    path selected_fasta
    path selected_meta
    path model_cfg

    output:
    path "ablated_${params.suffix}.fasta",          emit: fasta
    path "ablated_${params.suffix}_metadata.csv",   emit: meta
    path "logs/motif_ablation.log",                 emit: log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs
    python ${projectDir}/scripts/motif_ablation.py \
        --score_report ${eval_report} \
        --input_fasta ${selected_fasta} \
        --metadata ${selected_meta} \
        --output_fasta ablated_${params.suffix}.fasta \
        --output_metadata ablated_${params.suffix}_metadata.csv \
        --tf ANY \
        --config ${model_cfg} \
        --seed ${params.seed} \
        2>&1 | tee logs/motif_ablation.log
    """
}

// Process 6c: CpG Swap (changes CpG content in isolation)
process AUDIT_CPG_SWAP {
    tag "CpG swap (suffix: ${params.suffix})"
    publishDir "${params.outdir}/audit/cpg_swap", mode: 'copy'

    input:
    path eval_report
    path selected_fasta
    path selected_meta

    output:
    path "cpgswap_${params.suffix}.fasta",          emit: fasta
    path "cpgswap_${params.suffix}_metadata.csv",   emit: meta
    path "logs/cpg_swap.log",                       emit: log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs
    python ${projectDir}/scripts/cpg_swap.py \
        --score_report ${eval_report} \
        --input_fasta ${selected_fasta} \
        --metadata ${selected_meta} \
        --output_fasta cpgswap_${params.suffix}.fasta \
        --output_metadata cpgswap_${params.suffix}_metadata.csv \
        --seed ${params.seed} \
        2>&1 | tee logs/cpg_swap.log
    """
}

// Process 6d: Locus Survey (places one fixed intervention across many host loci)
process AUDIT_LOCUS_SURVEY {
    tag "Locus survey (suffix: ${params.suffix})"
    publishDir "${params.outdir}/audit/locus_survey", mode: 'copy'

    input:
    path windows_fasta
    path windows_meta
    path candidates_fasta

    output:
    path "survey_${params.suffix}.fasta",          emit: fasta
    path "survey_${params.suffix}_metadata.csv",   emit: meta
    path "survey_${params.suffix}_hosts.csv",      emit: hosts
    path "logs/locus_survey.log",                  emit: log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs
    python ${projectDir}/scripts/locus_survey.py \
        --windows_fasta ${windows_fasta} \
        --metadata ${windows_meta} \
        --candidates_fasta ${candidates_fasta} \
        --cell_type ${params.cell_type} \
        --output_fasta survey_${params.suffix}.fasta \
        --output_metadata survey_${params.suffix}_metadata.csv \
        --output_hosts survey_${params.suffix}_hosts.csv \
        --seed ${params.seed} \
        2>&1 | tee logs/locus_survey.log
    """
}

// Process 6e: Selected-vs-Rejected Grammar Comparison (no rescoring needed;
// reads the original oracle report directly)
process AUDIT_GRAMMAR {
    tag "Grammar comparison (suffix: ${params.suffix})"
    publishDir "${params.outdir}/audit/grammar", mode: 'copy'

    input:
    path eval_report
    path candidates_fasta
    path model_cfg

    output:
    path "grammar_${params.suffix}.csv",     emit: csv
    path "logs/selected_grammar.log",        emit: log

    script:
    """
    export PYTHONPATH="${projectDir}:\${PYTHONPATH:-}"
    if command -v conda >/dev/null 2>&1; then
        ENV_PATH=\$(conda env list | awk '\$1 == "ms_enhancer" {print \$NF}')
        if [ -n "\$ENV_PATH" ]; then
            export PATH="\$ENV_PATH/bin:\$PATH"
            export LD_LIBRARY_PATH="\$ENV_PATH/lib:\${LD_LIBRARY_PATH:-}"
        fi
    fi
    mkdir -p logs
    python ${projectDir}/scripts/compare_selected_grammar.py \
        --reports ${eval_report} \
        --fastas ${candidates_fasta} \
        --config ${model_cfg} \
        --output_csv grammar_${params.suffix}.csv \
        --seed ${params.seed}
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

    // Step 0: Fetch reference genome (skipped if data/hg38.fa already exists)
    GENOME_PREP()
    genome_fasta_ch  = GENOME_PREP.out.fasta

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
    EVALUATE_MAIN(
        params.suffix,
        GENERATE_SEQUENCES.out.candidates_fasta,
        GENERATE_SEQUENCES.out.candidates_meta,
        model_config_ch,
        genome_fasta_ch
    )
    main_eval_report_ch = EVALUATE_MAIN.out.eval_report

    // Step 5: Candidate selection
    SELECT_CANDIDATES(
        main_eval_report_ch,
        GENERATE_SEQUENCES.out.candidates_fasta,
        GENERATE_SEQUENCES.out.candidates_meta
    )

    // Step 6: Mechanistic auditing (set --run_audit false to skip for a quick smoketest)
    if (params.run_audit) {
        AUDIT_OCCLUSION(
            main_eval_report_ch,
            SELECT_CANDIDATES.out.selected_fasta,
            SELECT_CANDIDATES.out.selected_meta
        )
        EVALUATE_OCCLUSION(
            "${params.suffix}_occlusion",
            AUDIT_OCCLUSION.out.fasta,
            AUDIT_OCCLUSION.out.meta,
            model_config_ch,
            genome_fasta_ch
        )

        AUDIT_MOTIF_ABLATION(
            main_eval_report_ch,
            SELECT_CANDIDATES.out.selected_fasta,
            SELECT_CANDIDATES.out.selected_meta,
            model_config_ch
        )
        EVALUATE_MOTIF_ABLATION(
            "${params.suffix}_motif_ablation",
            AUDIT_MOTIF_ABLATION.out.fasta,
            AUDIT_MOTIF_ABLATION.out.meta,
            model_config_ch,
            genome_fasta_ch
        )

        AUDIT_CPG_SWAP(
            main_eval_report_ch,
            SELECT_CANDIDATES.out.selected_fasta,
            SELECT_CANDIDATES.out.selected_meta
        )
        EVALUATE_CPG_SWAP(
            "${params.suffix}_cpg_swap",
            AUDIT_CPG_SWAP.out.fasta,
            AUDIT_CPG_SWAP.out.meta,
            model_config_ch,
            genome_fasta_ch
        )

        AUDIT_LOCUS_SURVEY(
            BUILD_DATASET.out.windows_fasta,
            BUILD_DATASET.out.windows_meta,
            SELECT_CANDIDATES.out.selected_fasta
        )
        EVALUATE_LOCUS_SURVEY(
            "${params.suffix}_locus_survey",
            AUDIT_LOCUS_SURVEY.out.fasta,
            AUDIT_LOCUS_SURVEY.out.meta,
            model_config_ch,
            genome_fasta_ch
        )

        AUDIT_GRAMMAR(
            main_eval_report_ch,
            GENERATE_SEQUENCES.out.candidates_fasta,
            model_config_ch
        )
    }
}
