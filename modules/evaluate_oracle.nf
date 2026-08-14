nextflow.enable.dsl = 2

// Evaluate a FASTA with the in-silico oracle. Defined as a module so it can be
// included multiple times under different names: once for the main candidate
// scoring, and once per mechanistic-audit intervention (occlusion, motif
// ablation, CpG swap, locus survey) to rescore its perturbed sequences.
process EVALUATE_ORACLE {
    tag "Scoring ${label} with ${params.oracle} oracle"
    publishDir "${params.outdir}/evaluation", mode: 'copy'

    input:
    val  label
    path candidates_fasta
    path candidates_meta
    path model_cfg
    path genome_fa

    output:
    path "evaluation_results_${label}.json", emit: eval_report
    path "logs/evaluate_${label}.log",       emit: eval_log

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
    mkdir -p logs data
    ln -s \$(readlink -f ${genome_fa}) data/hg38.fa

    python ${projectDir}/evaluate.py \
        --input_fasta ${candidates_fasta} \
        --metadata ${candidates_meta} \
        --oracle ${params.oracle} \
        --config ${model_cfg} \
        --reference_fasta data/hg38.fa \
        --output_report evaluation_results_${label}.json
    mv logs/evaluate.log logs/evaluate_${label}.log
    """
}
