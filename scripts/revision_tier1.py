"""Reviewer-requested analyses that need no oracle re-run.

Everything here is recomputed from the scoring logs already in ``logs/`` and the
FASTAs in ``data/fasta/``, so it runs on CPU in seconds and adds no API or GPU
cost. Four reviewer asks are answered:

* per-locus edit effects with confidence intervals and a count of how many
  exclude zero in each direction, instead of a bare I-squared (R3.3);
* insert-level rank correlation of oracle score between host loci, which
  separates group enrichment from preservation of ranking (R2.4);
* a mixed-effects refit with an explicit edit-by-locus interaction and a random
  effect for insert identity, which matches the repeated-insert design better
  than the meta-analytic Q/I-squared summary (R2.7);
* the realised number of CpG edits per insert and of ablated sites per dose,
  so the effect sizes are reported with the dose that produced them (R2.5, R2.6).

Usage:
    python scripts/revision_tier1.py --out_dir paper/func_integ/revised
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# 10,000 permutations were used throughout, so this is the attainable floor.
P_FLOOR = 1e-4


def read_fasta(path: Path) -> Dict[str, str]:
    """Read a FASTA into {id: sequence}, taking the id up to the first space."""
    seqs: Dict[str, str] = {}
    name = None
    chunks: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line.upper())
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def load_scores(path: Path) -> Dict[str, float]:
    """Load {sequence_id: MSSI} from an evaluate.py output log."""
    payload = json.loads(path.read_text())
    ids = list(payload["sequences"])
    return dict(zip(ids, payload["mssi_scores"]))


def paired_effects(scores: Dict[str, float]) -> pd.DataFrame:
    """Tabulate the edit effect for every (locus, insert) pair.

    The survey writes three variants per pair: ``intact``, ``cpg_up`` (the CpG
    edit) and ``ctl`` (the composition-matched AT/TA control). The effect of
    interest is edit minus control, which cancels the "some DNA was altered"
    component that both variants share.

    Returns:
        One row per (locus, insert) with the edited, control and intact scores.
    """
    rows = []
    for key, value in scores.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        locus, insert, variant = parts[0], parts[1], "_".join(parts[2:])
        rows.append({"locus": locus, "insert": insert, "variant": variant, "mssi": value})

    wide = (
        pd.DataFrame(rows)
        .pivot_table(index=["locus", "insert"], columns="variant", values="mssi")
        .reset_index()
    )
    wide["effect"] = wide["cpg_up"] - wide["ctl"]
    return wide


def per_locus_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-locus mean effect with a 95% CI and a two-sided one-sample t-test.

    The oracle is deterministic, so the spread being summarised here is the
    spread across inserts sharing a host, not measurement error. That is what
    the standard error means in every interval below.
    """
    out = []
    for locus, block in wide.groupby("locus"):
        effects = block["effect"].to_numpy()
        n = len(effects)
        mean = float(effects.mean())
        se = float(effects.std(ddof=1) / np.sqrt(n))
        crit = stats.t.ppf(0.975, n - 1)
        t_stat, p_value = stats.ttest_1samp(effects, 0.0)
        out.append(
            {
                "locus": locus,
                "n": n,
                "baseline": float(block["intact"].mean()),
                "effect": mean,
                "se": se,
                "lo": mean - crit * se,
                "hi": mean + crit * se,
                "t": float(t_stat),
                "p": float(p_value),
                "frac_neg": float((effects < 0).mean()),
            }
        )
    table = pd.DataFrame(out).sort_values("effect").reset_index(drop=True)
    table["q"] = benjamini_hochberg(table["p"].to_numpy())
    return table


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values, order preserved."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def direction_counts(table: pd.DataFrame, alpha: float = 0.05) -> Dict[str, object]:
    """Count how many per-locus intervals exclude zero, by direction.

    A large p-value is not evidence of absence, so loci whose interval contains
    zero are counted as undetermined rather than as null.
    """
    significant = table["q"] < alpha
    negative = significant & (table["effect"] < 0)
    positive = significant & (table["effect"] > 0)
    sign_test = stats.binomtest(int((table["effect"] < 0).sum()), len(table), 0.5)
    return {
        "n_loci": int(len(table)),
        "n_significant": int(significant.sum()),
        "n_sig_negative": int(negative.sum()),
        "n_sig_positive": int(positive.sum()),
        "n_undetermined": int((~significant).sum()),
        "n_point_negative": int((table["effect"] < 0).sum()),
        "sign_test_p": float(sign_test.pvalue),
        "effect_min": float(table["effect"].min()),
        "effect_max": float(table["effect"].max()),
        "span": float(table["effect"].max() - table["effect"].min()),
    }


def insert_rank_transfer(wide: pd.DataFrame) -> Dict[str, object]:
    """Spearman correlation of per-insert score between every pair of host loci.

    Group-level transfer (selected inserts beating unselected ones at a new
    locus) is a weaker property than rank preservation: the group contrast can
    hold while the within-group ordering is scrambled. This measures the
    ordering directly, on the unedited inserts.
    """
    matrix = wide.pivot_table(index="insert", columns="locus", values="intact")
    loci = list(matrix.columns)
    rhos = []
    for i, a in enumerate(loci):
        for b in loci[i + 1:]:
            pair = matrix[[a, b]].dropna()
            if len(pair) < 4:
                continue
            rho, _ = stats.spearmanr(pair[a], pair[b])
            if np.isfinite(rho):
                rhos.append(rho)
    rhos = np.array(rhos)
    return {
        "n_pairs": int(len(rhos)),
        "mean_rho": float(rhos.mean()),
        "median_rho": float(np.median(rhos)),
        "q25": float(np.percentile(rhos, 25)),
        "q75": float(np.percentile(rhos, 75)),
        "frac_positive": float((rhos > 0).mean()),
        "min_rho": float(rhos.min()),
        "max_rho": float(rhos.max()),
    }


def mixed_effects(wide: pd.DataFrame) -> Dict[str, object]:
    """Fit MSSI ~ edit * locus with a random intercept for insert identity.

    Q and I-squared treat loci as independent studies and ignore that the same
    inserts recur at every locus. This fit states the edit-by-locus interaction
    directly and lets insert identity absorb its own variance, which is the
    design actually used.
    """
    import statsmodels.formula.api as smf

    long = wide.melt(
        id_vars=["locus", "insert"],
        value_vars=["cpg_up", "ctl"],
        var_name="variant",
        value_name="mssi",
    )
    long["edit"] = (long["variant"] == "cpg_up").astype(int)

    # Centre MSSI within locus: absolute level differs several-fold between
    # loci (that is Section 3.3), and leaving it in would let the locus main
    # effect dominate the fit without telling us anything about the edit.
    long["mssi_c"] = long["mssi"] - long.groupby("locus")["mssi"].transform("mean")

    reduced = smf.mixedlm("mssi_c ~ edit", long, groups=long["insert"]).fit(reml=False)
    full = smf.mixedlm("mssi_c ~ edit * C(locus)", long, groups=long["insert"]).fit(reml=False)

    # Likelihood-ratio test of the interaction block.
    lr = 2 * (full.llf - reduced.llf)
    df = full.df_modelwc - reduced.df_modelwc
    p_value = float(stats.chi2.sf(lr, df)) if df > 0 else float("nan")

    return {
        "lr_statistic": float(lr),
        "df": int(df),
        "p_interaction": p_value,
        "edit_main_effect": float(reduced.params["edit"]),
        "edit_main_se": float(reduced.bse["edit"]),
        "insert_var": float(reduced.cov_re.iloc[0, 0]),
        "residual_var": float(reduced.scale),
    }


def cpg_edit_doses(fasta: Path) -> pd.DataFrame:
    """Count how many CpG sites the edit actually added, per (locus, insert).

    An effect size is uninterpretable without the dose that produced it, and the
    dose is a property of each insert's own GC dinucleotide content rather than a
    constant.
    """
    seqs = read_fasta(fasta)
    rows = []
    for key, sequence in seqs.items():
        if not key.endswith("_intact"):
            continue
        stem = key[: -len("_intact")]
        edited = seqs.get(f"{stem}_cpg_up")
        control = seqs.get(f"{stem}_ctl")
        if edited is None:
            continue
        locus, insert = stem.split("_")[0], stem.split("_")[1]
        before = sequence.count("CG")
        after = edited.count("CG")
        rows.append(
            {
                "locus": locus,
                "insert": insert,
                "cpg_before": before,
                "cpg_after": after,
                "cpg_added": after - before,
                "n_control_swaps": hamming(sequence, control) // 2 if control else np.nan,
                "n_swaps": hamming(sequence, edited) // 2,
            }
        )
    return pd.DataFrame(rows)


def hamming(a: str, b: str) -> int:
    """Number of differing positions between two equal-length sequences."""
    return sum(1 for x, y in zip(a, b) if x != y)


def realised_doses(hitcounts_path: Path) -> pd.DataFrame:
    """Report how many sites each dose level actually ablated.

    ``motif_ablation.py`` skips a dose for an insert that carries fewer hits
    than the dose requests, so the dose levels are not evaluated on identical
    insert sets and the per-dose n has to be reported with the effect.
    """
    payload = json.loads(hitcounts_path.read_text())
    rows = []
    for key, value in payload.items():
        parts = key.split("_")
        if len(parts) < 2 or not parts[1].startswith("d"):
            continue
        rows.append({"insert": parts[0], "dose": parts[1], "variant": "_".join(parts[2:]), "hits": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame[frame["variant"] == "abl"]
        .groupby("dose")
        .agg(n_inserts=("hits", "size"), mean_hits=("hits", "mean"), min_hits=("hits", "min"))
        .reset_index()
    )


def latex_per_locus(table: pd.DataFrame, meta: pd.DataFrame) -> str:
    """Emit the per-locus effect table as a LaTeX tabular body."""
    merged = table.merge(meta[["locus", "chrom"]], on="locus", how="left")
    lines = []
    for _, row in merged.iterrows():
        mark = "" if row["q"] >= 0.05 else ("$^{-}$" if row["effect"] < 0 else "$^{+}$")
        lines.append(
            f"{row['locus']} & {row['chrom']} & {row['baseline']:.3f} & "
            f"{row['effect']:+.4f}{mark} & ({row['lo']:+.4f}, {row['hi']:+.4f}) & "
            f"{fmt_p(row['q'])} \\\\"
        )
    return "\n".join(lines)


def fmt_p(value: float) -> str:
    """Format a p/q value in LaTeX, respecting the permutation floor."""
    if value < P_FLOOR:
        return "$< 10^{-4}$"
    if value < 0.001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"${mantissa}\\times 10^{{{int(exponent)}}}$"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, default=Path("logs/survey.json"))
    parser.add_argument("--survey_borzoi", type=Path, default=Path("logs/survey_borzoi.json"))
    parser.add_argument("--survey_effects", type=Path, default=Path("logs/survey_effects.csv"))
    parser.add_argument("--survey_fasta", type=Path, default=Path("data/fasta/survey.fasta"))
    parser.add_argument("--hitcounts", type=Path, default=Path("logs/abl_dose_hitcounts.json"))
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wide = paired_effects(load_scores(args.survey))
    table = per_locus_table(wide)
    counts = direction_counts(table)
    ranks = insert_rank_transfer(wide)
    mixed = mixed_effects(wide)
    doses = cpg_edit_doses(args.survey_fasta)

    meta = pd.read_csv(args.survey_effects)[["locus", "chrom", "peak_score"]]

    table.merge(meta, on="locus", how="left").to_csv(args.out_dir / "per_locus_effects.csv", index=False)
    doses.to_csv(args.out_dir / "cpg_edit_doses.csv", index=False)

    summary = {"direction_counts": counts, "insert_rank_transfer": ranks, "mixed_effects": mixed}

    # Borzoi, so the same per-locus table exists for the replication oracle.
    if args.survey_borzoi.exists():
        bz = per_locus_table(paired_effects(load_scores(args.survey_borzoi)))
        bz.merge(meta, on="locus", how="left").to_csv(
            args.out_dir / "per_locus_effects_borzoi.csv", index=False
        )
        summary["direction_counts_borzoi"] = direction_counts(bz)
        joined = table.merge(bz, on="locus", suffixes=("_enf", "_bz"))
        rho, p_value = stats.spearmanr(joined["effect_enf"], joined["effect_bz"])
        rho_abs, p_abs = stats.spearmanr(joined["effect_enf"].abs(), joined["effect_bz"].abs())
        summary["cross_oracle"] = {
            "rho_signed": float(rho),
            "p_signed": float(p_value),
            "rho_magnitude": float(rho_abs),
            "p_magnitude": float(p_abs),
        }

    if args.hitcounts.exists():
        realised = realised_doses(args.hitcounts)
        if not realised.empty:
            realised.to_csv(args.out_dir / "realised_doses.csv", index=False)
            summary["realised_doses"] = realised.to_dict("records")

    summary["cpg_dose"] = {
        "mean_added": float(doses["cpg_added"].mean()),
        "sd_added": float(doses["cpg_added"].std(ddof=1)),
        "min_added": int(doses["cpg_added"].min()),
        "max_added": int(doses["cpg_added"].max()),
        "mean_swaps": float(doses["n_swaps"].mean()),
    }

    (args.out_dir / "tier1_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "per_locus_table.tex").write_text(latex_per_locus(table, meta))

    logger.info(json.dumps(summary, indent=2))
    logger.info("Wrote outputs to %s", args.out_dir)


def demo() -> None:
    """Self-check on synthetic data with a known answer."""
    # Two loci, opposite edit signs, five inserts each: the per-locus table must
    # recover both signs and the direction counts must see one of each.
    scores = {}
    rng = np.random.default_rng(0)
    for locus, sign in (("L00", +1.0), ("L01", -1.0)):
        for i in range(5):
            base = 1.0 + 0.01 * i
            scores[f"{locus}_i{i:02d}_intact"] = base
            scores[f"{locus}_i{i:02d}_ctl"] = base + rng.normal(0, 1e-5)
            scores[f"{locus}_i{i:02d}_cpg_up"] = base + sign * 0.1 + rng.normal(0, 1e-5)

    wide = paired_effects(scores)
    assert len(wide) == 10, wide
    table = per_locus_table(wide)
    assert len(table) == 2
    assert table.loc[table["locus"] == "L00", "effect"].iloc[0] > 0.09
    assert table.loc[table["locus"] == "L01", "effect"].iloc[0] < -0.09

    counts = direction_counts(table)
    assert counts["n_sig_negative"] == 1, counts
    assert counts["n_sig_positive"] == 1, counts
    assert counts["n_undetermined"] == 0, counts

    # A CI that straddles zero must count as undetermined, not as null.
    flat = {}
    for i in range(5):
        flat[f"L00_i{i:02d}_intact"] = 1.0
        flat[f"L00_i{i:02d}_ctl"] = 1.0 + rng.normal(0, 0.05)
        flat[f"L00_i{i:02d}_cpg_up"] = 1.0 + rng.normal(0, 0.05)
    flat_table = per_locus_table(paired_effects(flat))
    assert direction_counts(flat_table)["n_undetermined"] == 1

    # Rank transfer: identical ordering at both loci must give rho = 1.
    assert insert_rank_transfer(wide)["mean_rho"] > 0.99

    # BH is monotone and bounded.
    q = benjamini_hochberg(np.array([0.001, 0.02, 0.5, 0.9]))
    assert np.all(np.diff(q) >= -1e-12) and q.max() <= 1.0

    # hamming counts differing positions, which is twice the number of swaps.
    assert hamming("ACGT", "AGCT") == 2

    print("demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
