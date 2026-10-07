"""Analyse the TRUBA campaign outputs that answer Reviewers 1 and 2.

Four questions, each from a separate scoring run:

* **localized** — does the host dominance survive scoring the insert's own
  output bins instead of the whole window? Reviewer 1 argues that averaging over
  all 896 bins gives the host flanks the weight by construction, so the result
  may be the score's arithmetic rather than the model's behaviour. Both
  reductions come from one forward pass, so this compares them directly.
* **convergence** — do the direction proportions and heterogeneity stabilise as
  loci are added? Reviewer 2 asks whether 24 loci is enough; this subsamples a
  200-locus survey at increasing sizes to show where, or whether, it settles.
* **background** — are the conclusions specific to the non-immune background
  track set? The all-track dump makes alternative definitions a CPU
  recomputation rather than another GPU campaign.
* **separation** — can the oracle tell peak-derived from GC-matched non-peak
  inserts, and does it do so consistently at every host locus?

Usage:
    python scripts/revision_tier2.py localized --survey logs/survey_localized.json
    python scripts/revision_tier2.py convergence --survey logs/survey200.json
    python scripts/revision_tier2.py separation --survey logs/nonpeak_scored.json \
        --metadata data/fasta/nonpeak_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_scores(path: Path, field: str = "mssi_score") -> Dict[str, float]:
    """Load {sequence_id: score} for one of the two reductions."""
    payload = json.loads(path.read_text())
    return {sid: float(res[field]) for sid, res in payload["sequences"].items()}


def paired_effects(scores: Dict[str, float]) -> pd.DataFrame:
    """Edit minus control, per (locus, insert). Mirrors the Tier 1 definition."""
    rows = []
    for key, value in scores.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        rows.append({
            "locus": parts[0], "insert": parts[1],
            "variant": "_".join(parts[2:]), "mssi": value,
        })
    wide = (
        pd.DataFrame(rows)
        .pivot_table(index=["locus", "insert"], columns="variant", values="mssi")
        .reset_index()
    )
    wide["effect"] = wide["cpg_up"] - wide["ctl"]
    return wide


def per_locus(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-locus mean effect, SE and interval across inserts sharing a host."""
    out = []
    for locus, block in wide.groupby("locus"):
        eff = block["effect"].to_numpy()
        n = len(eff)
        mean, se = float(eff.mean()), float(eff.std(ddof=1) / np.sqrt(n))
        t_stat, p = stats.ttest_1samp(eff, 0.0)
        out.append({
            "locus": locus, "n": n, "baseline": float(block["intact"].mean()),
            "effect": mean, "se": se, "t": float(t_stat), "p": float(p),
        })
    return pd.DataFrame(out).sort_values("effect").reset_index(drop=True)


def heterogeneity(table: pd.DataFrame) -> Dict[str, float]:
    """Cochran's Q and I-squared on the per-locus estimates."""
    y = table["effect"].to_numpy(dtype=float)
    se = np.asarray(table["se"].to_numpy(), dtype=float).copy()

    # A locus whose inserts all move identically has SE 0, and an unguarded
    # 1/SE**2 turns every statistic below into nan without raising — a silent
    # wrong answer in a number the manuscript quotes. Two distinct cases:
    positive = se[se > 0]
    if not positive.size:
        # No locus carries any sampling uncertainty, so weighting is undefined
        # and Q diverges. Any spread between loci is then real by construction.
        spread = float(np.ptp(y))
        return {
            "Q": float("inf") if spread > 0 else 0.0,
            "df": len(y) - 1,
            "p": 0.0 if spread > 0 else 1.0,
            "I2": 100.0 if spread > 0 else 0.0,
            "pooled_effect": float(y.mean()),
        }
    # Otherwise a zero-SE locus is the most precisely estimated one, not an
    # undefined one, so it inherits the panel's smallest non-zero SE.
    se[se <= 0] = positive.min()

    w = 1.0 / se**2
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum())
    df = len(y) - 1
    return {
        "Q": q, "df": df, "p": float(stats.chi2.sf(q, df)),
        "I2": float(max(0.0, (q - df) / q) * 100) if q > 0 else 0.0,
        "pooled_effect": mu,
    }


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, order preserved."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def direction_counts(table: pd.DataFrame, alpha: float = 0.05) -> Dict[str, float]:
    """Significant negative/positive/undetermined counts, as Tier 1 reports them."""
    q = bh(table["p"].to_numpy())
    sig = q < alpha
    neg = table["effect"] < 0
    return {
        "n_loci": int(len(table)),
        "n_sig_neg": int((sig & neg).sum()),
        "n_sig_pos": int((sig & ~neg).sum()),
        "n_undetermined": int((~sig).sum()),
        "frac_point_negative": float(neg.mean()),
        "effect_min": float(table["effect"].min()),
        "effect_max": float(table["effect"].max()),
    }


def cmd_localized(args: argparse.Namespace) -> Dict[str, object]:
    """Compare the whole-window and insert-localised reductions side by side."""
    payload = json.loads(Path(args.survey).read_text())
    first = next(iter(payload["sequences"].values()))
    if "mssi_local" not in first:
        raise SystemExit(
            f"{args.survey} has no 'mssi_local'; it was produced before the "
            f"localised reduction existed. Re-score with the current oracle."
        )

    result: Dict[str, object] = {"local_bins": first["local_bins"]}

    tables = {}
    for label, field in (("global", "mssi_score"), ("local", "mssi_local")):
        wide = paired_effects(load_scores(Path(args.survey), field))
        table = per_locus(wide)
        tables[label] = (wide, table)
        result[label] = {
            "heterogeneity": heterogeneity(table),
            "directions": direction_counts(table),
        }

    # The claim under test is host dominance, so compare how much of the score's
    # variance each reduction assigns to the host rather than to the insert.
    for label, (wide, _) in tables.items():
        intact = wide.pivot_table(index="insert", columns="locus", values="intact")
        between = float(intact.mean(axis=0).std(ddof=1))
        within = float(intact.sub(intact.mean(axis=0), axis=1).to_numpy().std(ddof=1))
        result[label]["host_vs_insert"] = {
            "sd_between_loci": between,
            "sd_within_locus": within,
            "ratio": between / within if within else float("nan"),
        }

    # Do the two reductions even rank loci the same way?
    merged = tables["global"][1].merge(tables["local"][1], on="locus", suffixes=("_g", "_l"))
    rho, p = stats.spearmanr(merged["effect_g"], merged["effect_l"])
    result["agreement"] = {
        "spearman_rho": float(rho), "p": float(p),
        "sign_agreement": float((np.sign(merged["effect_g"]) == np.sign(merged["effect_l"])).mean()),
    }
    if args.out_csv:
        merged.to_csv(args.out_csv, index=False)
    return result


def cmd_convergence(args: argparse.Namespace) -> Dict[str, object]:
    """Subsample the large survey to show whether the estimates settle."""
    wide = paired_effects(load_scores(Path(args.survey), args.field))
    table = per_locus(wide)
    loci = table["locus"].tolist()
    rng = np.random.default_rng(args.seed)

    sizes = [s for s in (12, 24, 50, 100, 150, 200) if s <= len(loci)]
    if len(loci) not in sizes:
        sizes.append(len(loci))

    curve = []
    for size in sizes:
        reps = []
        # The full-panel point is exact, so it needs no resampling.
        n_reps = 1 if size == len(loci) else args.reps
        for _ in range(n_reps):
            pick = rng.choice(loci, size=size, replace=False) if size < len(loci) else loci
            sub = table[table["locus"].isin(pick)]
            het = heterogeneity(sub)
            dirs = direction_counts(sub)
            reps.append((het["I2"], dirs["frac_point_negative"],
                         dirs["n_sig_neg"] / size, dirs["n_sig_pos"] / size))
        arr = np.array(reps)
        curve.append({
            "n_loci": size,
            "I2_median": float(np.median(arr[:, 0])),
            "I2_iqr": [float(np.percentile(arr[:, 0], 25)), float(np.percentile(arr[:, 0], 75))],
            "frac_negative_median": float(np.median(arr[:, 1])),
            "frac_sig_neg_median": float(np.median(arr[:, 2])),
            "frac_sig_pos_median": float(np.median(arr[:, 3])),
        })

    out: Dict[str, object] = {
        "full_panel": {"heterogeneity": heterogeneity(table),
                       "directions": direction_counts(table)},
        "curve": curve,
    }
    if args.out_csv:
        table.to_csv(args.out_csv, index=False)
    return out


def cmd_separation(args: argparse.Namespace) -> Dict[str, object]:
    """Peak vs GC-matched non-peak, asked separately at every host locus."""
    scores_global = load_scores(Path(args.survey), "mssi_score")
    payload = json.loads(Path(args.survey).read_text())
    has_local = "mssi_local" in next(iter(payload["sequences"].values()))
    meta = pd.read_csv(args.metadata).set_index("peak_id")

    frames = []
    for label, field in [("global", "mssi_score")] + ([("local", "mssi_local")] if has_local else []):
        scores = load_scores(Path(args.survey), field)
        frame = meta.loc[list(scores)].copy()
        frame["mssi"] = [scores[i] for i in frame.index]
        frame["reduction"] = label
        frames.append(frame.reset_index())
    data = pd.concat(frames, ignore_index=True)

    out: Dict[str, object] = {}
    for label, block in data.groupby("reduction"):
        # The comparison class is whatever is paired against "peak" in this
        # file: real non-peak genomic sequence in the main test, GC-matched
        # synthetic random in the positive control. Hard-coding one name made
        # the other silently produce empty intersections and NaN everywhere.
        classes = set(block["insert_class"].unique()) - {"peak"}
        if len(classes) != 1:
            raise ValueError(
                f"Expected 'peak' plus exactly one comparison class, got "
                f"{sorted(block['insert_class'].unique())}."
            )
        other = classes.pop()

        per_host = []
        for locus, lb in block.groupby("locus"):
            peak = lb[lb.insert_class == "peak"].set_index("insert")["mssi"]
            non = lb[lb.insert_class == other].set_index("insert")["mssi"]
            shared = peak.index.intersection(non.index)
            if len(shared) < 2:
                raise ValueError(
                    f"Locus {locus}: only {len(shared)} inserts shared between "
                    f"'peak' and '{other}'. A silent NaN here would look like a null result."
                )
            # Paired by insert index: each non-peak insert was GC-matched to one
            # peak insert, so the pairing is part of the design, not a convenience.
            diff = (peak.loc[shared] - non.loc[shared]).to_numpy()
            t_stat, p = stats.ttest_rel(peak.loc[shared], non.loc[shared])
            auc = float((peak.loc[shared].to_numpy()[:, None] >
                         non.loc[shared].to_numpy()[None, :]).mean())
            per_host.append({
                "locus": locus, "n": len(shared), "mean_diff": float(diff.mean()),
                "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                "t": float(t_stat), "p": float(p), "auc": auc,
            })
        table = pd.DataFrame(per_host)
        table["q"] = bh(table["p"].to_numpy())
        sig_pos = int(((table["q"] < 0.05) & (table["mean_diff"] > 0)).sum())
        sig_neg = int(((table["q"] < 0.05) & (table["mean_diff"] < 0)).sum())
        out[label] = {
            "comparison_class": other,
            "n_hosts": int(len(table)),
            "hosts_separating_correctly": sig_pos,
            "hosts_separating_backwards": sig_neg,
            "hosts_undetermined": int((table["q"] >= 0.05).sum()),
            "mean_auc": float(table["auc"].mean()),
            "auc_range": [float(table["auc"].min()), float(table["auc"].max())],
            "mean_diff_range": [float(table["mean_diff"].min()), float(table["mean_diff"].max())],
            "per_host": table.to_dict("records"),
        }
        if args.out_csv:
            table.to_csv(f"{args.out_csv.rsplit('.', 1)[0]}_{label}.csv", index=False)
    return out


def cmd_background(args: argparse.Namespace) -> Dict[str, object]:
    """Recompute MSSI under alternative background track sets, no GPU needed."""
    import yaml

    dump = np.load(args.track_dump)
    ids = [str(s) for s in dump["sequence_ids"]]
    cfg = yaml.safe_load(Path(args.config).read_text())["evaluation"]["enformer"]
    targets = cfg["target_tracks_by_cell_type"][args.cell_type]

    sets = {"non_immune_original": cfg["background_tracks"]}
    if args.immune_background:
        sets["immune_related"] = [int(t) for t in args.immune_background.split(",")]

    out: Dict[str, object] = {}
    for reduction in ("tracks_global", "tracks_local"):
        if reduction not in dump:
            continue
        matrix = dump[reduction]
        for name, background in sets.items():
            scores = dict(zip(
                ids,
                matrix[:, targets].mean(axis=1) - matrix[:, background].mean(axis=1),
            ))
            table = per_locus(paired_effects({k: float(v) for k, v in scores.items()}))
            out[f"{reduction}:{name}"] = {
                "heterogeneity": heterogeneity(table),
                "directions": direction_counts(table),
            }
    return out


def cmd_dose(args: argparse.Namespace) -> Dict[str, object]:
    """Does the CpG edit's effect scale with how many dinucleotides were swapped?

    Reviewer 2 asks for a dose-response. The complication is that the effect's
    sign is host-dependent (Section 3.7), so pooling doses across loci would
    let opposite-signed hosts cancel and could show no trend even if every host
    had a clean one. The slope is therefore fitted per locus first, and the
    question "does it scale" is answered by how many loci show a monotonic
    relationship and in which direction.
    """
    surveys = []
    for spec in args.survey:
        label, path = spec.split("=", 1)
        surveys.append((float(label), Path(path)))
    surveys.sort()

    out: Dict[str, object] = {"doses": [d for d, _ in surveys]}

    for label, field in (("global", "mssi_score"), ("local", "mssi_local")):
        per_dose, by_locus = [], {}
        for dose, path in surveys:
            payload = json.loads(path.read_text())
            if field not in next(iter(payload["sequences"].values())):
                continue
            wide = paired_effects(load_scores(path, field))
            eff = wide["effect"].to_numpy()
            se = float(eff.std(ddof=1) / np.sqrt(len(eff)))
            per_dose.append({
                "dose": dose, "n": int(len(eff)),
                "mean_effect": float(eff.mean()), "se": se,
                "lo": float(eff.mean() - 1.96 * se), "hi": float(eff.mean() + 1.96 * se),
            })
            for locus, block in wide.groupby("locus"):
                by_locus.setdefault(locus, []).append((dose, float(block["effect"].mean())))

        if not per_dose:
            continue

        # Per-locus trend: Spearman against dose, so a monotone but non-linear
        # response still registers.
        slopes = []
        for locus, points in by_locus.items():
            points.sort()
            doses = [d for d, _ in points]
            effects = [e for _, e in points]
            rho, p = stats.spearmanr(doses, effects)
            fit = np.polyfit(doses, effects, 1)[0] if len(doses) > 1 else float("nan")
            slopes.append({
                "locus": locus, "spearman_rho": float(rho), "p": float(p),
                "slope_per_swap": float(fit),
                "effect_at_min": effects[0], "effect_at_max": effects[-1],
            })

        monotone_neg = sum(1 for s in slopes if s["spearman_rho"] <= -0.9)
        monotone_pos = sum(1 for s in slopes if s["spearman_rho"] >= 0.9)
        out[label] = {
            "per_dose": per_dose,
            "n_loci": len(slopes),
            "loci_monotone_negative": monotone_neg,
            "loci_monotone_positive": monotone_pos,
            "loci_non_monotone": len(slopes) - monotone_neg - monotone_pos,
            "median_slope_per_swap": float(np.median([s["slope_per_swap"] for s in slopes])),
            "per_locus": slopes,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("localized"); p.add_argument("--survey", required=True)
    p.add_argument("--out_csv", default=None); p.set_defaults(func=cmd_localized)

    p = sub.add_parser("convergence"); p.add_argument("--survey", required=True)
    p.add_argument("--field", default="mssi_score"); p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out_csv", default=None)
    p.set_defaults(func=cmd_convergence)

    p = sub.add_parser("separation"); p.add_argument("--survey", required=True)
    p.add_argument("--metadata", required=True); p.add_argument("--out_csv", default=None)
    p.set_defaults(func=cmd_separation)

    p = sub.add_parser("dose")
    p.add_argument("--survey", required=True, nargs="+",
                   help="One or more <dose>=<path.json>, dose being the realised CpG gain")
    p.add_argument("--out_csv", default=None); p.set_defaults(func=cmd_dose)

    p = sub.add_parser("background"); p.add_argument("--track_dump", required=True)
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--cell_type", default="CD4_T_cell")
    p.add_argument("--immune_background", default=None, help="Comma-separated track indices")
    p.add_argument("--out_csv", default=None); p.set_defaults(func=cmd_background)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print(json.dumps(args.func(args), indent=2, default=float))


def demo() -> None:
    """Self-check: the reductions and counts must behave on known input."""
    # Two loci with opposite, unambiguous edit effects.
    scores = {}
    rng0 = np.random.default_rng(1)
    for locus, sign in (("L00", +1.0), ("L01", -1.0)):
        for i in range(6):
            base = 1.0 + 0.01 * i
            scores[f"{locus}_i{i:02d}_intact"] = base
            scores[f"{locus}_i{i:02d}_ctl"] = base
            scores[f"{locus}_i{i:02d}_cpg_up"] = base + sign * 0.1 + rng0.normal(0, 1e-3)

    table = per_locus(paired_effects(scores))
    assert len(table) == 2
    counts = direction_counts(table)
    assert counts["n_sig_neg"] == 1 and counts["n_sig_pos"] == 1, counts

    het = heterogeneity(table)
    # Opposite effects with tiny within-locus spread is maximal heterogeneity.
    assert het["I2"] > 99, het

    # A panel drawn from one common effect must show little heterogeneity: the
    # loci differ only by sampling, so between-locus spread should not exceed
    # the within-locus spread that the SE already describes. Twelve loci, since
    # I-squared on three is too noisy to assert on.
    same = {}
    rng = np.random.default_rng(0)
    for li in range(12):
        for i in range(20):
            key = f"L{li:02d}_i{i:02d}"
            same[f"{key}_intact"] = 1.0
            same[f"{key}_ctl"] = 1.0
            same[f"{key}_cpg_up"] = 1.0 - 0.05 + rng.normal(0, 0.02)
    het_same = heterogeneity(per_locus(paired_effects(same)))
    assert het_same["I2"] < 50, het_same
    assert abs(het_same["pooled_effect"] + 0.05) < 0.01, het_same

    q = bh(np.array([0.001, 0.02, 0.5, 0.9]))
    assert np.all(np.diff(q) >= -1e-12) and q.max() <= 1.0

    # A locus with zero between-insert spread must not turn every statistic
    # into a silent nan, which is what an unguarded 1/SE**2 would do.
    degenerate = {}
    for locus, sign in (("L00", +1.0), ("L01", -1.0)):
        for i in range(6):
            degenerate[f"{locus}_i{i:02d}_intact"] = 1.0
            degenerate[f"{locus}_i{i:02d}_ctl"] = 1.0
            degenerate[f"{locus}_i{i:02d}_cpg_up"] = 1.0 + sign * 0.1
    het_deg = heterogeneity(per_locus(paired_effects(degenerate)))
    # Q legitimately diverges when no locus carries sampling error, but I2 must
    # stay a finite, reportable number rather than becoming nan.
    assert np.isfinite(het_deg["I2"]) and het_deg["I2"] == 100.0, het_deg
    assert not np.isnan(het_deg["Q"]), het_deg

    # Identical effects everywhere, still with no sampling error, is the other
    # half of that branch and must report no heterogeneity rather than 100.
    flat = {}
    for li in range(3):
        for i in range(6):
            key = f"L{li:02d}_i{i:02d}"
            flat[f"{key}_intact"] = 1.0
            flat[f"{key}_ctl"] = 1.0
            flat[f"{key}_cpg_up"] = 0.9
    het_flat = heterogeneity(per_locus(paired_effects(flat)))
    assert het_flat["I2"] == 0.0, het_flat
    print("demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
