# plot_modularity_suite_auto.py
#
# Post-hoc plotting for the 4-property modularity evaluation suite.
#
# It expects each experiment folder under a root (default: eval_results/<exp_name>)
# to contain:
#   - subspaces_assignment.json
#   - property1_selective_encoding.json
#   - property2_interference.json
#   - property3_factor_contribution.json
#   - property4_localized_adaptation.json
#   - (optional) summary_modularity_suite.json
#
# It will:
#   - Generate per-model plots in <exp_dir>/plots/
#   - Generate aggregated cross-model plots in eval_results/aggregated_plots/

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FACTORS = ["shape", "scale", "rot", "pos_x", "pos_y"]


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

REQUIRED_FILES = [
    "subspaces_assignment.json",
    "property1_selective_encoding.json",
    "property2_interference.json",
    "property3_factor_contribution.json",
    "property4_localized_adaptation.json",
]


def has_required_files(exp_dir: Path) -> bool:
    for fname in REQUIRED_FILES:
        if not (exp_dir / fname).exists():
            return False
    return True


def infer_label(exp_dir: Path, summary: dict | None) -> str:
    # Prefer a human-readable name from summary if available
    if summary is not None:
        name = summary.get("experiment_name", None)
        if isinstance(name, str) and len(name.strip()) > 0:
            return name.strip()
    # Fallback: folder name
    return exp_dir.name


def load_single_experiment(exp_dir: Path):
    exp_dir = Path(exp_dir)

    with open(exp_dir / "subspaces_assignment.json", "r") as f:
        subspaces = json.load(f)
    with open(exp_dir / "property1_selective_encoding.json", "r") as f:
        p1 = json.load(f)
    with open(exp_dir / "property2_interference.json", "r") as f:
        p2 = json.load(f)
    with open(exp_dir / "property3_factor_contribution.json", "r") as f:
        p3 = json.load(f)
    with open(exp_dir / "property4_localized_adaptation.json", "r") as f:
        p4 = json.load(f)

    summary_path = exp_dir / "summary_modularity_suite.json"
    summary = None
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

    label = infer_label(exp_dir, summary)

    return {
        "label": label,
        "exp_dir": exp_dir,
        "subspaces": subspaces,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "summary": summary,
    }


def discover_experiments(root: Path):
    """
    Scan root directory for subfolders that contain all required JSON files.
    """
    root = Path(root)
    experiments = []
    if not root.exists():
        print(f"[WARN] Root directory {root} does not exist.")
        return experiments

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if has_required_files(child):
            exp = load_single_experiment(child)
            experiments.append(exp)
        else:
            # You can uncomment this if you want to see which dirs are skipped
            # print(f"[INFO] Skipping {child}, missing required files.")
            pass

    if not experiments:
        print(f"[WARN] No valid experiments found under {root}.")
    else:
        print(f"Discovered {len(experiments)} experiments under {root}:")
        for e in experiments:
            print(f"  - {e['label']} ({e['exp_dir']})")

    return experiments


# -------------------------------------------------------------------------
# PER-MODEL PLOTS
# -------------------------------------------------------------------------

def plot_p1_selective_encoding_single(exp, out_dir):
    label = exp["label"]
    p1 = exp["p1"]["selective_encoding_per_factor"]

    rows = []
    for f in FACTORS:
        stats = p1[f]
        if stats["own_perf"] is None:
            continue
        rows.append({
            "factor": f,
            "own": stats["own_perf"],
            "comp": stats["comp_perf"],
            "rand": stats["rand_perf_mean"],
            "rand_std": stats["rand_perf_std"],
            "p_value": stats["p_value_own_gt_rand"],
        })

    if not rows:
        print(f"[WARN] No P1 stats for {label}, skipping P1 plot.")
        return

    df = pd.DataFrame(rows)

    # long format for grouped bar plot
    df_long = df.melt(
        id_vars=["factor"],
        value_vars=["own", "comp", "rand"],
        var_name="space",
        value_name="perf",
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(data=df_long, x="factor", y="perf", hue="space")
    plt.title(f"{label}: Property 1 – selective encoding")
    plt.ylabel("Probe performance (acc or R²)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P1_selective_encoding.png")
    plt.close()

    # Optional: p-values as bar of -log10(p)
    df["neg_log10_p"] = -np.log10(np.maximum(df["p_value"], 1e-12))
    plt.figure(figsize=(7, 3))
    sns.barplot(data=df, x="factor", y="neg_log10_p")
    plt.axhline(-np.log10(0.05), linestyle="--")
    plt.ylabel("-log10 p (own > random)")
    plt.title(f"{label}: P1 significance of own-subspace vs random")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P1_pvalues.png")
    plt.close()


def plot_p2_interference_heatmap_single(exp, out_dir):
    label = exp["label"]
    p2 = exp["p2"]

    inter = p2["interference_matrix"]  # dict[f][g]
    M = np.zeros((len(FACTORS), len(FACTORS)))
    for i, f in enumerate(FACTORS):
        for j, g in enumerate(FACTORS):
            M[i, j] = inter[f][g]

    plt.figure(figsize=(5.5, 5))
    sns.heatmap(
        M,
        cmap="coolwarm",
        xticklabels=FACTORS,
        yticklabels=FACTORS,
        cbar_kws={"label": "Δ error (f after ablating g)"},
    )
    plt.xlabel("Ablated factor subspace g")
    plt.ylabel("Affected factor f")
    plt.title(f"{label}: Property 2 – interference matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P2_interference_matrix.png")
    plt.close()

    summary = p2["summary"]
    print(f"[P2] {label}: modularity_score={summary['modularity_score']:.3f}, "
          f"self={summary['self_effect_mean']:.3f}, cross={summary['cross_effect_mean']:.3f}")


def plot_p3_drop_expert_heatmap_single(exp, out_dir):
    label = exp["label"]
    p3 = exp["p3"]

    mat = p3["drop_expert_contribution_matrix"]  # task t -> drop f
    M = np.zeros((len(FACTORS), len(FACTORS)))
    for i, t in enumerate(FACTORS):
        for j, f in enumerate(FACTORS):
            M[i, j] = mat[t][f]

    plt.figure(figsize=(5.5, 5))
    sns.heatmap(
        M,
        cmap="coolwarm",
        xticklabels=FACTORS,
        yticklabels=FACTORS,
        cbar_kws={"label": "Δ loss task[t] when dropping head[f]"},
    )
    plt.xlabel("Dropped factor head f")
    plt.ylabel("Task t")
    plt.title(f"{label}: Property 3 – drop-expert matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P3_drop_expert_matrix.png")
    plt.close()

    contrib_rows = []
    for f in FACTORS:
        s = p3["factor_contribution_scores"][f]
        contrib_rows.append({
            "factor": f,
            "self_contribution": s["self_contribution"],
            "leakage_mean": s["leakage_mean"],
        })
    df = pd.DataFrame(contrib_rows)

    plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df, x="factor", y="self_contribution")
    plt.title(f"{label}: P3 – self contribution per factor")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P3_self_contribution.png")
    plt.close()

    plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df, x="factor", y="leakage_mean")
    plt.title(f"{label}: P3 – leakage per factor (drop-expert)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P3_leakage.png")
    plt.close()


def plot_p4_localized_adaptation_single(exp, out_dir):
    label = exp["label"]
    p4 = exp["p4"]["per_factor"]

    rows = []
    for f in FACTORS:
        loc = p4[f]["localized"]
        uncon = p4[f]["unconstrained"]
        rows.append({
            "factor": f,
            "regime": "localized",
            "LA_score": loc["LA_score"],
            "G_target": loc["G_target"],
            "leak": loc["leak"],
        })
        rows.append({
            "factor": f,
            "regime": "unconstrained",
            "LA_score": uncon["LA_score"],
            "G_target": uncon["G_target"],
            "leak": uncon["leak"],
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="factor", y="LA_score", hue="regime")
    plt.title(f"{label}: Property 4 – localized vs unconstrained adaptation")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P4_LA_scores.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="factor", y="leak", hue="regime")
    plt.title(f"{label}: P4 – leakage (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_P4_leakage.png")
    plt.close()

    overall_loc = exp["p4"]["overall_LA_localized"]
    overall_uncon = exp["p4"]["overall_LA_unconstrained"]
    print(f"[P4] {label}: overall_LA_localized={overall_loc:.3f}, "
          f"overall_LA_unconstrained={overall_uncon:.3f}")


def make_per_model_plots(exp):
    label = exp["label"]
    out_dir = exp["exp_dir"] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Per-model plots for {label} ===")

    plot_p1_selective_encoding_single(exp, out_dir)
    plot_p2_interference_heatmap_single(exp, out_dir)
    plot_p3_drop_expert_heatmap_single(exp, out_dir)
    plot_p4_localized_adaptation_single(exp, out_dir)


# -------------------------------------------------------------------------
# AGGREGATED / CROSS-MODEL PLOTS
# -------------------------------------------------------------------------

def build_aggregated_data(experiments):
    rows_p1 = []
    rows_p2 = []
    rows_p3 = []
    rows_p4 = []

    for exp in experiments:
        label = exp["label"]

        # P1
        p1 = exp["p1"]["selective_encoding_per_factor"]
        for f in FACTORS:
            s = p1[f]
            if s["own_perf"] is None:
                continue
            rows_p1.append({
                "model": label,
                "factor": f,
                "own_perf": s["own_perf"],
                "comp_perf": s["comp_perf"],
                "rand_perf_mean": s["rand_perf_mean"],
                "delta_own_vs_comp": s["delta_own_vs_comp"],
                "delta_own_vs_rand": s["delta_own_vs_rand"],
                "p_value": s["p_value_own_gt_rand"],
            })

        # P2
        p2 = exp["p2"]
        rows_p2.append({
            "model": label,
            "modularity_score": p2["summary"]["modularity_score"],
            "self_effect_mean": p2["summary"]["self_effect_mean"],
            "cross_effect_mean": p2["summary"]["cross_effect_mean"],
        })

        # P3
        p3 = exp["p3"]
        for f in FACTORS:
            s = p3["factor_contribution_scores"][f]
            rows_p3.append({
                "model": label,
                "factor": f,
                "self_contribution": s["self_contribution"],
                "leakage_mean": s["leakage_mean"],
            })

        # P4
        p4 = exp["p4"]["per_factor"]
        for f in FACTORS:
            loc = p4[f]["localized"]
            uncon = p4[f]["unconstrained"]
            rows_p4.append({
                "model": label,
                "factor": f,
                "regime": "localized",
                "LA_score": loc["LA_score"],
                "G_target": loc["G_target"],
                "leak": loc["leak"],
            })
            rows_p4.append({
                "model": label,
                "factor": f,
                "regime": "unconstrained",
                "LA_score": uncon["LA_score"],
                "G_target": uncon["G_target"],
                "leak": uncon["leak"],
            })

    df_p1 = pd.DataFrame(rows_p1) if rows_p1 else pd.DataFrame()
    df_p2 = pd.DataFrame(rows_p2) if rows_p2 else pd.DataFrame()
    df_p3 = pd.DataFrame(rows_p3) if rows_p3 else pd.DataFrame()
    df_p4 = pd.DataFrame(rows_p4) if rows_p4 else pd.DataFrame()

    return df_p1, df_p2, df_p3, df_p4


def plot_crossmodel_p1(df_p1, out_dir):
    if df_p1.empty:
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p1, x="model", y="delta_own_vs_rand", hue="factor")
    plt.title("P1 – own-subspace advantage over random (by model and factor)")
    plt.ylabel("own_perf - rand_perf_mean")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P1_delta_own_vs_rand.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p1, x="model", y="delta_own_vs_comp", hue="factor")
    plt.title("P1 – own-subspace advantage over complement")
    plt.ylabel("own_perf - comp_perf")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P1_delta_own_vs_comp.png")
    plt.close()


def plot_crossmodel_p2(df_p2, out_dir):
    if df_p2.empty:
        return

    plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_p2, x="model", y="modularity_score")
    plt.title("P2 – modularity score (lower cross/self interference)")
    plt.ylabel("modularity_score")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P2_modularity_score.png")
    plt.close()


def plot_crossmodel_p3(df_p3, experiments, out_dir):
    if df_p3.empty:
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p3, x="model", y="self_contribution", hue="factor")
    plt.title("P3 – self contribution per factor across models")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P3_self_contribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p3, x="model", y="leakage_mean", hue="factor")
    plt.title("P3 – leakage per factor across models")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P3_leakage.png")
    plt.close()

    rows = []
    for exp in experiments:
        label = exp["label"]
        rec = exp["p3"]["recovery"]
        rows.append({
            "model": label,
            "shape_acc_ratio": rec["shape_acc_ratio"],
            "cont_mse_ratio": rec["cont_mse_ratio"],
        })
    df_rec = pd.DataFrame(rows)

    plt.figure(figsize=(7, 3.5))
    df_rec_long = df_rec.melt(
        id_vars=["model"],
        value_vars=["shape_acc_ratio", "cont_mse_ratio"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=df_rec_long, x="model", y="value", hue="metric")
    plt.title("P3 – recovery (factored vs baseline)")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P3_recovery.png")
    plt.close()


def plot_crossmodel_p4(df_p4, out_dir):
    if df_p4.empty:
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p4, x="model", y="LA_score", hue="regime")
    plt.title("P4 – localized adaptation score (localized vs unconstrained)")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P4_LA_scores.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_p4, x="model", y="leak", hue="regime")
    plt.title("P4 – leakage across models (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "allmodels_P4_leakage.png")
    plt.close()


def make_aggregated_plots(experiments, root):
    print("\n=== Cross-model aggregated plots ===")
    out_dir = Path(root) / "aggregated_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_p1, df_p2, df_p3, df_p4 = build_aggregated_data(experiments)

    plot_crossmodel_p1(df_p1, out_dir)
    plot_crossmodel_p2(df_p2, out_dir)
    plot_crossmodel_p3(df_p3, experiments, out_dir)
    plot_crossmodel_p4(df_p4, out_dir)

    print(f"Saved aggregated plots to {out_dir}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main(root):
    experiments = discover_experiments(root)
    if not experiments:
        return

    # per-model
    for exp in experiments:
        make_per_model_plots(exp)

    # aggregated
    if len(experiments) > 1:
        make_aggregated_plots(experiments, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="eval_results",
        help="Root directory containing experiment subfolders",
    )
    args = parser.parse_args()
    main(args.root)
