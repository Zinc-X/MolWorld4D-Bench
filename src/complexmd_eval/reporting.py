"""Compact, deterministic reports for evaluation runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPORT_METRICS = [
    "physical_valid_frame_rate",
    "physical_soft_valid_frame_rate",
    "bond_relative_mae",
    "mean_clashes_per_frame",
    "paired_backbone_rmsd_A_mean",
    "paired_ligand_rmsd_A_mean",
    "overmove_bb",
    "rmsf_backbone_pearson",
    "rmsf_backbone_amplitude_ratio",
    "pca_gaussian_w2",
    "contact_frequency_pearson",
    "contact_frequency_f1",
    "ligand_stay_rate",
    "acf_pca1_rmse",
    "state_occupancy_js_divergence",
    "transition_matrix_rmse",
    "implied_timescale_log_rmse",
    "mfpt_log_median_abs_error",
    "kinetics_estimable",
    "fes_js_divergence",
    "ramachandran_phi_js_divergence",
    "chi1_rotamer_js_divergence",
    "ligand_dihedral_js_divergence_mean",
    "protein_rg_wasserstein_A",
    "protein_sasa_wasserstein_A2",
    "secondary_structure_js_divergence",
    "interaction_class_occupancy_js_divergence",
    "prediction_throughput_ns_per_day",
    "speedup_vs_reference_wall",
    "trust_horizon_ns",
    "within_reference_envelope_rate",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.5g}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_by_model(per_case: pd.DataFrame) -> pd.DataFrame:
    if per_case.empty:
        return pd.DataFrame()
    numeric = [c for c in REPORT_METRICS if c in per_case.columns]
    summary = per_case.groupby("model", dropna=False)[numeric].median().reset_index()
    counts = per_case.groupby("model", dropna=False).agg(
        n_cases=("case_id", "nunique"),
        n_case_rows=("case_id", "size"),
    ).reset_index()
    return counts.merge(summary, on="model", how="left")


def _bar(summary: pd.DataFrame, metric: str, out: Path) -> None:
    if metric not in summary.columns or summary[metric].notna().sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(summary)), 4.0))
    ax.bar(summary["model"].astype(str), summary[metric])
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_figures(summary: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in (
        "physical_valid_frame_rate",
        "physical_soft_valid_frame_rate",
        "overmove_bb",
        "rmsf_backbone_amplitude_ratio",
        "pca_gaussian_w2",
        "contact_frequency_f1",
        "state_occupancy_js_divergence",
        "transition_matrix_rmse",
        "fes_js_divergence",
        "protein_rg_wasserstein_A",
        "secondary_structure_js_divergence",
    ):
        _bar(summary, metric, fig_dir / f"{metric}.png")
    if windows.empty or "window_end_ns" not in windows.columns:
        return
    for metric in ("physical_valid_frame_rate", "physical_soft_valid_frame_rate", "overmove_bb"):
        if metric not in windows.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        grouped = windows.groupby(["model", "window_end_ns"])[metric].median().reset_index()
        for model, group in grouped.groupby("model"):
            ax.plot(group["window_end_ns"], group[metric], marker="o", label=str(model))
        ax.set_xlabel("physical time window (ns)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"window_{metric}.png", dpi=180)
        plt.close(fig)


def write_markdown_report(
    protocol: dict,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    out_dir: Path,
    coverage: pd.DataFrame | None = None,
    module_status: pd.DataFrame | None = None,
) -> None:
    lines = [
        f"# ComplexMD-Eval: {protocol['suite']}",
        "",
        f"- Profile: `{protocol['profile']}`",
        f"- Protocol hash: `{protocol['_protocol_hash']}`",
        f"- Cases requested: {len(protocol['cases'])}",
        f"- Failures: {len(failures)}",
        "",
        "## Model summary",
        "",
    ]
    if summary.empty:
        lines.append("No successful case results.")
    else:
        lines.append(_markdown_table(summary))
    lines.extend(["", "## Interpretation rules", ""])
    lines.extend(
        [
            "- Physical validity, motion amplitude, ensemble agreement, temporal agreement, and interaction fidelity are separate claims.",
            "- Paired RMSD is a short-horizon diagnostic, not a long-horizon trajectory score.",
            "- `overmove_bb` and RMSF amplitude ratio near one indicate matched motion amplitude; low values can indicate freezing.",
            "- Missing or non-estimable metrics remain missing and are never converted to zero.",
            "- Compare only runs with the same resolved common clock, horizon, atom representation, and physical windows.",
            "- Inspect `timebase_audit.csv` and `parameter_resolution.csv` for resampling and time-rounding decisions.",
            "- Kinetic metrics are conditional on transition evidence; inspect `kinetics_estimable` and module status.",
            "- Proxy chemistry is labelled explicitly when hydrogens or formal charges are unavailable.",
        ]
    )
    if coverage is not None and not coverage.empty:
        incomplete = coverage[coverage["coverage"] < 1.0]
        lines.extend(["", "## Metric coverage", ""])
        lines.append(
            f"Scalar metrics: {len(coverage)} total; "
            f"{int((coverage['coverage'] == 1.0).sum())} complete; {len(incomplete)} incomplete."
        )
        if not incomplete.empty:
            lines.extend(["", _markdown_table(incomplete)])
    if module_status is not None and not module_status.empty:
        status_summary = module_status.groupby(["module", "status"], dropna=False).size().reset_index(name="count")
        lines.extend(["", "## Module estimability", "", _markdown_table(status_summary)])
    if not failures.empty:
        lines.extend(["", "## Failures", "", _markdown_table(failures)])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
