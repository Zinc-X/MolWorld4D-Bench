"""End-to-end protocol evaluation."""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ._version import __version__
from .advanced import (
    bootstrap_case_medians,
    moving_block_bootstrap_mean,
    replica_normalized_trust_metrics,
    slow_state_metrics,
    windowed_dynamics_metrics,
)
from .geometry import align_trajectory
from .interactions import interaction_metrics, solvent_metrics
from .io import build_masks, load_trajectory, validate_atom_mapping
from .metrics import (
    acf_rmse,
    acf_curve,
    contact_frequency,
    contact_metrics,
    contact_profile,
    motion_metrics,
    paired_metrics,
    paired_curve,
    pca_metrics,
    physical_metrics,
    project_pca,
    rmsf_metrics,
    rmsf_profile,
    window_metrics,
)
from .models import Trajectory
from .observables import efficiency_metrics, thermodynamic_metrics
from .protocol import load_protocol, public_protocol, resolve_path
from .reporting import summarize_by_model, write_figures, write_markdown_report
from .structural import structural_metrics


def _items(group: dict, base: Path) -> list[dict]:
    values = []
    group_metadata = dict(group.get("metadata", {}) or {})
    for item in group["trajectories"]:
        raw = dict(item) if isinstance(item, dict) else {"path": item}
        metadata = dict(group_metadata)
        metadata.update(raw.get("metadata", {}) or {})
        for key in ("runtime_s", "hardware", "precision", "batch_size"):
            if key in raw:
                metadata[key] = raw[key]
            elif key in group:
                metadata[key] = group[key]
        values.append(
            {
                "path": resolve_path(str(raw["path"]), base),
                "observables": resolve_path(str(raw["observables"]), base) if raw.get("observables") else None,
                "metadata": metadata,
            }
        )
    return values


def _load_group(group: dict, topology: Path, base: Path, analysis: dict) -> list[Trajectory]:
    stride = int(analysis.get("frame_stride", 1))
    max_frames = analysis.get("max_frames")
    max_frames = None if max_frames is None else int(max_frames)
    return [
        load_trajectory(
            item["path"], topology, float(group["dt_ns"]), stride=stride, max_frames=max_frames,
            observables_path=item["observables"], metadata=item["metadata"],
        )
        for item in _items(group, base)
    ]


def _pool(trajectories: list[Trajectory]) -> Trajectory:
    positions = np.concatenate([trajectory.positions for trajectory in trajectories], axis=0)
    time = np.arange(len(positions), dtype=float)
    return Trajectory(
        positions, time, trajectories[0].atoms, "pooled-reference",
        trajectories[0].topology_path,
    )


def _tag(table: pd.DataFrame, case_id: str, model: str, replica: int) -> pd.DataFrame:
    if table.empty:
        return table
    table = table.copy()
    table.insert(0, "replica", replica)
    table.insert(0, "model", model)
    table.insert(0, "case_id", case_id)
    return table


def _runtime_lookup(config: dict, base: Path) -> tuple[pd.DataFrame | None, dict]:
    efficiency = dict(config.get("efficiency", {}) or {})
    path = efficiency.get("runtime_table")
    if not path:
        return None, efficiency
    return pd.read_csv(resolve_path(str(path), base)), efficiency


def _attach_runtime(
    trajectory: Trajectory,
    table: pd.DataFrame | None,
    config: dict,
    case_id: str,
    model: str,
    replica: int,
    kind: str,
) -> Trajectory:
    if table is None or "runtime_s" in trajectory.metadata:
        return trajectory
    applies_to = str(config.get("applies_to", "prediction"))
    if applies_to not in {"both", kind}:
        return trajectory
    case_column = str(config.get("case_column", "case_id"))
    runtime_column = str(config.get("runtime_column", "runtime_s"))
    if case_column not in table or runtime_column not in table:
        raise ValueError(f"efficiency runtime table requires columns {case_column!r} and {runtime_column!r}")
    selected = table[table[case_column].astype(str) == str(case_id)]
    model_column = config.get("model_column")
    if model_column and model_column in selected:
        selected = selected[selected[model_column].astype(str) == str(model)]
    replica_column = config.get("replica_column")
    if replica_column and replica_column in selected:
        selected = selected[pd.to_numeric(selected[replica_column], errors="coerce") == replica]
    if len(selected) != 1:
        raise ValueError(f"efficiency runtime lookup for {case_id}/{model}/{replica} matched {len(selected)} rows")
    metadata = dict(trajectory.metadata)
    metadata["runtime_s"] = float(selected.iloc[0][runtime_column])
    for key in ("hardware", "precision", "batch_size"):
        column = config.get(f"{key}_column")
        if column and column in selected:
            metadata[key] = selected.iloc[0][column]
        elif key in config:
            metadata[key] = config[key]
    return replace(trajectory, metadata=metadata)


def _numeric_median_frame(group: pd.DataFrame) -> dict:
    row = {
        "case_id": group["case_id"].iloc[0],
        "model": group["model"].iloc[0],
        "n_prediction_replicas": int(group["replica"].nunique()),
    }
    for column in group.columns:
        if column in {"case_id", "model", "replica", "source"}:
            continue
        values = pd.to_numeric(group[column], errors="coerce")
        if values.notna().any():
            row[column] = float(values.median())
    return row


def _reference_baselines(
    case_id: str,
    references: list[Trajectory],
    masks,
    analysis: dict,
    enabled: dict,
) -> tuple[pd.DataFrame, dict]:
    if len(references) < 2:
        return pd.DataFrame(), {
            "case_id": case_id, "model": "*", "replica": -1, "module": "reference_baseline",
            "status": "not_estimable", "reason": "requires_at_least_two_reference_replicas",
        }
    rows = []
    for left_index in range(len(references)):
        for right_index in range(left_index + 1, len(references)):
            left, right = references[left_index], references[right_index]
            row = {"case_id": case_id, "reference_i": left_index, "reference_j": right_index}
            row.update(paired_metrics(left, right, masks))
            row.update(rmsf_metrics(left.positions, right, masks))
            pca, _, _ = pca_metrics(left.positions, right, masks, int(analysis.get("pca_components", 3)))
            row.update(pca)
            contact, _, _ = contact_metrics(
                left.positions, right, masks, float(analysis.get("contact_cutoff_A", 4.5)),
                float(analysis.get("contact_frequency_threshold", 0.10)),
            )
            row.update(contact)
            if enabled.get("slow_modes"):
                slow, _, slow_status = slow_state_metrics(left, right, masks, analysis)
                row.update(slow)
                row["kinetics_status"] = slow_status["status"]
            rows.append(row)
    return pd.DataFrame(rows), {
        "case_id": case_id, "model": "*", "replica": -1, "module": "reference_baseline",
        "status": "ok", "reason": "",
    }


def _evaluate_replica(
    case_id: str,
    model: str,
    replica: int,
    reference_primary: Trajectory,
    reference_pool: Trajectory,
    reference_replicas: list[Trajectory],
    prediction: Trajectory,
    masks,
    resolved_profile: dict,
) -> tuple[dict, dict[str, pd.DataFrame], list[dict]]:
    analysis = resolved_profile["analysis"]
    enabled = resolved_profile["metrics"]
    scalar: dict = {"case_id": case_id, "model": model, "replica": replica, "source": prediction.source}
    frames = pd.DataFrame({"time_ns": prediction.time_ns})
    curves = pd.DataFrame({"time_ns": prediction.time_ns})
    tables: dict[str, pd.DataFrame] = {}
    statuses: list[dict] = []

    def status(module: str, state: str, reason: str = "") -> None:
        statuses.append(
            {"case_id": case_id, "model": model, "replica": replica, "module": module, "status": state, "reason": reason}
        )

    if enabled.get("physical"):
        values, physical_frames = physical_metrics(
            reference_primary, prediction, masks, analysis.get("physical", {}) or {}
        )
        scalar.update(values)
        frames = physical_frames
        completeness = bool(values.get("physical_definition_complete", 0))
        status("physical", "ok" if completeness else "partial", f"bond_source={prediction.atoms.bond_source}")
    else:
        physical_frames = pd.DataFrame({"time_ns": prediction.time_ns, "physical_ok": 1})

    if enabled.get("paired_short"):
        scalar.update(paired_metrics(reference_primary, prediction, masks))
        tables["paired_curves"] = paired_curve(reference_primary, prediction, masks)
        status("paired_short", "ok")

    if enabled.get("motion"):
        motion, motion_curves = motion_metrics(
            reference_primary,
            prediction,
            masks,
            float(analysis.get("ligand_stay_cutoff_A", 5.0)),
        )
        scalar.update(motion)
        curves = motion_curves
        status("motion", "ok")
    else:
        motion_curves = curves

    if enabled.get("rmsf"):
        scalar.update(rmsf_metrics(reference_pool.positions, prediction, masks))
        tables["rmsf_profiles"] = rmsf_profile(reference_pool.positions, prediction, masks)
        status("rmsf", "ok")

    ref_projection = pred_projection = None
    if enabled.get("pca"):
        pca, _, pred_projection = pca_metrics(
            reference_pool.positions, prediction, masks, int(analysis.get("pca_components", 3))
        )
        scalar.update(pca)
        # Use the primary continuous reference for temporal metrics; pooled replicas have boundaries.
        from .metrics import fit_pca

        mean, components = fit_pca(
            reference_pool.positions, masks.backbone, int(analysis.get("pca_components", 3))
        )
        ref_projection = project_pca(reference_primary.positions, masks.backbone, mean, components)
        projection_rows = []
        for kind, trajectory, projection in (
            ("reference", reference_primary, ref_projection),
            ("prediction", prediction, pred_projection),
        ):
            for time_ns, values_row in zip(trajectory.time_ns, projection):
                row = {"source_kind": kind, "time_ns": time_ns}
                row.update({f"pc{i + 1}": value for i, value in enumerate(values_row)})
                projection_rows.append(row)
        tables["pca_projections"] = pd.DataFrame(projection_rows)
        status("pca", "ok")

    ref_contact_counts = pred_contact_counts = None
    if enabled.get("contacts"):
        contact, _, pred_contact_counts = contact_metrics(
            reference_pool.positions,
            prediction,
            masks,
            float(analysis.get("contact_cutoff_A", 4.5)),
            float(analysis.get("contact_frequency_threshold", 0.10)),
        )
        scalar.update(contact)
        tables["contact_profiles"] = contact_profile(
            reference_pool.positions, prediction, masks,
            float(analysis.get("contact_cutoff_A", 4.5)),
            float(analysis.get("contact_frequency_threshold", 0.10)),
        )
        _, ref_contact_counts = contact_frequency(
            reference_primary.positions, masks, float(analysis.get("contact_cutoff_A", 4.5))
        )
        status("contacts", "ok")

    if enabled.get("acf"):
        max_lag = int(analysis.get("acf_max_lag_frames", 100))
        acf_tables = []
        dt_ns = float(prediction.time_ns[1] - prediction.time_ns[0]) if len(prediction.time_ns) > 1 else float("nan")
        if ref_projection is not None and pred_projection is not None:
            scalar["acf_pca1_rmse"] = acf_rmse(ref_projection[:, 0], pred_projection[:, 0], max_lag)
            acf_tables.append(acf_curve(ref_projection[:, 0], pred_projection[:, 0], max_lag, dt_ns, "pca1"))
        if ref_contact_counts is not None and pred_contact_counts is not None:
            scalar["acf_contact_count_rmse"] = acf_rmse(ref_contact_counts, pred_contact_counts, max_lag)
            acf_tables.append(acf_curve(ref_contact_counts, pred_contact_counts, max_lag, dt_ns, "contact_count"))
        if acf_tables:
            tables["acf_curves"] = pd.concat(acf_tables, ignore_index=True)
        status("acf", "ok" if "acf_pca1_rmse" in scalar and "acf_contact_count_rmse" in scalar else "partial")

    windows = pd.DataFrame()
    configured_windows = list(analysis.get("windows_ns", []) or [])
    if enabled.get("long_windows") and configured_windows and enabled.get("motion"):
        windows = window_metrics(
            reference_primary, prediction, physical_frames, motion_curves, configured_windows
        )
        status("long_windows", "ok" if not windows.empty else "not_estimable", "no_eligible_windows" if windows.empty else "")

    tica_table = pd.DataFrame()
    if enabled.get("slow_modes"):
        values, advanced_tables, advanced_status = slow_state_metrics(reference_primary, prediction, masks, analysis)
        scalar.update(values)
        tables.update(advanced_tables)
        tica_table = advanced_tables.get("tica_projections", pd.DataFrame())
        status("slow_modes_states_msm_fes", advanced_status["status"], advanced_status.get("reason", ""))

    if enabled.get("structural"):
        values, structural_tables, structural_status = structural_metrics(reference_primary, prediction, masks, analysis)
        scalar.update(values)
        tables.update(structural_tables)
        status("torsions_rg_sasa_secondary_structure", structural_status["status"], structural_status.get("reason", ""))

    if enabled.get("chemical_interactions"):
        values, interaction_tables, interaction_status = interaction_metrics(reference_primary, prediction, masks, analysis)
        scalar.update(values)
        tables.update(interaction_tables)
        status("chemical_interactions", interaction_status["status"], interaction_status.get("reason", ""))

    if enabled.get("solvent"):
        values, solvent_tables, solvent_status = solvent_metrics(reference_primary, prediction, masks, analysis)
        scalar.update(values)
        tables.update(solvent_tables)
        status("water_bridges_solvent", solvent_status["status"], solvent_status.get("reason", ""))

    if enabled.get("thermodynamics"):
        values, observable_tables, observable_status = thermodynamic_metrics(reference_primary, prediction)
        scalar.update(values)
        tables.update(observable_tables)
        status("energy_temperature_pressure", observable_status["status"], observable_status.get("reason", ""))

    if enabled.get("efficiency"):
        values, efficiency_table, efficiency_status = efficiency_metrics(reference_primary, prediction)
        scalar.update(values)
        tables["efficiency"] = efficiency_table
        status("efficiency", efficiency_status["status"], efficiency_status.get("reason", ""))

    if enabled.get("reference_baseline"):
        trust, trust_table, trust_status = replica_normalized_trust_metrics(
            reference_replicas, prediction, masks,
            float(analysis.get("trust_threshold_ratio", 2.0)),
            int(analysis.get("trust_persistence_frames", 5)),
        )
        scalar.update(trust)
        if not trust_table.empty:
            tables["trust_curves"] = trust_table
        status("replica_normalized_trust", trust_status["status"], trust_status.get("reason", ""))

    if enabled.get("advanced_windows") and configured_windows:
        advanced_windows = windowed_dynamics_metrics(
            reference_primary, prediction, masks, configured_windows, analysis, tica_table
        )
        tables["window_advanced"] = advanced_windows
        status("advanced_windows", "ok" if not advanced_windows.empty else "not_estimable")

    if enabled.get("uncertainty"):
        uncertainty_rows = []
        block = int(analysis.get("bootstrap_block_frames", 20))
        n_bootstrap = int(analysis.get("bootstrap_samples", 500))
        confidence = float(analysis.get("bootstrap_confidence", 0.95))
        seed = int(analysis.get("random_seed", 20260903)) + replica
        series = {
            "physical_valid_frame_rate": physical_frames["physical_ok"].to_numpy(float),
            "bb_rmsd0_pred_mean_A": motion_curves["pred_bb_rmsd0_A"].to_numpy(float),
            "ligand_rmsd0_pred_mean_A": motion_curves["pred_ligand_rmsd0_A"].to_numpy(float),
        }
        structural_table = tables.get("structural_timeseries", pd.DataFrame())
        if not structural_table.empty:
            predicted_structural = structural_table[structural_table["source_kind"] == "prediction"]
            for column in ("protein_rg_A", "ligand_rg_A", "protein_sasa_A2", "ligand_sasa_A2"):
                if column in predicted_structural:
                    series[f"{column}_mean"] = predicted_structural[column].to_numpy(float)
        interaction_table = tables.get("interaction_timeseries", pd.DataFrame())
        if not interaction_table.empty:
            predicted_interactions = interaction_table[interaction_table["source_kind"] == "prediction"]
            for column in (
                "hbond_proxy_count", "salt_bridge_proxy_count", "hydrophobic_contact_count", "pi_contact_count"
            ):
                if column in predicted_interactions:
                    series[f"{column}_mean"] = predicted_interactions[column].to_numpy(float)
        for metric, values in series.items():
            estimate, lower, upper = moving_block_bootstrap_mean(values, block, n_bootstrap, seed, confidence)
            uncertainty_rows.append(
                {"metric": metric, "estimate": estimate, "ci_lower": lower, "ci_upper": upper,
                 "confidence": confidence, "block_frames": block, "n_bootstrap": n_bootstrap, "seed": seed}
            )
        tables["case_uncertainty"] = pd.DataFrame(uncertainty_rows)
        status("case_uncertainty", "ok")

    tables["frame_metrics"] = frames
    tables["motion_curves"] = curves
    tables["window_metrics"] = windows
    tables = {name: _tag(table, case_id, model, replica) for name, table in tables.items() if not table.empty}
    return scalar, tables, statuses


def evaluate_protocol(
    protocol_path: str | Path,
    out_dir: str | Path,
    profile_override: str | None = None,
) -> Path:
    protocol = load_protocol(protocol_path, profile_override=profile_override, check_paths=True)
    protocol_file = Path(protocol["_protocol_path"])
    base = protocol_file.parent
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.resolved.yaml").write_text(
        yaml.safe_dump(public_protocol(protocol), sort_keys=False), encoding="utf-8"
    )
    resolved_profile = protocol["resolved_profile"]
    selection = protocol.get("selections", {}) or {}
    ligand_resname = str(selection.get("ligand_resname", "MOL"))
    alignment = str(selection.get("alignment", "protein_backbone"))
    runtime_table, runtime_config = _runtime_lookup(resolved_profile["analysis"], base)

    replica_rows: list[dict] = []
    output_tables: dict[str, list[pd.DataFrame]] = {}
    module_status_rows: list[dict] = []
    baseline_tables: list[pd.DataFrame] = []
    failures: list[dict] = []

    for case in protocol["cases"]:
        case_id = str(case["id"])
        topology = resolve_path(str(case["topology"]), base)
        try:
            references = _load_group(case["reference"], topology, base, resolved_profile["analysis"])
            references = [
                _attach_runtime(item, runtime_table, runtime_config, case_id, "reference", i, "reference")
                for i, item in enumerate(references)
            ]
            masks = build_masks(references[0].atoms, ligand_resname, alignment)
            anchor = references[0].positions[0]
            aligned_refs = []
            for reference in references:
                validate_atom_mapping(references[0], reference)
                aligned_refs.append(align_trajectory(reference, anchor, masks.alignment))
            reference_primary = aligned_refs[0]
            reference_pool = _pool(aligned_refs)
            if resolved_profile["metrics"].get("reference_baseline"):
                baseline, baseline_status = _reference_baselines(
                    case_id, aligned_refs, masks, resolved_profile["analysis"], resolved_profile["metrics"]
                )
                if not baseline.empty:
                    baseline_tables.append(baseline)
                module_status_rows.append(baseline_status)
        except Exception as exc:
            failures.append({"case_id": case_id, "model": "*", "replica": -1, "stage": "reference", "error": str(exc)})
            continue

        for prediction_group in case["predictions"]:
            model = str(prediction_group["name"])
            try:
                predictions = _load_group(prediction_group, topology, base, resolved_profile["analysis"])
                predictions = [
                    _attach_runtime(item, runtime_table, runtime_config, case_id, model, i, "prediction")
                    for i, item in enumerate(predictions)
                ]
            except Exception as exc:
                failures.append({"case_id": case_id, "model": model, "replica": -1, "stage": "load", "error": str(exc)})
                continue
            for replica, prediction in enumerate(predictions):
                try:
                    validate_atom_mapping(reference_primary, prediction)
                    aligned_prediction = align_trajectory(prediction, anchor, masks.alignment)
                    scalar, tables, statuses = _evaluate_replica(
                        case_id,
                        model,
                        replica,
                        reference_primary,
                        reference_pool,
                        aligned_refs,
                        aligned_prediction,
                        masks,
                        resolved_profile,
                    )
                    replica_rows.append(scalar)
                    for name, table in tables.items():
                        output_tables.setdefault(name, []).append(table)
                    module_status_rows.extend(statuses)
                except Exception as exc:
                    failures.append({"case_id": case_id, "model": model, "replica": replica, "stage": "evaluate", "error": str(exc)})

    per_replica = pd.DataFrame(replica_rows)
    if per_replica.empty:
        failure_df = pd.DataFrame(failures)
        failure_df.to_csv(out / "failures.csv", index=False)
        raise RuntimeError(f"no trajectories evaluated successfully; see {out / 'failures.csv'}")
    per_case = pd.DataFrame(
        [_numeric_median_frame(group) for _, group in per_replica.groupby(["case_id", "model"], sort=True)]
    )
    combined_tables = {
        name: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        for name, tables in output_tables.items()
    }
    windows = combined_tables.get("window_metrics", pd.DataFrame())
    failure_df = pd.DataFrame(failures, columns=["case_id", "model", "replica", "stage", "error"])
    summary = summarize_by_model(per_case)
    bootstrap = pd.DataFrame()
    if resolved_profile["metrics"].get("uncertainty"):
        bootstrap = bootstrap_case_medians(
            per_case,
            [column for column in per_case.columns if column not in {"case_id", "model", "n_prediction_replicas"}],
            int(resolved_profile["analysis"].get("bootstrap_samples", 500)),
            int(resolved_profile["analysis"].get("random_seed", 20260903)),
            float(resolved_profile["analysis"].get("bootstrap_confidence", 0.95)),
        )
    module_status = pd.DataFrame(
        module_status_rows, columns=["case_id", "model", "replica", "module", "status", "reason"]
    )

    per_replica.to_csv(out / "per_replica.csv", index=False)
    id_columns = [c for c in ("case_id", "model", "replica", "source") if c in per_replica.columns]
    value_columns = [c for c in per_replica.columns if c not in id_columns]
    metrics_long = per_replica.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    metrics_long.to_csv(out / "metrics_long.csv", index=False)
    per_case.to_csv(out / "per_case.csv", index=False)
    summary.to_csv(out / "summary_by_model.csv", index=False)
    failure_df.to_csv(out / "failures.csv", index=False)
    module_status.to_csv(out / "module_status.csv", index=False)
    for name, table in combined_tables.items():
        if not table.empty:
            table.to_csv(out / f"{name}.csv", index=False)
    if baseline_tables:
        pd.concat(baseline_tables, ignore_index=True).to_csv(out / "reference_baseline.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(out / "bootstrap_ci.csv", index=False)
    metric_columns = [column for column in per_replica.columns if column not in {"case_id", "model", "replica", "source"}]
    coverage_rows = []
    for metric in metric_columns:
        values = pd.to_numeric(per_replica[metric], errors="coerce")
        coverage_rows.append(
            {"metric": metric, "n_valid": int(values.notna().sum()), "n_total": len(values),
             "coverage": float(values.notna().mean())}
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "metric_coverage.csv", index=False)

    manifest = {
        "software": "complexmd-eval",
        "version": __version__,
        "protocol_hash": protocol["_protocol_hash"],
        "profile": protocol["profile"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "n_replica_rows": len(per_replica),
        "n_case_rows": len(per_case),
        "n_failures": len(failure_df),
        "n_scalar_metrics": len(metric_columns),
        "n_fully_covered_scalar_metrics": int(sum(row["coverage"] == 1.0 for row in coverage_rows)),
        "implemented_modules": [name for name, enabled in resolved_profile["metrics"].items() if enabled],
        "scientific_guards": [
            "thermodynamic observables require an observables CSV; coordinates alone are insufficient",
            "water metrics require explicit solvent atoms",
            "kinetic rates and MFPT are emitted only when minimum transition evidence is met",
            "reference baselines require at least two independent reference replicas",
            "hydrogen bonds and ligand salt bridges are labelled proxies without hydrogens/formal charges",
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figures(summary, windows, out)
    write_markdown_report(protocol, summary, failure_df, out, coverage, module_status)
    return out
