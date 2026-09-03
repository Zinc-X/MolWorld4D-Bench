"""Thermodynamic-observable and standardized efficiency comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .models import Trajectory


OBSERVABLES = {
    "potential_energy_kJ_mol": "potential_energy",
    "kinetic_energy_kJ_mol": "kinetic_energy",
    "total_energy_kJ_mol": "total_energy",
    "temperature_K": "temperature",
    "pressure_bar": "pressure",
    "volume_nm3": "volume",
}


def _slope(time_ns: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time_ns) & np.isfinite(values)
    if mask.sum() < 3 or np.ptp(time_ns[mask]) <= 0:
        return float("nan")
    return float(np.polyfit(time_ns[mask], values[mask], 1)[0])


def thermodynamic_metrics(
    reference: Trajectory,
    prediction: Trajectory,
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    metrics: dict[str, float] = {}
    available = sorted(set(reference.observables) & set(prediction.observables) & set(OBSERVABLES))
    for column in OBSERVABLES:
        prefix = OBSERVABLES[column]
        if column not in available:
            metrics[f"{prefix}_ref_mean"] = float("nan")
            metrics[f"{prefix}_pred_mean"] = float("nan")
            metrics[f"{prefix}_wasserstein"] = float("nan")
            continue
        left = np.asarray(reference.observables[column], dtype=float)
        right = np.asarray(prediction.observables[column], dtype=float)
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        metrics[f"{prefix}_ref_mean"] = float(np.mean(left)) if len(left) else float("nan")
        metrics[f"{prefix}_pred_mean"] = float(np.mean(right)) if len(right) else float("nan")
        metrics[f"{prefix}_wasserstein"] = float(wasserstein_distance(left, right)) if len(left) and len(right) else float("nan")
    n_atoms = max(len(prediction.atoms.names), 1)
    for column, prefix in (("potential_energy_kJ_mol", "potential_energy"), ("total_energy_kJ_mol", "total_energy")):
        if column in reference.observables and column in prediction.observables:
            ref_slope = _slope(reference.time_ns, np.asarray(reference.observables[column])) / n_atoms
            pred_slope = _slope(prediction.time_ns, np.asarray(prediction.observables[column])) / n_atoms
        else:
            ref_slope = pred_slope = float("nan")
        metrics[f"{prefix}_drift_ref_kJ_mol_ns_atom"] = ref_slope
        metrics[f"{prefix}_drift_pred_kJ_mol_ns_atom"] = pred_slope
        metrics[f"{prefix}_drift_abs_error_kJ_mol_ns_atom"] = abs(pred_slope - ref_slope)

    rows = []
    for kind, trajectory in (("reference", reference), ("prediction", prediction)):
        if not trajectory.observables:
            continue
        for i, time_ns in enumerate(trajectory.time_ns):
            row = {"source_kind": kind, "time_ns": time_ns}
            row.update({name: values[i] for name, values in trajectory.observables.items()})
            rows.append(row)
    if not available:
        return metrics, {}, {"status": "not_estimable", "reason": "no_shared_observables_csv"}
    missing = sorted(set(OBSERVABLES) - set(available))
    return metrics, {"thermodynamic_timeseries": pd.DataFrame(rows)}, {
        "status": "ok" if not missing else "partial",
        "reason": "missing:" + ",".join(missing) if missing else "",
    }


def efficiency_metrics(reference: Trajectory, prediction: Trajectory) -> tuple[dict, pd.DataFrame, dict]:
    ref_runtime = reference.metadata.get("runtime_s")
    pred_runtime = prediction.metadata.get("runtime_s")
    ref_duration = float(reference.time_ns[-1] - reference.time_ns[0]) if len(reference.time_ns) > 1 else 0.0
    pred_duration = float(prediction.time_ns[-1] - prediction.time_ns[0]) if len(prediction.time_ns) > 1 else 0.0
    metrics = {
        "reference_runtime_s": float(ref_runtime) if ref_runtime is not None else float("nan"),
        "prediction_runtime_s": float(pred_runtime) if pred_runtime is not None else float("nan"),
        "reference_throughput_ns_per_day": ref_duration * 86400.0 / float(ref_runtime) if ref_runtime and ref_runtime > 0 else float("nan"),
        "prediction_throughput_ns_per_day": pred_duration * 86400.0 / float(pred_runtime) if pred_runtime and pred_runtime > 0 else float("nan"),
    }
    metrics["speedup_vs_reference_wall"] = (
        metrics["prediction_throughput_ns_per_day"] / metrics["reference_throughput_ns_per_day"]
        if np.isfinite(metrics["prediction_throughput_ns_per_day"]) and np.isfinite(metrics["reference_throughput_ns_per_day"])
        and metrics["reference_throughput_ns_per_day"] > 0 else float("nan")
    )
    row = {
        **metrics,
        "reference_simulated_ns": ref_duration,
        "prediction_simulated_ns": pred_duration,
        "reference_hardware": reference.metadata.get("hardware", ""),
        "prediction_hardware": prediction.metadata.get("hardware", ""),
        "reference_precision": reference.metadata.get("precision", ""),
        "prediction_precision": prediction.metadata.get("precision", ""),
        "reference_batch_size": reference.metadata.get("batch_size", np.nan),
        "prediction_batch_size": prediction.metadata.get("batch_size", np.nan),
    }
    if pred_runtime is None:
        status = {"status": "not_estimable", "reason": "prediction_runtime_s_missing"}
    elif ref_runtime is None:
        status = {"status": "partial", "reason": "reference_runtime_s_missing_speedup_unavailable"}
    else:
        status = {"status": "ok", "reason": ""}
    return metrics, pd.DataFrame([row]), status
