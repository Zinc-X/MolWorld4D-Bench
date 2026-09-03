"""Slow-mode, state-model, free-energy, and uncertainty analyses.

All learned coordinates and state definitions are fitted on reference MD and
then applied unchanged to predictions.  This avoids giving each model its own
favourable representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .models import Masks, Trajectory
from .geometry import rmsd


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.size != right.size or left.size == 0:
        return float("nan")
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    if left.sum() <= 0 or right.sum() <= 0:
        return float("nan")
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    mask_l = left > 0
    mask_r = right > 0
    kl_l = np.sum(left[mask_l] * np.log(left[mask_l] / middle[mask_l]))
    kl_r = np.sum(right[mask_r] * np.log(right[mask_r] / middle[mask_r]))
    return float(0.5 * (kl_l + kl_r))


def _coordinate_features(trajectory: Trajectory, masks: Masks) -> np.ndarray:
    names = np.char.upper(trajectory.atoms.names.astype(str))
    ca = masks.protein & (names == "CA")
    selected = ca if ca.sum() >= 3 else masks.backbone
    return trajectory.positions[:, selected].reshape(len(trajectory.positions), -1)


@dataclass
class LinearFeatures:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(values - self.mean) @ self.components


def fit_linear_features(values: np.ndarray, n_components: int) -> LinearFeatures:
    mean = np.nanmean(values, axis=0)
    centered = np.nan_to_num(values - mean)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = max(1, min(int(n_components), len(vt), max(1, len(values) - 1)))
    return LinearFeatures(mean, vt[:n].T)


@dataclass
class TicaModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(values - self.mean) @ self.components


def fit_tica(values: np.ndarray, lag: int, n_components: int, ridge: float = 1e-6) -> TicaModel:
    values = np.asarray(values, dtype=float)
    lag = int(lag)
    if lag < 1 or len(values) <= lag + 2:
        raise ValueError("TICA requires at least lag+3 ordered frames")
    mean = np.nanmean(values, axis=0)
    centered = np.nan_to_num(values - mean)
    x0, xt = centered[:-lag], centered[lag:]
    c00 = (x0.T @ x0 + xt.T @ xt) / (2.0 * len(x0))
    ct = (x0.T @ xt + xt.T @ x0) / (2.0 * len(x0))
    scale = max(float(np.trace(c00)) / max(len(c00), 1), 1.0)
    eigenvalues, eigenvectors = eigh(ct, c00 + ridge * scale * np.eye(len(c00)))
    order = np.argsort(np.abs(eigenvalues))[::-1]
    n = max(1, min(int(n_components), len(order)))
    order = order[:n]
    return TicaModel(mean, eigenvectors[:, order], np.real(eigenvalues[order]))


def vamp2_score(values: np.ndarray, lag: int, n_components: int, ridge: float = 1e-6) -> float:
    if len(values) <= lag + 2:
        return float("nan")
    x0, xt = np.array(values[:-lag], float, copy=True), np.array(values[lag:], float, copy=True)
    x0 -= x0.mean(axis=0)
    xt -= xt.mean(axis=0)
    c00 = x0.T @ x0 / len(x0)
    c11 = xt.T @ xt / len(x0)
    c01 = x0.T @ xt / len(x0)

    def invsqrt(matrix: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(matrix + ridge * np.eye(len(matrix)))
        vals = np.maximum(vals, ridge)
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T

    koopman = invsqrt(c00) @ c01 @ invsqrt(c11)
    singular = np.linalg.svd(koopman, compute_uv=False)
    return float(np.sum(singular[: max(1, int(n_components))] ** 2))


def _kmeans(values: np.ndarray, n_states: int, seed: int, max_iter: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_states = max(2, min(int(n_states), len(values)))
    rng = np.random.default_rng(seed)
    centers = [values[rng.integers(len(values))]]
    for _ in range(1, n_states):
        dist2 = np.min(
            np.stack([np.sum((values - center) ** 2, axis=1) for center in centers]), axis=0
        )
        if dist2.sum() <= 1e-15:
            centers.append(values[rng.integers(len(values))])
        else:
            centers.append(values[rng.choice(len(values), p=dist2 / dist2.sum())])
    centers = np.asarray(centers)
    for _ in range(max_iter):
        labels = np.argmin(((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        updated = centers.copy()
        for state in range(n_states):
            members = values[labels == state]
            if len(members):
                updated[state] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _assign(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


def transition_counts(states: np.ndarray, lag: int, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    if lag < 1 or len(states) <= lag:
        return counts
    np.add.at(counts, (states[:-lag], states[lag:]), 1.0)
    return counts


def transition_matrix(counts: np.ndarray, pseudocount: float = 0.0) -> np.ndarray:
    matrix = np.asarray(counts, dtype=float) + float(pseudocount)
    totals = matrix.sum(axis=1)
    out = np.zeros_like(matrix)
    active = totals > 0
    out[active] = matrix[active] / totals[active, None]
    for state in np.flatnonzero(~active):
        out[state, state] = 1.0
    return out


def implied_timescales(matrix: np.ndarray, lag_ns: float, n: int = 5) -> np.ndarray:
    values = np.linalg.eigvals(matrix)
    values = np.sort(np.abs(np.real(values)))[::-1]
    values = values[values < 1.0 - 1e-10][:n]
    values = np.clip(values, 1e-12, 1.0 - 1e-12)
    return -float(lag_ns) / np.log(values)


def mfpt_matrix(matrix: np.ndarray) -> np.ndarray:
    n = len(matrix)
    out = np.full((n, n), np.nan)
    np.fill_diagonal(out, 0.0)
    for target in range(n):
        keep = [i for i in range(n) if i != target]
        try:
            solution = np.linalg.solve(np.eye(n - 1) - matrix[np.ix_(keep, keep)], np.ones(n - 1))
        except np.linalg.LinAlgError:
            continue
        out[keep, target] = solution
    return out


def _ck_error(states: np.ndarray, lag: int, n_states: int, multiplier: int = 2) -> float:
    if len(states) <= lag * multiplier + 2:
        return float("nan")
    short = transition_matrix(transition_counts(states, lag, n_states))
    observed = transition_matrix(transition_counts(states, lag * multiplier, n_states))
    expected = np.linalg.matrix_power(short, multiplier)
    return float(np.sqrt(np.mean((expected - observed) ** 2)))


def _fes(
    reference: np.ndarray,
    prediction: np.ndarray,
    bins: int,
    temperature_K: float,
) -> tuple[dict, pd.DataFrame]:
    n_dim = min(2, reference.shape[1])
    ref = reference[:, :n_dim]
    pred = prediction[:, :n_dim]
    if n_dim == 1:
        ref = np.column_stack([ref[:, 0], np.zeros(len(ref))])
        pred = np.column_stack([pred[:, 0], np.zeros(len(pred))])
    edges = []
    for axis in range(2):
        low, high = np.nanpercentile(ref[:, axis], [0.5, 99.5])
        if high - low < 1e-8:
            low, high = low - 0.5, high + 0.5
        edges.append(np.linspace(low, high, int(bins) + 1))
    outside = np.zeros(len(pred), dtype=bool)
    for axis in range(2):
        outside |= (pred[:, axis] < edges[axis][0]) | (pred[:, axis] > edges[axis][-1])
    epsilon = 1e-10
    ref_clip = np.column_stack([np.clip(ref[:, axis], edges[axis][0] + epsilon, edges[axis][-1] - epsilon) for axis in range(2)])
    pred_clip = np.column_stack([np.clip(pred[:, axis], edges[axis][0] + epsilon, edges[axis][-1] - epsilon) for axis in range(2)])
    ref_hist, _, _ = np.histogram2d(ref_clip[:, 0], ref_clip[:, 1], bins=edges)
    pred_hist, _, _ = np.histogram2d(pred_clip[:, 0], pred_clip[:, 1], bins=edges)
    ref_prob = ref_hist / max(ref_hist.sum(), 1.0)
    pred_prob = pred_hist / max(pred_hist.sum(), 1.0)
    eps = 1e-12
    kbt = 0.008314462618 * float(temperature_K)
    ref_energy = -kbt * np.log(ref_prob + eps)
    pred_energy = -kbt * np.log(pred_prob + eps)
    ref_energy -= np.nanmin(ref_energy)
    pred_energy -= np.nanmin(pred_energy)
    occupied = ref_prob >= max(1.0 / max(len(ref), 1), eps)
    rmse = float(np.sqrt(np.mean((ref_energy[occupied] - pred_energy[occupied]) ** 2))) if occupied.any() else float("nan")
    rows = []
    for i in range(int(bins)):
        for j in range(int(bins)):
            rows.append(
                {
                    "bin_x": i,
                    "bin_y": j,
                    "x_center": 0.5 * (edges[0][i] + edges[0][i + 1]),
                    "y_center": 0.5 * (edges[1][j] + edges[1][j + 1]),
                    "reference_probability": ref_prob[i, j],
                    "prediction_probability": pred_prob[i, j],
                    "reference_free_energy_kJ_mol": ref_energy[i, j],
                    "prediction_free_energy_kJ_mol": pred_energy[i, j],
                }
            )
    metrics = {
        "fes_js_divergence": js_divergence(ref_prob, pred_prob),
        "fes_rmse_kJ_mol": rmse,
        "fes_reference_bin_coverage": float(np.mean(pred_prob[occupied] > 0)) if occupied.any() else float("nan"),
        "fes_prediction_outside_reference_fraction": float(np.mean(outside)),
    }
    return metrics, pd.DataFrame(rows)


def slow_state_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    masks: Masks,
    config: dict,
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    """Fit a reference TICA/state model and compare an ordered prediction."""
    lag = int(config.get("tica_lag_frames", 10))
    input_components = int(config.get("tica_input_components", 10))
    tica_components = int(config.get("tica_components", 3))
    n_states = int(config.get("n_states", 6))
    seed = int(config.get("random_seed", 20260903))
    min_events = int(config.get("min_transition_events", 5))
    rare_max = int(config.get("rare_transition_max_count", 3))
    if len(reference.positions) <= lag + 3 or len(prediction.positions) <= lag + 3:
        return {}, {}, {"status": "not_estimable", "reason": "insufficient_frames_for_tica"}

    ref_raw = _coordinate_features(reference, masks)
    pred_raw = _coordinate_features(prediction, masks)
    linear = fit_linear_features(ref_raw, input_components)
    ref_input = linear.transform(ref_raw)
    pred_input = linear.transform(pred_raw)
    tica = fit_tica(ref_input, lag, tica_components)
    ref_tica = tica.transform(ref_input)
    pred_tica = tica.transform(pred_input)
    pred_tica_model = fit_tica(pred_input, lag, tica_components)

    centers = _kmeans(ref_tica[:, : min(3, ref_tica.shape[1])], n_states, seed)
    ref_states = _assign(ref_tica[:, : centers.shape[1]], centers)
    pred_states = _assign(pred_tica[:, : centers.shape[1]], centers)
    n_states = len(centers)
    ref_occ = np.bincount(ref_states, minlength=n_states).astype(float)
    pred_occ = np.bincount(pred_states, minlength=n_states).astype(float)
    ref_occ /= ref_occ.sum()
    pred_occ /= pred_occ.sum()

    ref_counts = transition_counts(ref_states, lag, n_states)
    pred_counts = transition_counts(pred_states, lag, n_states)
    ref_matrix = transition_matrix(ref_counts)
    pred_matrix = transition_matrix(pred_counts)
    lag_ns = lag * float(reference.time_ns[1] - reference.time_ns[0])
    ref_its = implied_timescales(ref_matrix, lag_ns)
    pred_its = implied_timescales(pred_matrix, lag_ns)
    n_its = min(len(ref_its), len(pred_its))
    its_error = (
        float(np.sqrt(np.mean((np.log(ref_its[:n_its]) - np.log(pred_its[:n_its])) ** 2)))
        if n_its else float("nan")
    )
    ref_mfpt = mfpt_matrix(ref_matrix) * lag_ns
    pred_mfpt = mfpt_matrix(pred_matrix) * lag_ns
    mfpt_mask = np.isfinite(ref_mfpt) & np.isfinite(pred_mfpt) & (ref_mfpt > 0) & (pred_mfpt > 0)
    mfpt_mask &= ~np.eye(n_states, dtype=bool)
    mfpt_error = (
        float(np.median(np.abs(np.log(pred_mfpt[mfpt_mask] / ref_mfpt[mfpt_mask]))))
        if mfpt_mask.any() else float("nan")
    )

    ref_edges = transition_counts(ref_states, 1, n_states)
    pred_edges = transition_counts(pred_states, 1, n_states)
    offdiag = ~np.eye(n_states, dtype=bool)
    rare = (ref_edges >= 1) & (ref_edges <= rare_max) & offdiag
    predicted = (pred_edges > 0) & offdiag
    tp = int(np.sum(rare & predicted))
    fp = int(np.sum(~rare & predicted & (ref_edges == 0)))
    fn = int(np.sum(rare & ~predicted))
    duration_ref = max(float(reference.time_ns[-1] - reference.time_ns[0]), 1e-12)
    duration_pred = max(float(prediction.time_ns[-1] - prediction.time_ns[0]), 1e-12)
    ref_event_count = int(np.sum(ref_states[1:] != ref_states[:-1]))
    pred_event_count = int(np.sum(pred_states[1:] != pred_states[:-1]))
    transition_estimable = ref_event_count >= min_events
    min_mfpt_pairs = int(config.get("min_mfpt_pairs", 1))
    mfpt_estimable = transition_estimable and int(mfpt_mask.sum()) >= min_mfpt_pairs
    ck_reference = _ck_error(ref_states, lag, n_states)
    max_ck_rmse = float(config.get("max_ck_rmse", 0.15))
    kinetics_estimable = mfpt_estimable and np.isfinite(ck_reference) and ck_reference <= max_ck_rmse

    ref_vamp = vamp2_score(ref_input, lag, tica_components)
    pred_vamp = vamp2_score(pred_input, lag, tica_components)
    metrics = {
        "tica_lag_ns": lag_ns,
        "tica1_eigenvalue_ref": float(tica.eigenvalues[0]),
        "tica1_eigenvalue_pred": float(pred_tica_model.eigenvalues[0]),
        "tica1_timescale_ref_ns": float(-lag_ns / np.log(np.clip(abs(tica.eigenvalues[0]), 1e-12, 1 - 1e-12))),
        "tica1_timescale_pred_ns": float(-lag_ns / np.log(np.clip(abs(pred_tica_model.eigenvalues[0]), 1e-12, 1 - 1e-12))),
        "vamp2_ref": ref_vamp,
        "vamp2_pred": pred_vamp,
        "vamp2_ratio": pred_vamp / ref_vamp if ref_vamp > 1e-12 else float("nan"),
        "tica_gaussian_w2": float("nan"),
        "state_occupancy_js_divergence": js_divergence(ref_occ, pred_occ),
        "state_coverage": float(np.mean(pred_occ[ref_occ > 0] > 0)),
        "transition_matrix_rmse": float(np.sqrt(np.mean((ref_matrix - pred_matrix) ** 2))),
        "msm_self_transition_ref_mean": float(np.mean(np.diag(ref_matrix))),
        "msm_self_transition_pred_mean": float(np.mean(np.diag(pred_matrix))),
        "ck_rmse_ref": ck_reference,
        "ck_rmse_pred": _ck_error(pred_states, lag, n_states),
        "implied_timescale_log_rmse": its_error,
        "mfpt_log_median_abs_error": mfpt_error if kinetics_estimable else float("nan"),
        "n_mfpt_pairs": int(mfpt_mask.sum()),
        "reference_transition_events": ref_event_count,
        "prediction_transition_events": pred_event_count,
        "transition_rate_ref_per_ns": ref_event_count / duration_ref,
        "transition_rate_pred_per_ns": pred_event_count / duration_pred,
        "transition_rate_log_error": (
            float(abs(np.log((pred_event_count / duration_pred) / (ref_event_count / duration_ref))))
            if transition_estimable and pred_event_count > 0 else float("nan")
        ),
        "rare_transition_recall": tp / max(tp + fn, 1) if rare.any() else float("nan"),
        "rare_transition_precision": tp / max(tp + fp, 1) if rare.any() else float("nan"),
        "n_reference_rare_transition_types": int(rare.sum()),
        "transition_statistics_estimable": int(transition_estimable),
        "mfpt_estimable": int(mfpt_estimable),
        "kinetics_estimable": int(kinetics_estimable),
    }
    from .metrics import gaussian_w2

    metrics["tica_gaussian_w2"] = gaussian_w2(ref_tica, pred_tica)
    fes_metrics, fes = _fes(
        ref_tica,
        pred_tica,
        int(config.get("fes_bins", 24)),
        float(config.get("temperature_K", 300.0)),
    )
    metrics.update(fes_metrics)

    projection_rows = []
    for source, trajectory, projection, states in (
        ("reference", reference, ref_tica, ref_states),
        ("prediction", prediction, pred_tica, pred_states),
    ):
        for frame, (time_ns, values, state) in enumerate(zip(trajectory.time_ns, projection, states)):
            row = {"source_kind": source, "frame": frame, "time_ns": time_ns, "state": int(state)}
            row.update({f"tica{i + 1}": value for i, value in enumerate(values)})
            projection_rows.append(row)
    occupancy = pd.DataFrame(
        {
            "state": np.arange(n_states),
            "reference_occupancy": ref_occ,
            "prediction_occupancy": pred_occ,
        }
    )
    transitions = []
    for i in range(n_states):
        for j in range(n_states):
            transitions.append(
                {
                    "from_state": i,
                    "to_state": j,
                    "reference_count": ref_counts[i, j],
                    "prediction_count": pred_counts[i, j],
                    "reference_probability": ref_matrix[i, j],
                    "prediction_probability": pred_matrix[i, j],
                    "reference_mfpt_ns": ref_mfpt[i, j],
                    "prediction_mfpt_ns": pred_mfpt[i, j],
                }
            )
    its_rows = []
    configured_lags = config.get("msm_lag_frames", [lag, 2 * lag, 5 * lag])
    for current_lag in sorted(set(int(x) for x in configured_lags if int(x) > 0)):
        if len(ref_states) <= current_lag + 2 or len(pred_states) <= current_lag + 2:
            continue
        current_ns = current_lag * float(reference.time_ns[1] - reference.time_ns[0])
        left = implied_timescales(transition_matrix(transition_counts(ref_states, current_lag, n_states)), current_ns)
        right = implied_timescales(transition_matrix(transition_counts(pred_states, current_lag, n_states)), current_ns)
        for mode in range(max(len(left), len(right))):
            its_rows.append(
                {
                    "lag_frames": current_lag,
                    "lag_ns": current_ns,
                    "mode": mode + 1,
                    "reference_timescale_ns": left[mode] if mode < len(left) else np.nan,
                    "prediction_timescale_ns": right[mode] if mode < len(right) else np.nan,
                }
            )
    tables = {
        "tica_projections": pd.DataFrame(projection_rows),
        "state_occupancy": occupancy,
        "state_transitions": pd.DataFrame(transitions),
        "implied_timescales": pd.DataFrame(its_rows),
        "fes": fes,
    }
    if not transition_estimable:
        reason = f"only_{ref_event_count}_reference_transition_events"
    elif not mfpt_estimable:
        reason = "mfpt_not_estimable_disconnected_or_undersampled_state_model"
    elif not kinetics_estimable:
        reason = f"reference_ck_rmse_exceeds_{max_ck_rmse}"
    else:
        reason = ""
    status = {"status": "ok" if kinetics_estimable else "partial", "reason": reason}
    return metrics, tables, status


def moving_block_bootstrap_mean(
    values: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan"), float("nan"), float("nan")
    block = max(1, min(int(block_length), len(values)))
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block + 1))
    samples = np.empty(int(n_bootstrap), dtype=float)
    n_blocks = int(np.ceil(len(values) / block))
    for i in range(int(n_bootstrap)):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        draw = np.concatenate([values[start : start + block] for start in chosen])[: len(values)]
        samples[i] = np.mean(draw)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.mean(values)), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def bootstrap_case_medians(
    per_case: pd.DataFrame,
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    for model, group in per_case.groupby("model", dropna=False):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values) < 2:
                continue
            draws = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                draws[i] = np.median(rng.choice(values, size=len(values), replace=True))
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": float(np.median(values)),
                    "ci_lower": float(np.quantile(draws, alpha)),
                    "ci_upper": float(np.quantile(draws, 1 - alpha)),
                    "confidence": confidence,
                    "n_cases": len(values),
                    "method": "case_bootstrap_median",
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def windowed_dynamics_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    masks: Masks,
    windows_ns: list[float],
    config: dict,
    tica_projections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate ensemble and state agreement in cumulative physical windows."""
    from .metrics import contact_metrics, gaussian_w2, rmsf_metrics

    ref_raw = _coordinate_features(reference, masks)
    pred_raw = _coordinate_features(prediction, masks)
    linear = fit_linear_features(ref_raw, int(config.get("pca_components", 3)))
    ref_pca = linear.transform(ref_raw)
    pred_pca = linear.transform(pred_raw)
    rows = []
    for end_ns in sorted(float(value) for value in windows_ns):
        ref_mask = reference.time_ns <= end_ns + 1e-9
        pred_mask = prediction.time_ns <= end_ns + 1e-9
        if ref_mask.sum() < 3 or pred_mask.sum() < 3:
            continue
        sliced_prediction = Trajectory(
            prediction.positions[pred_mask], prediction.time_ns[pred_mask], prediction.atoms,
            prediction.source, prediction.topology_path,
            {key: value[pred_mask] for key, value in prediction.observables.items()}, prediction.metadata,
        )
        rmsf_values = rmsf_metrics(reference.positions[ref_mask], sliced_prediction, masks)
        contact_values, _, _ = contact_metrics(
            reference.positions[ref_mask], sliced_prediction, masks,
            float(config.get("contact_cutoff_A", 4.5)),
            float(config.get("contact_frequency_threshold", 0.10)),
        )
        row = {
            "window_end_ns": end_ns,
            "rmsf_backbone_pearson": rmsf_values["rmsf_backbone_pearson"],
            "rmsf_backbone_amplitude_ratio": rmsf_values["rmsf_backbone_amplitude_ratio"],
            "contact_frequency_pearson": contact_values["contact_frequency_pearson"],
            "contact_frequency_f1": contact_values["contact_frequency_f1"],
            "pca_gaussian_w2": gaussian_w2(ref_pca[ref_mask], pred_pca[pred_mask]),
        }
        if tica_projections is not None and not tica_projections.empty:
            ref_t = tica_projections[(tica_projections["source_kind"] == "reference") & (tica_projections["time_ns"] <= end_ns + 1e-9)]
            pred_t = tica_projections[(tica_projections["source_kind"] == "prediction") & (tica_projections["time_ns"] <= end_ns + 1e-9)]
            states = sorted(set(ref_t["state"].astype(int)) | set(pred_t["state"].astype(int)))
            n_states = max(states) + 1 if states else 0
            if n_states:
                ref_occ = np.bincount(ref_t["state"].astype(int), minlength=n_states)
                pred_occ = np.bincount(pred_t["state"].astype(int), minlength=n_states)
                row["state_occupancy_js_divergence"] = js_divergence(ref_occ, pred_occ)
                lag = int(config.get("tica_lag_frames", 10))
                if len(ref_t) > lag and len(pred_t) > lag:
                    ref_tm = transition_matrix(transition_counts(ref_t["state"].to_numpy(int), lag, n_states))
                    pred_tm = transition_matrix(transition_counts(pred_t["state"].to_numpy(int), lag, n_states))
                    row["transition_matrix_rmse"] = float(np.sqrt(np.mean((ref_tm - pred_tm) ** 2)))
            dims = [column for column in ("tica1", "tica2") if column in ref_t and column in pred_t]
            if dims and len(ref_t) >= 3 and len(pred_t) >= 3:
                combined = pd.concat([ref_t[dims], pred_t[dims]], ignore_index=True).to_numpy(float)
                if len(dims) == 1:
                    combined = np.column_stack([combined[:, 0], np.zeros(len(combined))])
                edges = []
                for axis in range(2):
                    low, high = np.percentile(combined[:, axis], [0.5, 99.5])
                    if high - low < 1e-8:
                        low, high = low - 0.5, high + 0.5
                    edges.append(np.linspace(low, high, int(config.get("fes_bins", 24)) + 1))
                left = ref_t[dims].to_numpy(float)
                right = pred_t[dims].to_numpy(float)
                if len(dims) == 1:
                    left = np.column_stack([left[:, 0], np.zeros(len(left))])
                    right = np.column_stack([right[:, 0], np.zeros(len(right))])
                row["fes_js_divergence"] = js_divergence(
                    np.histogram2d(left[:, 0], left[:, 1], bins=edges)[0],
                    np.histogram2d(right[:, 0], right[:, 1], bins=edges)[0],
                )
        rows.append(row)
    return pd.DataFrame(rows)


def replica_normalized_trust_metrics(
    references: list[Trajectory],
    prediction: Trajectory,
    masks: Masks,
    threshold_ratio: float = 2.0,
    persistence_frames: int = 5,
) -> tuple[dict, pd.DataFrame, dict]:
    """Find when paired prediction error persistently exceeds reference-replica divergence."""
    names = ("trust_horizon_ns", "replica_normalized_error_median", "within_reference_envelope_rate")
    if len(references) < 2:
        return {name: float("nan") for name in names}, pd.DataFrame(), {
            "status": "not_estimable", "reason": "requires_at_least_two_reference_replicas"
        }
    time_keys = [set(np.round(reference.time_ns, 9)) for reference in references]
    common = sorted(set(np.round(prediction.time_ns, 9)).intersection(*time_keys))
    if len(common) < 3:
        return {name: float("nan") for name in names}, pd.DataFrame(), {
            "status": "not_estimable", "reason": "insufficient_common_physical_times"
        }
    lookups = [{round(float(t), 9): i for i, t in enumerate(item.time_ns)} for item in references]
    pred_lookup = {round(float(t), 9): i for i, t in enumerate(prediction.time_ns)}
    baseline_curves = []
    for i in range(len(references)):
        for j in range(i + 1, len(references)):
            left = references[i].positions[[lookups[i][t] for t in common]]
            right = references[j].positions[[lookups[j][t] for t in common]]
            baseline_curves.append(rmsd(left, right, masks.backbone))
    baseline = np.median(np.stack(baseline_curves), axis=0)
    pred_positions = prediction.positions[[pred_lookup[t] for t in common]]
    ref_positions = references[0].positions[[lookups[0][t] for t in common]]
    error = rmsd(pred_positions, ref_positions, masks.backbone)
    ratio = error / np.maximum(baseline, 1e-6)
    outside = ratio > float(threshold_ratio)
    persistence = max(1, int(persistence_frames))
    trust_horizon = float("nan")
    for start in range(max(0, len(outside) - persistence + 1)):
        if np.all(outside[start : start + persistence]):
            trust_horizon = float(common[start])
            break
    table = pd.DataFrame(
        {"time_ns": common, "prediction_backbone_error_A": error,
         "reference_replica_baseline_A": baseline, "normalized_error": ratio,
         "outside_envelope": outside.astype(int)}
    )
    metrics = {
        "trust_horizon_ns": trust_horizon,
        "replica_normalized_error_median": float(np.median(ratio)),
        "within_reference_envelope_rate": float(np.mean(~outside)),
    }
    return metrics, table, {"status": "ok", "reason": ""}
