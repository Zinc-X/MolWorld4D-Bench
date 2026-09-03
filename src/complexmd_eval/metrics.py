"""Dependency-light metrics for protein-small-molecule trajectories."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree

from .geometry import angle_triplets, angles, bond_lengths, rmsd, rmsd_to_anchor, rmsf
from .models import Masks, Trajectory


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.nanstd(left[mask]) < 1e-12 or np.nanstd(right[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _excluded_clash_pairs(bonds: np.ndarray, resids: np.ndarray) -> set[tuple[int, int]]:
    excluded: set[tuple[int, int]] = set()
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in bonds:
        a, b = sorted((int(left), int(right)))
        excluded.add((a, b))
        neighbors[a].add(b)
        neighbors[b].add(a)
    for center, adjacent in neighbors.items():
        for left in adjacent:
            for right in adjacent:
                if left < right:
                    excluded.add((left, right))
    unique_resids = np.unique(resids)
    for resid in unique_resids:
        indices = np.flatnonzero(resids == resid)
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                excluded.add((int(left), int(right)))
    return excluded


def clash_counts(positions: np.ndarray, atoms, cutoff_A: float) -> np.ndarray:
    heavy = np.char.upper(atoms.elements.astype(str)) != "H"
    heavy_indices = np.flatnonzero(heavy)
    excluded = _excluded_clash_pairs(atoms.bonds, atoms.resids)
    counts = np.zeros(len(positions), dtype=int)
    for frame_i, frame in enumerate(positions):
        xyz = frame[heavy]
        if not np.isfinite(xyz).all():
            counts[frame_i] = -1
            continue
        local_pairs = cKDTree(xyz).query_pairs(cutoff_A)
        count = 0
        for left, right in local_pairs:
            pair = tuple(sorted((int(heavy_indices[left]), int(heavy_indices[right]))))
            if pair not in excluded:
                count += 1
        counts[frame_i] = count
    return counts


def physical_metrics(reference: Trajectory, prediction: Trajectory, masks: Masks, config: dict) -> tuple[dict, pd.DataFrame]:
    bond_rel_max = float(config.get("bond_relative_error_max", 0.20))
    angle_max = float(config.get("angle_error_max_rad", 0.35))
    clash_cutoff = float(config.get("clash_distance_A", 1.0))
    bond_fraction_min = float(config.get("bond_valid_fraction_min", 0.99))
    ligand_bond_fraction_min = float(config.get("ligand_bond_valid_fraction_min", 0.90))
    angle_fraction_min = float(config.get("angle_valid_fraction_min", 0.90))
    positions = prediction.positions
    finite = np.isfinite(positions).all(axis=(1, 2))
    frame_ok = finite.copy()
    soft_frame_ok = finite.copy()
    bond_valid_fraction = np.full(len(positions), np.nan)
    ligand_bond_valid_fraction = np.full(len(positions), np.nan)
    ligand_angle_valid_fraction = np.full(len(positions), np.nan)
    metrics: dict[str, float | int] = {
        "finite_frame_rate": float(finite.mean()),
        "n_frames": int(len(positions)),
        "n_topology_bonds": int(len(prediction.atoms.bonds)),
    }

    ref_bonds = bond_lengths(reference.positions, reference.atoms.bonds)
    pred_bonds = bond_lengths(positions, prediction.atoms.bonds)
    if pred_bonds.shape[1]:
        target = np.nanmedian(ref_bonds, axis=0)
        abs_error = np.abs(pred_bonds - target)
        rel_error = abs_error / np.maximum(target, 1e-6)
        bond_frame_ok = np.nanmax(rel_error, axis=1) <= bond_rel_max
        bond_valid_fraction = np.mean(rel_error <= bond_rel_max, axis=1)
        frame_ok &= bond_frame_ok & np.isfinite(rel_error).all(axis=1)
        soft_frame_ok &= (bond_valid_fraction >= bond_fraction_min) & np.isfinite(rel_error).all(axis=1)
        metrics["bond_mae_A"] = float(np.nanmean(abs_error))
        metrics["bond_relative_mae"] = float(np.nanmean(rel_error))
        metrics["bond_valid_frame_rate"] = float(bond_frame_ok.mean())
        metrics["bond_valid_fraction_mean"] = float(np.mean(bond_valid_fraction))
        metrics["bond_relative_error_p99"] = float(np.nanpercentile(rel_error, 99))
        ligand_bonds = masks.ligand[prediction.atoms.bonds].all(axis=1)
        metrics["n_ligand_bonds"] = int(ligand_bonds.sum())
        if ligand_bonds.any():
            lig_rel = rel_error[:, ligand_bonds]
            ligand_bond_valid_fraction = np.mean(lig_rel <= bond_rel_max, axis=1)
            soft_frame_ok &= ligand_bond_valid_fraction >= ligand_bond_fraction_min
            metrics["ligand_bond_relative_mae"] = float(np.nanmean(lig_rel))
            metrics["ligand_bond_valid_frame_rate"] = float((np.nanmax(lig_rel, axis=1) <= bond_rel_max).mean())
            metrics["ligand_bond_valid_fraction_mean"] = float(np.mean(ligand_bond_valid_fraction))
        else:
            metrics["ligand_bond_relative_mae"] = float("nan")
            metrics["ligand_bond_valid_frame_rate"] = float("nan")
            metrics["ligand_bond_valid_fraction_mean"] = float("nan")
            soft_frame_ok &= False
    else:
        metrics["bond_mae_A"] = float("nan")
        metrics["bond_relative_mae"] = float("nan")
        metrics["bond_valid_frame_rate"] = float("nan")
        metrics["bond_valid_fraction_mean"] = float("nan")
        metrics["bond_relative_error_p99"] = float("nan")
        metrics["n_ligand_bonds"] = 0
        metrics["ligand_bond_relative_mae"] = float("nan")
        metrics["ligand_bond_valid_frame_rate"] = float("nan")
        metrics["ligand_bond_valid_fraction_mean"] = float("nan")
        soft_frame_ok &= False

    ligand_triplets = angle_triplets(prediction.atoms.bonds, masks.ligand)
    metrics["n_ligand_angles"] = int(len(ligand_triplets))
    pred_angles = angles(positions, ligand_triplets)
    ref_angles = angles(reference.positions, ligand_triplets)
    if pred_angles.shape[1]:
        target_angles = np.nanmedian(ref_angles, axis=0)
        angle_error = np.abs(pred_angles - target_angles)
        angle_frame_ok = np.nanmax(angle_error, axis=1) <= angle_max
        ligand_angle_valid_fraction = np.mean(angle_error <= angle_max, axis=1)
        frame_ok &= angle_frame_ok & np.isfinite(angle_error).all(axis=1)
        soft_frame_ok &= (ligand_angle_valid_fraction >= angle_fraction_min) & np.isfinite(angle_error).all(axis=1)
        metrics["ligand_angle_mae_rad"] = float(np.nanmean(angle_error))
        metrics["ligand_angle_valid_frame_rate"] = float(angle_frame_ok.mean())
        metrics["ligand_angle_valid_fraction_mean"] = float(np.mean(ligand_angle_valid_fraction))
    else:
        metrics["ligand_angle_mae_rad"] = float("nan")
        metrics["ligand_angle_valid_frame_rate"] = float("nan")
        metrics["ligand_angle_valid_fraction_mean"] = float("nan")
        soft_frame_ok &= False

    clashes = clash_counts(positions, prediction.atoms, clash_cutoff)
    clash_free = clashes == 0
    frame_ok &= clash_free
    soft_frame_ok &= clash_free
    metrics["clash_free_frame_rate"] = float(clash_free.mean())
    metrics["mean_clashes_per_frame"] = float(np.mean(np.maximum(clashes, 0)))
    metrics["physical_valid_frame_rate"] = float(frame_ok.mean())
    metrics["physical_soft_valid_frame_rate"] = float(soft_frame_ok.mean())
    metrics["physical_definition_complete"] = int(
        len(prediction.atoms.bonds) >= max(1, len(prediction.atoms.names) // 2)
        and metrics["n_ligand_bonds"] > 0
        and metrics["n_ligand_angles"] > 0
    )
    invalid = np.flatnonzero(~frame_ok)
    metrics["first_invalid_time_ns"] = float(prediction.time_ns[invalid[0]]) if len(invalid) else float("nan")
    frames = pd.DataFrame(
        {
            "time_ns": prediction.time_ns,
            "finite": finite.astype(int),
            "clash_count": clashes,
            "bond_valid_fraction": bond_valid_fraction,
            "ligand_bond_valid_fraction": ligand_bond_valid_fraction,
            "ligand_angle_valid_fraction": ligand_angle_valid_fraction,
            "physical_ok": frame_ok.astype(int),
            "physical_soft_ok": soft_frame_ok.astype(int),
        }
    )
    return metrics, frames


def _common_indices(left_time: np.ndarray, right_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_lookup = {round(float(t), 9): i for i, t in enumerate(left_time)}
    right_lookup = {round(float(t), 9): i for i, t in enumerate(right_time)}
    common = sorted(set(left_lookup) & set(right_lookup))
    return (
        np.asarray([left_lookup[t] for t in common], dtype=int),
        np.asarray([right_lookup[t] for t in common], dtype=int),
    )


def paired_metrics(reference: Trajectory, prediction: Trajectory, masks: Masks) -> dict:
    ref_idx, pred_idx = _common_indices(reference.time_ns, prediction.time_ns)
    out: dict[str, float | int] = {"n_paired_times": int(len(ref_idx))}
    if not len(ref_idx):
        return out
    for name, mask in (("backbone", masks.backbone), ("protein", masks.protein), ("ligand", masks.ligand)):
        values = rmsd(prediction.positions[pred_idx], reference.positions[ref_idx], mask)
        out[f"paired_{name}_rmsd_A_mean"] = float(np.nanmean(values))
        out[f"paired_{name}_rmsd_A_last"] = float(values[-1])
    return out


def paired_curve(reference: Trajectory, prediction: Trajectory, masks: Masks) -> pd.DataFrame:
    ref_idx, pred_idx = _common_indices(reference.time_ns, prediction.time_ns)
    if not len(ref_idx):
        return pd.DataFrame()
    out = pd.DataFrame({"time_ns": prediction.time_ns[pred_idx]})
    for name, mask in (("backbone", masks.backbone), ("protein", masks.protein), ("ligand", masks.ligand)):
        out[f"paired_{name}_rmsd_A"] = rmsd(prediction.positions[pred_idx], reference.positions[ref_idx], mask)
    return out


def motion_metrics(reference: Trajectory, prediction: Trajectory, masks: Masks, stay_cutoff_A: float) -> tuple[dict, pd.DataFrame]:
    anchor = reference.positions[0]
    ref_bb = rmsd_to_anchor(reference.positions, anchor, masks.backbone)
    pred_bb = rmsd_to_anchor(prediction.positions, anchor, masks.backbone)
    ref_lig = rmsd_to_anchor(reference.positions, anchor, masks.ligand)
    pred_lig = rmsd_to_anchor(prediction.positions, anchor, masks.ligand)
    ref_mean = float(np.nanmean(ref_bb))
    pred_mean = float(np.nanmean(pred_bb))
    n_tail = max(1, len(pred_bb) // 5)
    first = float(np.nanmean(pred_bb[:n_tail]))
    last = float(np.nanmean(pred_bb[-n_tail:]))
    out = {
        "bb_rmsd0_ref_mean_A": ref_mean,
        "bb_rmsd0_pred_mean_A": pred_mean,
        "overmove_bb": pred_mean / ref_mean if ref_mean > 1e-8 else float("nan"),
        "bb_rmsd0_drift_last_over_first": last / max(first, 1e-8),
        "ligand_rmsd0_pred_mean_A": float(np.nanmean(pred_lig)),
        "ligand_stay_rate": float(np.mean(pred_lig < stay_cutoff_A)),
    }
    curves = pd.DataFrame(
        {
            "time_ns": prediction.time_ns,
            "pred_bb_rmsd0_A": pred_bb,
            "pred_ligand_rmsd0_A": pred_lig,
        }
    )
    # Reference values are sampled onto their own time axis in separate columns where possible.
    ref_idx, pred_idx = _common_indices(reference.time_ns, prediction.time_ns)
    curves["ref_bb_rmsd0_A"] = np.nan
    curves["ref_ligand_rmsd0_A"] = np.nan
    if len(ref_idx):
        curves.loc[pred_idx, "ref_bb_rmsd0_A"] = ref_bb[ref_idx]
        curves.loc[pred_idx, "ref_ligand_rmsd0_A"] = ref_lig[ref_idx]
    return out, curves


def rmsf_metrics(reference_positions: np.ndarray, prediction: Trajectory, masks: Masks) -> dict:
    ref = rmsf(reference_positions, masks.backbone)
    pred = rmsf(prediction.positions, masks.backbone)
    ref_mean = float(np.nanmean(ref))
    return {
        "rmsf_backbone_pearson": safe_corr(ref, pred),
        "rmsf_backbone_ref_mean_A": ref_mean,
        "rmsf_backbone_pred_mean_A": float(np.nanmean(pred)),
        "rmsf_backbone_amplitude_ratio": float(np.nanmean(pred)) / ref_mean if ref_mean > 1e-8 else float("nan"),
    }


def rmsf_profile(reference_positions: np.ndarray, prediction: Trajectory, masks: Masks) -> pd.DataFrame:
    indices = np.flatnonzero(masks.backbone)
    ref = rmsf(reference_positions, masks.backbone)
    pred = rmsf(prediction.positions, masks.backbone)
    atoms = prediction.atoms
    chainids = atoms.chainids if len(atoms.chainids) else np.full(len(atoms.names), "A", dtype="U1")
    return pd.DataFrame(
        {
            "atom_index": indices,
            "chain": chainids[indices],
            "resid": atoms.resids[indices],
            "resname": atoms.resnames[indices],
            "atom_name": atoms.names[indices],
            "reference_rmsf_A": ref,
            "prediction_rmsf_A": pred,
        }
    )


def fit_pca(reference_positions: np.ndarray, mask: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    x = reference_positions[:, mask].reshape(len(reference_positions), -1)
    mean = np.nanmean(x, axis=0)
    centered = np.nan_to_num(x - mean)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = max(1, min(int(n_components), len(vt), max(1, len(x) - 1)))
    return mean, vt[:n].T


def project_pca(positions: np.ndarray, mask: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x = positions[:, mask].reshape(len(positions), -1)
    return np.nan_to_num(x - mean) @ components


def gaussian_w2(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    mu_l, mu_r = np.mean(left, axis=0), np.mean(right, axis=0)
    cov_l = np.atleast_2d(np.cov(left, rowvar=False))
    cov_r = np.atleast_2d(np.cov(right, rowvar=False))
    root_l = sqrtm(cov_l)
    middle = sqrtm(root_l @ cov_r @ root_l)
    value = np.sum((mu_l - mu_r) ** 2) + np.trace(cov_l + cov_r - 2 * middle)
    return float(np.sqrt(max(float(np.real(value)), 0.0)))


def pca_metrics(reference_positions: np.ndarray, prediction: Trajectory, masks: Masks, n_components: int) -> tuple[dict, np.ndarray, np.ndarray]:
    mean, components = fit_pca(reference_positions, masks.backbone, n_components)
    ref_projection = project_pca(reference_positions, masks.backbone, mean, components)
    pred_projection = project_pca(prediction.positions, masks.backbone, mean, components)
    return {"pca_gaussian_w2": gaussian_w2(ref_projection, pred_projection)}, ref_projection, pred_projection


def contact_frequency(positions: np.ndarray, masks: Masks, cutoff_A: float) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    protein_indices = np.flatnonzero(masks.protein)
    ligand_indices = np.flatnonzero(masks.ligand)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    frame_counts = np.zeros(len(positions), dtype=float)
    for frame_i, frame in enumerate(positions):
        protein_xyz = frame[protein_indices]
        ligand_xyz = frame[ligand_indices]
        if not np.isfinite(protein_xyz).all() or not np.isfinite(ligand_xyz).all():
            frame_counts[frame_i] = np.nan
            continue
        tree = cKDTree(protein_xyz)
        n_contacts = 0
        for lig_local, neighbors in enumerate(tree.query_ball_point(ligand_xyz, cutoff_A)):
            for protein_local in neighbors:
                pair = (int(protein_indices[protein_local]), int(ligand_indices[lig_local]))
                counts[pair] += 1
                n_contacts += 1
        frame_counts[frame_i] = n_contacts
    denom = max(len(positions), 1)
    return {key: value / denom for key, value in counts.items()}, frame_counts


def contact_metrics(reference_positions: np.ndarray, prediction: Trajectory, masks: Masks, cutoff_A: float, frequency_threshold: float) -> tuple[dict, np.ndarray, np.ndarray]:
    ref_freq, ref_counts = contact_frequency(reference_positions, masks, cutoff_A)
    pred_freq, pred_counts = contact_frequency(prediction.positions, masks, cutoff_A)
    keys = sorted(set(ref_freq) | set(pred_freq))
    if not keys:
        return {
            "contact_frequency_pearson": float("nan"),
            "contact_frequency_rmse": float("nan"),
            "contact_frequency_f1": float("nan"),
        }, ref_counts, pred_counts
    ref = np.asarray([ref_freq.get(key, 0.0) for key in keys])
    pred = np.asarray([pred_freq.get(key, 0.0) for key in keys])
    ref_on = ref >= frequency_threshold
    pred_on = pred >= frequency_threshold
    tp = int(np.sum(ref_on & pred_on))
    fp = int(np.sum(~ref_on & pred_on))
    fn = int(np.sum(ref_on & ~pred_on))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "contact_frequency_pearson": safe_corr(ref, pred),
        "contact_frequency_rmse": float(np.sqrt(np.mean((ref - pred) ** 2))),
        "contact_frequency_f1": float(f1),
        "n_reference_contact_pairs": int(ref_on.sum()),
        "n_prediction_contact_pairs": int(pred_on.sum()),
    }, ref_counts, pred_counts


def contact_profile(
    reference_positions: np.ndarray,
    prediction: Trajectory,
    masks: Masks,
    cutoff_A: float,
    frequency_threshold: float,
) -> pd.DataFrame:
    ref_freq, _ = contact_frequency(reference_positions, masks, cutoff_A)
    pred_freq, _ = contact_frequency(prediction.positions, masks, cutoff_A)
    atoms = prediction.atoms
    rows = []
    for protein_index, ligand_index in sorted(set(ref_freq) | set(pred_freq)):
        rows.append(
            {
                "protein_atom_index": protein_index,
                "protein_resid": atoms.resids[protein_index],
                "protein_resname": atoms.resnames[protein_index],
                "protein_atom_name": atoms.names[protein_index],
                "ligand_atom_index": ligand_index,
                "ligand_atom_name": atoms.names[ligand_index],
                "reference_frequency": ref_freq.get((protein_index, ligand_index), 0.0),
                "prediction_frequency": pred_freq.get((protein_index, ligand_index), 0.0),
                "reference_stable": int(ref_freq.get((protein_index, ligand_index), 0.0) >= frequency_threshold),
                "prediction_stable": int(pred_freq.get((protein_index, ligand_index), 0.0) >= frequency_threshold),
            }
        )
    return pd.DataFrame(rows)


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.asarray([], dtype=float)
    centered = values - values.mean()
    variance = np.dot(centered, centered)
    if variance < 1e-12:
        return np.ones(min(max_lag, len(values) - 1) + 1)
    max_lag = min(int(max_lag), len(values) - 1)
    return np.asarray([np.dot(centered[: len(values) - lag], centered[lag:]) / variance for lag in range(max_lag + 1)])


def acf_rmse(reference_values: np.ndarray, prediction_values: np.ndarray, max_lag: int) -> float:
    left = autocorrelation(reference_values, max_lag)
    right = autocorrelation(prediction_values, max_lag)
    n = min(len(left), len(right))
    if n < 2:
        return float("nan")
    return float(np.sqrt(np.mean((left[:n] - right[:n]) ** 2)))


def acf_curve(reference_values: np.ndarray, prediction_values: np.ndarray, max_lag: int, dt_ns: float, observable: str) -> pd.DataFrame:
    left = autocorrelation(reference_values, max_lag)
    right = autocorrelation(prediction_values, max_lag)
    n = min(len(left), len(right))
    return pd.DataFrame(
        {
            "observable": observable,
            "lag_frames": np.arange(n, dtype=int),
            "lag_ns": np.arange(n, dtype=float) * float(dt_ns),
            "reference_acf": left[:n],
            "prediction_acf": right[:n],
        }
    )


def window_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    physical_frames: pd.DataFrame,
    motion_curves: pd.DataFrame,
    windows_ns: list[float],
) -> pd.DataFrame:
    rows = []
    for end_ns in sorted(float(x) for x in windows_ns):
        pred_mask = prediction.time_ns <= end_ns + 1e-9
        ref_mask = reference.time_ns <= end_ns + 1e-9
        if pred_mask.sum() < 2 or ref_mask.sum() < 2:
            continue
        row = {
            "window_end_ns": end_ns,
            "n_pred_frames": int(pred_mask.sum()),
            "n_ref_frames": int(ref_mask.sum()),
            "physical_valid_frame_rate": float(physical_frames.loc[pred_mask, "physical_ok"].mean()),
            "physical_soft_valid_frame_rate": float(physical_frames.loc[pred_mask, "physical_soft_ok"].mean()) if "physical_soft_ok" in physical_frames else float("nan"),
            "pred_bb_rmsd0_A_mean": float(motion_curves.loc[pred_mask, "pred_bb_rmsd0_A"].mean()),
            "pred_ligand_rmsd0_A_mean": float(motion_curves.loc[pred_mask, "pred_ligand_rmsd0_A"].mean()),
        }
        ref_values = motion_curves.loc[pred_mask, "ref_bb_rmsd0_A"].dropna()
        if len(ref_values):
            row["ref_bb_rmsd0_A_mean"] = float(ref_values.mean())
            row["overmove_bb"] = row["pred_bb_rmsd0_A_mean"] / max(row["ref_bb_rmsd0_A_mean"], 1e-8)
        rows.append(row)
    return pd.DataFrame(rows)
