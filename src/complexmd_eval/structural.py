"""Protein/ligand torsions and standard structural observables."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .advanced import js_divergence
from .models import Masks, Trajectory


ELEMENT_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.90,
}

CHI1_GAMMA = {
    "ARG": "CG", "ASN": "CG", "ASP": "CG", "CYS": "SG", "GLN": "CG",
    "GLU": "CG", "HIS": "CG", "ILE": "CG1", "LEU": "CG", "LYS": "CG",
    "MET": "CG", "PHE": "CG", "PRO": "CG", "SER": "OG", "THR": "OG1",
    "TRP": "CG", "TYR": "CG", "VAL": "CG1",
}


def dihedral_angles(positions: np.ndarray, quartets: np.ndarray) -> np.ndarray:
    if len(quartets) == 0:
        return np.empty((len(positions), 0), dtype=float)
    p0 = positions[:, quartets[:, 0]]
    p1 = positions[:, quartets[:, 1]]
    p2 = positions[:, quartets[:, 2]]
    p3 = positions[:, quartets[:, 3]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def radius_of_gyration(positions: np.ndarray, mask: np.ndarray, elements: np.ndarray) -> np.ndarray:
    xyz = positions[:, mask]
    masses = np.asarray([ELEMENT_MASS.get(str(x).upper(), 12.0) for x in elements[mask]], dtype=float)
    center = np.sum(xyz * masses[None, :, None], axis=1) / masses.sum()
    squared = np.sum((xyz - center[:, None, :]) ** 2, axis=-1)
    return np.sqrt(np.sum(squared * masses[None, :], axis=1) / masses.sum())


def _residues(trajectory: Trajectory, protein_mask: np.ndarray) -> list[dict]:
    atoms = trajectory.atoms
    chains = atoms.chainids if len(atoms.chainids) else np.full(len(atoms.names), "A", dtype="U1")
    groups: dict[tuple[str, int, str], dict[str, int]] = {}
    for index in np.flatnonzero(protein_mask):
        key = (str(chains[index]), int(atoms.resids[index]), str(atoms.resnames[index]).upper())
        groups.setdefault(key, {})[str(atoms.names[index]).upper()] = int(index)
    return [
        {"chain": key[0], "resid": key[1], "resname": key[2], "atoms": value}
        for key, value in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _protein_torsions(trajectory: Trajectory, masks: Masks) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    residues = _residues(trajectory, masks.protein)
    rama = []
    rama_meta = []
    for i, residue in enumerate(residues):
        atoms = residue["atoms"]
        if i > 0 and residues[i - 1]["chain"] == residue["chain"]:
            previous = residues[i - 1]["atoms"]
            if all(name in atoms for name in ("N", "CA", "C")) and "C" in previous:
                rama.append((previous["C"], atoms["N"], atoms["CA"], atoms["C"]))
                rama_meta.append({"kind": "phi", **{k: residue[k] for k in ("chain", "resid", "resname")}})
        if i + 1 < len(residues) and residues[i + 1]["chain"] == residue["chain"]:
            following = residues[i + 1]["atoms"]
            if all(name in atoms for name in ("N", "CA", "C")) and "N" in following:
                rama.append((atoms["N"], atoms["CA"], atoms["C"], following["N"]))
                rama_meta.append({"kind": "psi", **{k: residue[k] for k in ("chain", "resid", "resname")}})
    chi = []
    chi_meta = []
    for residue in residues:
        atoms = residue["atoms"]
        gamma = CHI1_GAMMA.get(residue["resname"])
        if gamma and all(name in atoms for name in ("N", "CA", "CB", gamma)):
            chi.append((atoms["N"], atoms["CA"], atoms["CB"], atoms[gamma]))
            chi_meta.append({"kind": "chi1", **{k: residue[k] for k in ("chain", "resid", "resname")}})
    return (
        np.asarray(rama, dtype=int).reshape(-1, 4),
        np.asarray(chi, dtype=int).reshape(-1, 4),
        rama_meta + chi_meta,
    )


def _bridges(nodes: set[int], edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    discovered: dict[int, int] = {}
    low: dict[int, int] = {}
    result: set[tuple[int, int]] = set()
    counter = [0]

    def visit(node: int, parent: int | None) -> None:
        counter[0] += 1
        discovered[node] = low[node] = counter[0]
        for neighbor in adjacency[node]:
            if neighbor == parent:
                continue
            if neighbor not in discovered:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovered[node]:
                    result.add(tuple(sorted((node, neighbor))))
            else:
                low[node] = min(low[node], discovered[neighbor])

    for node in nodes:
        if node not in discovered:
            visit(node, None)
    return result


def _ligand_torsions(trajectory: Trajectory, masks: Masks, max_torsions: int) -> np.ndarray:
    ligand_nodes = set(map(int, np.flatnonzero(masks.ligand)))
    edges = [tuple(map(int, edge)) for edge in trajectory.atoms.bonds if int(edge[0]) in ligand_nodes and int(edge[1]) in ligand_nodes]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    bridge_edges = _bridges(ligand_nodes, edges)
    quartets = []
    for left, right in sorted(bridge_edges):
        left_neighbors = [x for x in adjacency[left] if x != right]
        right_neighbors = [x for x in adjacency[right] if x != left]
        if not left_neighbors or not right_neighbors:
            continue
        for outer_left in left_neighbors:
            for outer_right in right_neighbors:
                quartets.append((outer_left, left, right, outer_right))
                if len(quartets) >= max_torsions:
                    return np.asarray(quartets, dtype=int).reshape(-1, 4)
    return np.asarray(quartets, dtype=int).reshape(-1, 4)


def _hist(values: np.ndarray, bins: int = 36) -> np.ndarray:
    values = values[np.isfinite(values)]
    return np.histogram(values, bins=bins, range=(-np.pi, np.pi))[0].astype(float)


def _rotamers(values: np.ndarray) -> np.ndarray:
    degrees = (np.degrees(values) + 360.0) % 360.0
    states = np.zeros_like(degrees, dtype=int)
    states[(degrees >= 120) & (degrees < 240)] = 1
    states[degrees >= 240] = 2
    return np.bincount(states.ravel(), minlength=3).astype(float)


def _mdtraj_observables(
    trajectory: Trajectory,
    masks: Masks,
    stride: int,
) -> tuple[dict[str, np.ndarray], str | None]:
    if not trajectory.topology_path or not trajectory.topology_path.lower().endswith((".pdb", ".gro", ".prmtop", ".psf")):
        return {}, "topology_format_not_supported_by_mdtraj"
    try:
        import mdtraj as md
    except ImportError:
        return {}, "mdtraj_not_installed"
    try:
        topology = md.load_topology(trajectory.topology_path)
        if topology.n_atoms != trajectory.positions.shape[1]:
            return {}, "mdtraj_atom_count_mismatch"
        sampled = trajectory.positions[::stride] / 10.0
        mdtraj = md.Trajectory(sampled.astype(np.float32), topology)
        sasa = md.shrake_rupley(mdtraj, mode="atom") * 100.0
        dssp = md.compute_dssp(mdtraj, simplified=True)
        out = {
            "time_ns": trajectory.time_ns[::stride],
            "protein_sasa_A2": np.sum(sasa[:, masks.protein], axis=1),
            "ligand_sasa_A2": np.sum(sasa[:, masks.ligand], axis=1),
            "dssp_H_fraction": np.mean(dssp == "H", axis=1),
            "dssp_E_fraction": np.mean(dssp == "E", axis=1),
            "dssp_C_fraction": np.mean(dssp == "C", axis=1),
        }
        return out, None
    except Exception as exc:
        return {}, f"mdtraj_failed:{type(exc).__name__}:{exc}"


def structural_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    masks: Masks,
    config: dict,
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    stride = max(1, int(config.get("structural_stride", 10)))
    bins = int(config.get("torsion_bins", 36))
    ref_idx = slice(None, None, stride)
    pred_idx = slice(None, None, stride)
    ref_rg_p = radius_of_gyration(reference.positions[ref_idx], masks.protein, reference.atoms.elements)
    pred_rg_p = radius_of_gyration(prediction.positions[pred_idx], masks.protein, prediction.atoms.elements)
    ref_rg_l = radius_of_gyration(reference.positions[ref_idx], masks.ligand, reference.atoms.elements)
    pred_rg_l = radius_of_gyration(prediction.positions[pred_idx], masks.ligand, prediction.atoms.elements)
    metrics = {
        "protein_rg_ref_mean_A": float(np.mean(ref_rg_p)),
        "protein_rg_pred_mean_A": float(np.mean(pred_rg_p)),
        "protein_rg_wasserstein_A": float(wasserstein_distance(ref_rg_p, pred_rg_p)),
        "ligand_rg_ref_mean_A": float(np.mean(ref_rg_l)),
        "ligand_rg_pred_mean_A": float(np.mean(pred_rg_l)),
        "ligand_rg_wasserstein_A": float(wasserstein_distance(ref_rg_l, pred_rg_l)),
    }
    timeseries = []
    for kind, trajectory, protein_rg, ligand_rg in (
        ("reference", reference, ref_rg_p, ref_rg_l),
        ("prediction", prediction, pred_rg_p, pred_rg_l),
    ):
        for time_ns, protein_value, ligand_value in zip(trajectory.time_ns[::stride], protein_rg, ligand_rg):
            timeseries.append({"source_kind": kind, "time_ns": time_ns, "protein_rg_A": protein_value, "ligand_rg_A": ligand_value})

    rama_quartets, chi_quartets, metadata = _protein_torsions(reference, masks)
    n_rama = len(rama_quartets)
    rama_meta, chi_meta = metadata[:n_rama], metadata[n_rama:]
    ref_rama = dihedral_angles(reference.positions[ref_idx], rama_quartets)
    pred_rama = dihedral_angles(prediction.positions[pred_idx], rama_quartets)
    ref_phi = ref_rama[:, [i for i, item in enumerate(rama_meta) if item["kind"] == "phi"]]
    pred_phi = pred_rama[:, [i for i, item in enumerate(rama_meta) if item["kind"] == "phi"]]
    ref_psi = ref_rama[:, [i for i, item in enumerate(rama_meta) if item["kind"] == "psi"]]
    pred_psi = pred_rama[:, [i for i, item in enumerate(rama_meta) if item["kind"] == "psi"]]
    phi_columns = {(item["chain"], item["resid"]): i for i, item in enumerate(rama_meta) if item["kind"] == "phi"}
    psi_columns = {(item["chain"], item["resid"]): i for i, item in enumerate(rama_meta) if item["kind"] == "psi"}
    paired_residues = sorted(set(phi_columns) & set(psi_columns))
    if paired_residues:
        ref_pairs = np.concatenate(
            [np.column_stack([ref_rama[:, phi_columns[key]], ref_rama[:, psi_columns[key]]]) for key in paired_residues], axis=0
        )
        pred_pairs = np.concatenate(
            [np.column_stack([pred_rama[:, phi_columns[key]], pred_rama[:, psi_columns[key]]]) for key in paired_residues], axis=0
        )
        ref_joint = np.histogram2d(ref_pairs[:, 0], ref_pairs[:, 1], bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])[0]
        pred_joint = np.histogram2d(pred_pairs[:, 0], pred_pairs[:, 1], bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])[0]
        joint_js = js_divergence(ref_joint, pred_joint)
    else:
        joint_js = float("nan")
    metrics.update(
        {
            "ramachandran_phi_js_divergence": js_divergence(_hist(ref_phi, bins), _hist(pred_phi, bins)),
            "ramachandran_psi_js_divergence": js_divergence(_hist(ref_psi, bins), _hist(pred_psi, bins)),
            "ramachandran_joint_js_divergence": joint_js,
            "n_ramachandran_torsions": int(len(rama_quartets)),
        }
    )
    ref_chi = dihedral_angles(reference.positions[ref_idx], chi_quartets)
    pred_chi = dihedral_angles(prediction.positions[pred_idx], chi_quartets)
    metrics["chi1_rotamer_js_divergence"] = js_divergence(_rotamers(ref_chi), _rotamers(pred_chi)) if ref_chi.size else float("nan")
    metrics["n_chi1_torsions"] = int(len(chi_quartets))

    ligand_quartets = _ligand_torsions(reference, masks, int(config.get("max_ligand_torsions", 64)))
    ref_lig = dihedral_angles(reference.positions[ref_idx], ligand_quartets)
    pred_lig = dihedral_angles(prediction.positions[pred_idx], ligand_quartets)
    ligand_js = []
    torsion_rows = []
    for torsion, item in enumerate(rama_meta):
        torsion_rows.append(
            {"kind": item["kind"], "torsion": torsion, "chain": item["chain"], "resid": item["resid"],
             "resname": item["resname"], "atom_indices": "-".join(map(str, rama_quartets[torsion])),
             "js_divergence": js_divergence(_hist(ref_rama[:, torsion], bins), _hist(pred_rama[:, torsion], bins))}
        )
    for torsion, item in enumerate(chi_meta):
        torsion_rows.append(
            {"kind": "chi1", "torsion": torsion, "chain": item["chain"], "resid": item["resid"],
             "resname": item["resname"], "atom_indices": "-".join(map(str, chi_quartets[torsion])),
             "js_divergence": js_divergence(_hist(ref_chi[:, torsion], bins), _hist(pred_chi[:, torsion], bins))}
        )
    for torsion in range(len(ligand_quartets)):
        value = js_divergence(_hist(ref_lig[:, torsion], bins), _hist(pred_lig[:, torsion], bins))
        ligand_js.append(value)
        torsion_rows.append(
            {
                "kind": "ligand_graph_dihedral",
                "torsion": torsion,
                "atom_indices": "-".join(map(str, ligand_quartets[torsion])),
                "js_divergence": value,
            }
        )
    metrics["ligand_dihedral_js_divergence_mean"] = float(np.nanmean(ligand_js)) if ligand_js else float("nan")
    metrics["n_ligand_graph_dihedrals"] = int(len(ligand_quartets))

    ref_md, ref_error = _mdtraj_observables(reference, masks, stride)
    pred_md, pred_error = _mdtraj_observables(prediction, masks, stride)
    md_error = ref_error or pred_error
    if not md_error:
        for name in ("protein_sasa_A2", "ligand_sasa_A2"):
            prefix = name.replace("_A2", "")
            metrics[f"{prefix}_ref_mean_A2"] = float(np.mean(ref_md[name]))
            metrics[f"{prefix}_pred_mean_A2"] = float(np.mean(pred_md[name]))
            metrics[f"{prefix}_wasserstein_A2"] = float(wasserstein_distance(ref_md[name], pred_md[name]))
        ref_fraction = np.asarray([np.mean(ref_md[f"dssp_{x}_fraction"]) for x in "HEC"])
        pred_fraction = np.asarray([np.mean(pred_md[f"dssp_{x}_fraction"]) for x in "HEC"])
        metrics["secondary_structure_js_divergence"] = js_divergence(ref_fraction, pred_fraction)
        for label, value in zip(("helix", "sheet", "coil"), ref_fraction):
            metrics[f"dssp_{label}_ref_fraction"] = float(value)
        for label, value in zip(("helix", "sheet", "coil"), pred_fraction):
            metrics[f"dssp_{label}_pred_fraction"] = float(value)
        for kind, values in (("reference", ref_md), ("prediction", pred_md)):
            lookup = {round(float(row["time_ns"]), 9): row for row in timeseries if row["source_kind"] == kind}
            for i, time_ns in enumerate(values["time_ns"]):
                row = lookup[round(float(time_ns), 9)]
                for key in ("protein_sasa_A2", "ligand_sasa_A2", "dssp_H_fraction", "dssp_E_fraction", "dssp_C_fraction"):
                    row[key] = values[key][i]
    else:
        for name in (
            "protein_sasa_ref_mean_A2", "protein_sasa_pred_mean_A2", "protein_sasa_wasserstein_A2",
            "ligand_sasa_ref_mean_A2", "ligand_sasa_pred_mean_A2", "ligand_sasa_wasserstein_A2",
            "secondary_structure_js_divergence",
        ):
            metrics[name] = float("nan")
    status = {"status": "ok" if not md_error else "partial", "reason": md_error or ""}
    tables = {
        "structural_timeseries": pd.DataFrame(timeseries),
        "torsion_profiles": pd.DataFrame(torsion_rows),
    }
    return metrics, tables, status
