"""Chemically classified protein-ligand and solvent interaction diagnostics.

Hydrogen bonds and ligand salt bridges are explicitly labelled as heavy-atom
proxies when hydrogens/formal charges are absent from the input topology.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from .advanced import js_divergence
from .models import Masks, Trajectory


WATER_RESNAMES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS"}
POSITIVE_PROTEIN = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
NEGATIVE_PROTEIN = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
AROMATIC_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}


def _pair_count(frame: np.ndarray, left: np.ndarray, right: np.ndarray, cutoff: float) -> int:
    if len(left) == 0 or len(right) == 0:
        return 0
    tree = cKDTree(frame[left])
    return int(sum(len(x) for x in tree.query_ball_point(frame[right], cutoff)))


def _hydrogen_bond_count(
    frame: np.ndarray,
    atoms,
    masks: Masks,
    acceptor_elements: np.ndarray,
    distance_cutoff: float,
    angle_cutoff_deg: float,
) -> float:
    elements = np.char.upper(atoms.elements.astype(str))
    donor_pairs = []
    for left, right in atoms.bonds:
        if elements[left] == "H" and elements[right] in {"N", "O", "S"}:
            donor_pairs.append((int(right), int(left)))
        elif elements[right] == "H" and elements[left] in {"N", "O", "S"}:
            donor_pairs.append((int(left), int(right)))
    if not donor_pairs:
        return float("nan")
    protein_acceptors = np.flatnonzero(masks.protein & acceptor_elements)
    ligand_acceptors = np.flatnonzero(masks.ligand & acceptor_elements)
    count = 0
    for donor, hydrogen in donor_pairs:
        acceptors = ligand_acceptors if masks.protein[donor] else protein_acceptors if masks.ligand[donor] else []
        for acceptor in acceptors:
            if np.linalg.norm(frame[donor] - frame[acceptor]) > distance_cutoff:
                continue
            first = frame[donor] - frame[hydrogen]
            second = frame[acceptor] - frame[hydrogen]
            cosine = np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            if angle >= angle_cutoff_deg:
                count += 1
    return float(count)


def _protein_rings(trajectory: Trajectory, masks: Masks) -> list[np.ndarray]:
    atoms = trajectory.atoms
    chains = atoms.chainids if len(atoms.chainids) else np.full(len(atoms.names), "A", dtype="U1")
    residues: dict[tuple[str, int, str], dict[str, int]] = defaultdict(dict)
    for index in np.flatnonzero(masks.protein):
        key = (str(chains[index]), int(atoms.resids[index]), str(atoms.resnames[index]).upper())
        residues[key][str(atoms.names[index]).upper()] = int(index)
    rings = []
    for (_, _, resname), mapping in residues.items():
        names = AROMATIC_ATOMS.get(resname)
        if names and all(name in mapping for name in names):
            rings.append(np.asarray([mapping[name] for name in names], dtype=int))
    return rings


def _ligand_rings(trajectory: Trajectory, masks: Masks) -> tuple[list[np.ndarray], str | None]:
    try:
        import networkx as nx
    except ImportError:
        return [], "networkx_not_installed"
    graph = nx.Graph()
    ligand = set(map(int, np.flatnonzero(masks.ligand)))
    graph.add_nodes_from(ligand)
    graph.add_edges_from(
        (int(a), int(b)) for a, b in trajectory.atoms.bonds if int(a) in ligand and int(b) in ligand
    )
    cycles = []
    for cycle in nx.cycle_basis(graph):
        if 5 <= len(cycle) <= 7:
            cycles.append(np.asarray(cycle, dtype=int))
    return cycles, None if cycles else "no_ligand_ring_detected"


def _ring_geometry(frame: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = frame[ring]
    center = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - center, full_matrices=False)
    return center, vt[-1]


def _pi_count(frame: np.ndarray, protein_rings: list[np.ndarray], ligand_rings: list[np.ndarray], cutoff: float) -> int:
    count = 0
    protein_geometry = [_ring_geometry(frame, ring) for ring in protein_rings]
    ligand_geometry = [_ring_geometry(frame, ring) for ring in ligand_rings]
    for p_center, p_normal in protein_geometry:
        for l_center, l_normal in ligand_geometry:
            distance = np.linalg.norm(p_center - l_center)
            angle_cos = abs(float(np.dot(p_normal, l_normal)))
            if distance <= cutoff and (angle_cos >= np.cos(np.deg2rad(35)) or angle_cos <= np.cos(np.deg2rad(55))):
                count += 1
    return count


def _interaction_series(trajectory: Trajectory, masks: Masks, config: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    atoms = trajectory.atoms
    elements = np.char.upper(atoms.elements.astype(str))
    names = np.char.upper(atoms.names.astype(str))
    resnames = np.char.upper(atoms.resnames.astype(str))
    protein_hetero = np.flatnonzero(masks.protein & np.isin(elements, ["N", "O", "S"]))
    ligand_hetero = np.flatnonzero(masks.ligand & np.isin(elements, ["N", "O", "S"]))
    acceptor_elements = np.isin(elements, ["N", "O", "S"])
    protein_positive = np.asarray(
        [i for i in np.flatnonzero(masks.protein) if (resnames[i], names[i]) in POSITIVE_PROTEIN], dtype=int
    )
    protein_negative = np.asarray(
        [i for i in np.flatnonzero(masks.protein) if (resnames[i], names[i]) in NEGATIVE_PROTEIN], dtype=int
    )
    has_hydrogens = bool(np.any(elements == "H"))
    notes = [] if has_hydrogens else ["hbond_heavy_atom_distance_proxy"]
    if len(atoms.charges) == len(atoms.names) and np.any(np.abs(atoms.charges[masks.ligand]) > 0.2):
        ligand_positive = np.flatnonzero(masks.ligand & (atoms.charges > 0.2))
        ligand_negative = np.flatnonzero(masks.ligand & (atoms.charges < -0.2))
        salt_note = "formal_or_partial_charges"
    else:
        ligand_positive = np.flatnonzero(masks.ligand & (elements == "N"))
        ligand_negative = np.flatnonzero(masks.ligand & np.isin(elements, ["O", "S"]))
        salt_note = "ligand_element_charge_proxy"
    notes.append(salt_note)
    protein_hydrophobic = np.flatnonzero(
        masks.protein & np.isin(resnames, list(HYDROPHOBIC_RESIDUES)) & np.isin(elements, ["C", "S"])
    )
    ligand_hydrophobic = np.flatnonzero(masks.ligand & np.isin(elements, ["C", "S", "F", "CL", "BR", "I"]))
    protein_rings = _protein_rings(trajectory, masks)
    ligand_rings, ring_note = _ligand_rings(trajectory, masks)
    if ring_note:
        notes.append(ring_note)
    stride = max(1, int(config.get("interaction_stride", 1)))
    indices = np.arange(0, len(trajectory.positions), stride)
    hbond, hbond_geometry, salt, salt_charge, hydrophobic, pi = [], [], [], [], [], []
    for frame in trajectory.positions[indices]:
        hbond.append(_pair_count(frame, protein_hetero, ligand_hetero, float(config.get("hbond_distance_A", 3.5))))
        hbond_geometry.append(
            _hydrogen_bond_count(
                frame, atoms, masks, acceptor_elements,
                float(config.get("hbond_distance_A", 3.5)),
                float(config.get("hbond_angle_deg", 150.0)),
            ) if has_hydrogens else np.nan
        )
        salt.append(
            _pair_count(frame, protein_positive, ligand_negative, float(config.get("salt_bridge_distance_A", 4.0)))
            + _pair_count(frame, protein_negative, ligand_positive, float(config.get("salt_bridge_distance_A", 4.0)))
        )
        salt_charge.append(salt[-1] if salt_note == "formal_or_partial_charges" else np.nan)
        hydrophobic.append(_pair_count(frame, protein_hydrophobic, ligand_hydrophobic, float(config.get("hydrophobic_distance_A", 4.5))))
        pi.append(_pi_count(frame, protein_rings, ligand_rings, float(config.get("pi_distance_A", 5.5))))
    return {
        "time_ns": trajectory.time_ns[indices],
        "hbond_proxy_count": np.asarray(hbond, dtype=float),
        "hbond_geometry_count": np.asarray(hbond_geometry, dtype=float),
        "salt_bridge_proxy_count": np.asarray(salt, dtype=float),
        "salt_bridge_charge_count": np.asarray(salt_charge, dtype=float),
        "hydrophobic_contact_count": np.asarray(hydrophobic, dtype=float),
        "pi_contact_count": np.asarray(pi, dtype=float),
    }, notes


def interaction_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    masks: Masks,
    config: dict,
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    ref, ref_notes = _interaction_series(reference, masks, config)
    pred, pred_notes = _interaction_series(prediction, masks, config)
    metrics = {}
    classes = ("hbond_proxy", "salt_bridge_proxy", "hydrophobic_contact", "pi_contact")
    for prefix in classes:
        key = f"{prefix}_count"
        left, right = ref[key], pred[key]
        metrics[f"{prefix}_ref_mean"] = float(np.mean(left))
        metrics[f"{prefix}_pred_mean"] = float(np.mean(right))
        metrics[f"{prefix}_occupancy_error"] = float(abs(np.mean(left > 0) - np.mean(right > 0)))
        metrics[f"{prefix}_count_wasserstein"] = float(wasserstein_distance(left, right))
    ref_occ = np.asarray([np.mean(ref[f"{prefix}_count"] > 0) for prefix in classes])
    pred_occ = np.asarray([np.mean(pred[f"{prefix}_count"] > 0) for prefix in classes])
    metrics["interaction_class_occupancy_js_divergence"] = js_divergence(ref_occ, pred_occ)
    for prefix in ("hbond_geometry", "salt_bridge_charge"):
        key = f"{prefix}_count"
        left, right = ref[key], pred[key]
        valid_left, valid_right = left[np.isfinite(left)], right[np.isfinite(right)]
        metrics[f"{prefix}_ref_mean"] = float(np.mean(valid_left)) if len(valid_left) else float("nan")
        metrics[f"{prefix}_pred_mean"] = float(np.mean(valid_right)) if len(valid_right) else float("nan")
        metrics[f"{prefix}_occupancy_error"] = (
            float(abs(np.mean(valid_left > 0) - np.mean(valid_right > 0))) if len(valid_left) and len(valid_right) else float("nan")
        )
        metrics[f"{prefix}_count_wasserstein"] = (
            float(wasserstein_distance(valid_left, valid_right)) if len(valid_left) and len(valid_right) else float("nan")
        )
    rows = []
    for kind, values in (("reference", ref), ("prediction", pred)):
        for i, time_ns in enumerate(values["time_ns"]):
            row = {"source_kind": kind, "time_ns": time_ns}
            row.update({key: values[key][i] for key in values if key != "time_ns"})
            rows.append(row)
    notes = sorted(set(ref_notes + pred_notes))
    return metrics, {"interaction_timeseries": pd.DataFrame(rows)}, {"status": "ok", "reason": ";".join(notes)}


def _water_series(trajectory: Trajectory, masks: Masks, config: dict) -> dict[str, np.ndarray] | None:
    atoms = trajectory.atoms
    elements = np.char.upper(atoms.elements.astype(str))
    resnames = np.char.upper(atoms.resnames.astype(str))
    water_oxygen = np.flatnonzero(np.isin(resnames, list(WATER_RESNAMES)) & (elements == "O"))
    if len(water_oxygen) == 0:
        return None
    protein_hetero = np.flatnonzero(masks.protein & np.isin(elements, ["N", "O", "S"]))
    ligand_hetero = np.flatnonzero(masks.ligand & np.isin(elements, ["N", "O", "S"]))
    ligand = np.flatnonzero(masks.ligand)
    stride = max(1, int(config.get("solvent_stride", 1)))
    indices = np.arange(0, len(trajectory.positions), stride)
    bridge_counts, shell_counts = [], []
    bridge_cutoff = float(config.get("water_bridge_distance_A", 3.5))
    shell_cutoff = float(config.get("solvent_shell_distance_A", 5.0))
    for frame in trajectory.positions[indices]:
        water_xyz = frame[water_oxygen]
        near_protein = np.asarray([bool(x) for x in cKDTree(frame[protein_hetero]).query_ball_point(water_xyz, bridge_cutoff)])
        near_ligand = np.asarray([bool(x) for x in cKDTree(frame[ligand_hetero]).query_ball_point(water_xyz, bridge_cutoff)])
        bridge_counts.append(int(np.sum(near_protein & near_ligand)))
        shell = cKDTree(frame[ligand]).query_ball_point(water_xyz, shell_cutoff)
        shell_counts.append(int(sum(bool(x) for x in shell)))
    return {
        "time_ns": trajectory.time_ns[indices],
        "water_bridge_count": np.asarray(bridge_counts, dtype=float),
        "ligand_solvent_shell_count": np.asarray(shell_counts, dtype=float),
    }


def solvent_metrics(
    reference: Trajectory,
    prediction: Trajectory,
    masks: Masks,
    config: dict,
) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    ref = _water_series(reference, masks, config)
    pred = _water_series(prediction, masks, config)
    names = (
        "water_bridge_ref_mean", "water_bridge_pred_mean", "water_bridge_occupancy_error",
        "water_bridge_count_wasserstein", "ligand_solvent_shell_ref_mean",
        "ligand_solvent_shell_pred_mean", "ligand_solvent_shell_count_wasserstein",
    )
    if ref is None or pred is None:
        return {name: float("nan") for name in names}, {}, {"status": "not_estimable", "reason": "no_solvent_atoms"}
    metrics = {
        "water_bridge_ref_mean": float(np.mean(ref["water_bridge_count"])),
        "water_bridge_pred_mean": float(np.mean(pred["water_bridge_count"])),
        "water_bridge_occupancy_error": float(abs(np.mean(ref["water_bridge_count"] > 0) - np.mean(pred["water_bridge_count"] > 0))),
        "water_bridge_count_wasserstein": float(wasserstein_distance(ref["water_bridge_count"], pred["water_bridge_count"])),
        "ligand_solvent_shell_ref_mean": float(np.mean(ref["ligand_solvent_shell_count"])),
        "ligand_solvent_shell_pred_mean": float(np.mean(pred["ligand_solvent_shell_count"])),
        "ligand_solvent_shell_count_wasserstein": float(wasserstein_distance(ref["ligand_solvent_shell_count"], pred["ligand_solvent_shell_count"])),
    }
    rows = []
    for kind, values in (("reference", ref), ("prediction", pred)):
        for i, time_ns in enumerate(values["time_ns"]):
            rows.append({"source_kind": kind, "time_ns": time_ns, "water_bridge_count": values["water_bridge_count"][i], "ligand_solvent_shell_count": values["ligand_solvent_shell_count"][i]})
    return metrics, {"solvent_timeseries": pd.DataFrame(rows)}, {
        "status": "partial", "reason": "water_bridge_uses_heavy_atom_distance_proxy"
    }
