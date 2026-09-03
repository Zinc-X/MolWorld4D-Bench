"""Rigid alignment and geometric primitives."""

from __future__ import annotations

import numpy as np

from .models import Trajectory


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    x = mobile - mobile_center
    y = target - target_center
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    return rotation, mobile_center, target_center


def align_trajectory(trajectory: Trajectory, target: np.ndarray, mask: np.ndarray) -> Trajectory:
    aligned = np.empty_like(trajectory.positions, dtype=float)
    for i, frame in enumerate(trajectory.positions):
        if not np.isfinite(frame[mask]).all():
            aligned[i] = frame
            continue
        rotation, mobile_center, target_center = kabsch_transform(frame[mask], target[mask])
        aligned[i] = (frame - mobile_center) @ rotation + target_center
    return Trajectory(
        aligned,
        trajectory.time_ns.copy(),
        trajectory.atoms,
        trajectory.source,
        trajectory.topology_path,
        {key: value.copy() for key, value in trajectory.observables.items()},
        dict(trajectory.metadata),
    )


def rmsd(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = a[:, mask] - b[:, mask]
    return np.sqrt(np.nanmean(np.sum(delta * delta, axis=-1), axis=-1))


def rmsd_to_anchor(positions: np.ndarray, anchor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = positions[:, mask] - anchor[None, mask]
    return np.sqrt(np.nanmean(np.sum(delta * delta, axis=-1), axis=-1))


def rmsf(positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = positions[:, mask]
    center = np.nanmean(selected, axis=0)
    return np.sqrt(np.nanmean(np.sum((selected - center) ** 2, axis=-1), axis=0))


def bond_lengths(positions: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    if len(bonds) == 0:
        return np.empty((len(positions), 0), dtype=float)
    delta = positions[:, bonds[:, 0]] - positions[:, bonds[:, 1]]
    return np.linalg.norm(delta, axis=-1)


def angle_triplets(bonds: np.ndarray, atom_mask: np.ndarray | None = None) -> np.ndarray:
    neighbors: dict[int, set[int]] = {}
    for left, right in bonds:
        neighbors.setdefault(int(left), set()).add(int(right))
        neighbors.setdefault(int(right), set()).add(int(left))
    triplets = []
    for center, adjacent in neighbors.items():
        ordered = sorted(adjacent)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if atom_mask is None or (atom_mask[left] and atom_mask[center] and atom_mask[right]):
                    triplets.append((left, center, right))
    return np.asarray(triplets, dtype=int).reshape(-1, 3)


def angles(positions: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    if len(triplets) == 0:
        return np.empty((len(positions), 0), dtype=float)
    a = positions[:, triplets[:, 0]] - positions[:, triplets[:, 1]]
    b = positions[:, triplets[:, 2]] - positions[:, triplets[:, 1]]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(denom, 1e-12)
    return np.arccos(np.clip(cosine, -1.0, 1.0))
