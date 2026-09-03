"""Trajectory readers and atom-selection helpers.

NPZ is the dependency-light exchange format. MDAnalysis is used lazily for
standard trajectory formats so the core package stays usable in small CI jobs.
Coordinates are interpreted as Angstrom; protocol dt_ns is authoritative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import AtomTable, Masks, Trajectory


AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HID", "HIE", "HIP", "CYX", "MSE",
}
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}


def _strings(data: np.lib.npyio.NpzFile, key: str, n: int, default: str) -> np.ndarray:
    if key not in data:
        return np.full(n, default, dtype="U8")
    values = np.asarray(data[key]).astype("U16")
    if len(values) != n:
        raise ValueError(f"NPZ {key} length does not match atom count")
    return values


def load_npz(path: Path, dt_ns: float, stride: int, max_frames: int | None) -> Trajectory:
    with np.load(path, allow_pickle=False) as data:
        key = "positions" if "positions" in data else "coords"
        if key not in data:
            raise ValueError(f"{path}: NPZ requires positions or coords")
        raw = np.asarray(data[key], dtype=float)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(f"{path}: positions must have shape (frames, atoms, 3)")
        stop = None if max_frames is None else stride * int(max_frames)
        positions = raw[:stop:stride].copy()
        n_atoms = positions.shape[1]
        names = _strings(data, "atom_names", n_atoms, "X")
        resnames = _strings(data, "resnames", n_atoms, "UNK")
        elements = _strings(data, "elements", n_atoms, "C")
        chainids = _strings(data, "chainids", n_atoms, "A")
        resids = np.asarray(data["resids"], dtype=int) if "resids" in data else np.arange(n_atoms)
        bonds = np.asarray(data["bonds"], dtype=int) if "bonds" in data else np.empty((0, 2), dtype=int)
        charges = np.asarray(data["charges"], dtype=float) if "charges" in data else np.empty(0, dtype=float)
    time_ns = np.arange(len(positions), dtype=float) * float(dt_ns) * stride
    atoms = AtomTable(names, resnames, resids, elements, bonds, chainids, charges, "npz")
    return Trajectory(positions, time_ns, atoms, str(path), str(path))


def _load_observables(
    path: Path, stride: int, max_frames: int | None, expected_time_ns: np.ndarray
) -> dict[str, np.ndarray]:
    table = pd.read_csv(path)
    stop = None if max_frames is None else stride * int(max_frames)
    table = table.iloc[:stop:stride].reset_index(drop=True)
    if len(table) != len(expected_time_ns):
        raise ValueError(f"{path}: observable rows ({len(table)}) do not match loaded frames ({len(expected_time_ns)})")
    if "time_ns" in table:
        observed_time = pd.to_numeric(table["time_ns"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(observed_time, expected_time_ns, rtol=0, atol=1e-8, equal_nan=False):
            raise ValueError(f"{path}: time_ns does not match protocol dt_ns/frame_stride")
    out: dict[str, np.ndarray] = {}
    for column in table.columns:
        if column == "time_ns":
            continue
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        out[str(column)] = values
    return out


def load_mdanalysis(
    path: Path,
    topology: Path,
    dt_ns: float,
    stride: int,
    max_frames: int | None,
) -> Trajectory:
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "MDAnalysis is required for XTC/DCD/TRR inputs; install complexmd-eval[trajectory]"
        ) from exc
    universe = mda.Universe(str(topology), str(path))
    names = np.asarray(universe.atoms.names).astype("U16")
    resnames = np.asarray(universe.atoms.resnames).astype("U16")
    resids = np.asarray(universe.atoms.resids, dtype=int)
    try:
        elements = np.asarray(universe.atoms.elements).astype("U4")
    except Exception:
        elements = np.asarray([str(name)[0] if str(name) else "C" for name in names], dtype="U4")
    try:
        bonds = np.asarray(universe.bonds.indices, dtype=int)
    except Exception:
        bonds = np.empty((0, 2), dtype=int)
    initial_bond_count = len(bonds)
    bond_source = "topology"
    # PDB files often contain a handful of incomplete CONECT records. Treat a
    # topology with far fewer than one bond per two heavy atoms as incomplete
    # and augment it geometrically instead of silently evaluating those records.
    if len(bonds) < max(1, len(universe.atoms) // 2):
        try:
            universe.atoms.guess_bonds()
            bonds = np.asarray(universe.bonds.indices, dtype=int)
            bond_source = "topology+guessed" if initial_bond_count else "guessed"
        except Exception:
            bond_source = "incomplete_topology"
    try:
        chainids = np.asarray(universe.atoms.chainIDs).astype("U16")
        if np.all(chainids == ""):
            raise ValueError
    except Exception:
        try:
            chainids = np.asarray(universe.atoms.segids).astype("U16")
        except Exception:
            chainids = np.full(len(names), "A", dtype="U16")
    try:
        charges = np.asarray(universe.atoms.charges, dtype=float)
    except Exception:
        charges = np.empty(0, dtype=float)
    frames: list[np.ndarray] = []
    for ts in universe.trajectory[::stride]:
        frames.append(universe.atoms.positions.astype(float, copy=True))
        if max_frames is not None and len(frames) >= max_frames:
            break
    positions = np.asarray(frames, dtype=float)
    time_ns = np.arange(len(positions), dtype=float) * float(dt_ns) * stride
    atoms = AtomTable(names, resnames, resids, elements, bonds, chainids, charges, bond_source)
    return Trajectory(positions, time_ns, atoms, str(path), str(topology))


def load_trajectory(
    path: str | Path,
    topology: str | Path,
    dt_ns: float,
    stride: int = 1,
    max_frames: int | None = None,
    observables_path: str | Path | None = None,
    metadata: dict | None = None,
) -> Trajectory:
    path = Path(path)
    if stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if path.suffix.lower() == ".npz":
        trajectory = load_npz(path, dt_ns, stride, max_frames)
    else:
        trajectory = load_mdanalysis(path, Path(topology), dt_ns, stride, max_frames)
    observables = trajectory.observables
    if observables_path is not None:
        observables = _load_observables(Path(observables_path), stride, max_frames, trajectory.time_ns)
    return Trajectory(
        trajectory.positions,
        trajectory.time_ns,
        trajectory.atoms,
        trajectory.source,
        trajectory.topology_path,
        observables,
        dict(metadata or {}),
    )


def validate_atom_mapping(reference: Trajectory, other: Trajectory) -> None:
    a, b = reference.atoms, other.atoms
    if len(a.names) != len(b.names):
        raise ValueError(f"atom-count mismatch: {len(a.names)} vs {len(b.names)}")
    for label, left, right in (
        ("atom_names", a.names, b.names),
        ("resnames", a.resnames, b.resnames),
        ("resids", a.resids, b.resids),
        ("chainids", a.chainids, b.chainids),
    ):
        if not np.array_equal(left, right):
            bad = int(np.flatnonzero(left != right)[0])
            raise ValueError(f"{label} mismatch at atom {bad}: {left[bad]!r} vs {right[bad]!r}")
    bonds_a = sorted(tuple(sorted(map(int, pair))) for pair in a.bonds)
    bonds_b = sorted(tuple(sorted(map(int, pair))) for pair in b.bonds)
    if bonds_a != bonds_b:
        raise ValueError("bond topology mismatch between reference and prediction")


def build_masks(atoms: AtomTable, ligand_resname: str, alignment: str) -> Masks:
    resnames = np.char.upper(atoms.resnames.astype(str))
    names = np.char.upper(atoms.names.astype(str))
    heavy = np.char.upper(atoms.elements.astype(str)) != "H"
    ligand = (resnames == ligand_resname.upper()) & heavy
    protein = np.isin(resnames, sorted(AMINO_ACIDS)) & heavy
    if not protein.any():
        protein = (~ligand) & heavy
    backbone = protein & np.isin(names, sorted(BACKBONE_NAMES))
    if alignment == "protein_heavy":
        alignment_mask = protein
    else:
        alignment_mask = backbone
    if ligand.sum() < 2:
        raise ValueError(f"ligand selection {ligand_resname!r} matched fewer than two atoms")
    if alignment_mask.sum() < 3:
        raise ValueError(f"alignment selection {alignment!r} matched fewer than three atoms")
    return Masks(protein=protein, backbone=backbone, ligand=ligand, alignment=alignment_mask)
