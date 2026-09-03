"""Data models used internally by the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AtomTable:
    names: np.ndarray
    resnames: np.ndarray
    resids: np.ndarray
    elements: np.ndarray
    bonds: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    chainids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="U16"))
    charges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    bond_source: str = "unknown"

    def __post_init__(self) -> None:
        n = len(self.names)
        if any(len(x) != n for x in (self.resnames, self.resids, self.elements)):
            raise ValueError("atom metadata arrays must have the same length")
        if len(self.chainids) not in (0, n):
            raise ValueError("chainids must be empty or match atom count")
        if len(self.charges) not in (0, n):
            raise ValueError("charges must be empty or match atom count")
        if self.bonds.ndim != 2 or self.bonds.shape[1:] != (2,):
            raise ValueError("bonds must have shape (n_bonds, 2)")


@dataclass(frozen=True)
class Trajectory:
    positions: np.ndarray
    time_ns: np.ndarray
    atoms: AtomTable
    source: str = ""
    topology_path: str = ""
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape (frames, atoms, 3)")
        if self.positions.shape[0] != len(self.time_ns):
            raise ValueError("time_ns length must match frames")
        if self.positions.shape[1] != len(self.atoms.names):
            raise ValueError("atom metadata length must match positions")
        for name, values in self.observables.items():
            if len(values) != len(self.positions):
                raise ValueError(f"observable {name!r} length must match frames")


@dataclass(frozen=True)
class Masks:
    protein: np.ndarray
    backbone: np.ndarray
    ligand: np.ndarray
    alignment: np.ndarray
