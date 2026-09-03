#!/usr/bin/env python3
"""Generate a tiny deterministic protein-ligand example in portable NPZ form."""

from pathlib import Path

import numpy as np


def main() -> None:
    out = Path(__file__).resolve().parent / "demo_data"
    out.mkdir(parents=True, exist_ok=True)
    atom_names = np.asarray(["N", "CA", "C", "N", "CA", "C", "N", "CA", "C", "C1", "C2", "O1"])
    resnames = np.asarray(["ALA"] * 9 + ["MOL"] * 3)
    resids = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3, 900, 900, 900])
    elements = np.asarray(["N", "C", "C", "N", "C", "C", "N", "C", "C", "C", "C", "O"])
    bonds = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11]],
        dtype=int,
    )
    base = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.45, 0.1, 0.0], [2.85, 0.0, 0.0],
            [4.15, 0.1, 0.0], [5.55, 0.0, 0.0], [6.95, 0.1, 0.0],
            [8.25, 0.0, 0.0], [9.65, 0.1, 0.0], [11.05, 0.0, 0.0],
            [5.2, 3.2, 0.2], [6.6, 3.2, 0.2], [7.7, 3.4, 0.2],
        ]
    )
    n_frames = 60
    time = np.linspace(0, 2 * np.pi, n_frames)
    reference = np.repeat(base[None], n_frames, axis=0)
    reference[:, 0:3, 2] += 0.08 * np.sin(time)[:, None]
    reference[:, 3:6, 2] += 0.22 * np.sin(1.3 * time + 0.4)[:, None]
    reference[:, 6:9, 2] -= 0.14 * np.sin(0.7 * time + 0.8)[:, None]
    reference[:, 1, 1] += 0.04 * np.sin(2.1 * time)
    reference[:, 4, 1] += 0.09 * np.sin(1.6 * time + 0.2)
    reference[:, 7, 1] += 0.06 * np.cos(1.2 * time)
    reference[:, 9:, 1] += 0.35 * np.sin(1.7 * time)[:, None]
    rng = np.random.default_rng(7)
    good = reference + rng.normal(0, 0.003, reference.shape)
    frozen = np.repeat(reference[:1], n_frames, axis=0)

    common = dict(atom_names=atom_names, resnames=resnames, resids=resids, elements=elements, bonds=bonds)
    np.savez_compressed(out / "reference.npz", positions=reference, **common)
    np.savez_compressed(out / "prediction_good.npz", positions=good, **common)
    np.savez_compressed(out / "prediction_frozen.npz", positions=frozen, **common)
    print(out)


if __name__ == "__main__":
    main()
