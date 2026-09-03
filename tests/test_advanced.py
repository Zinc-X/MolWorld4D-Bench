import unittest

import numpy as np

from complexmd_eval.advanced import (
    bootstrap_case_medians, js_divergence, replica_normalized_trust_metrics,
    slow_state_metrics, transition_matrix,
)
from complexmd_eval.interactions import solvent_metrics
from complexmd_eval.models import AtomTable, Masks, Trajectory
from complexmd_eval.observables import efficiency_metrics, thermodynamic_metrics
from complexmd_eval.pipeline import _reference_baselines
from complexmd_eval.structural import dihedral_angles


class AdvancedMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        n_frames = 80
        time = np.arange(n_frames, dtype=float) * 0.5
        names = np.asarray(["N", "CA", "C", "N", "CA", "C", "C1", "O1"])
        resnames = np.asarray(["ALA"] * 3 + ["GLY"] * 3 + ["MOL"] * 2)
        resids = np.asarray([1, 1, 1, 2, 2, 2, 900, 900])
        elements = np.asarray(["N", "C", "C", "N", "C", "C", "C", "O"])
        bonds = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7]])
        atoms = AtomTable(names, resnames, resids, elements, bonds, np.asarray(["A"] * 8), bond_source="test")
        base = np.arange(24, dtype=float).reshape(8, 3) * 0.2
        positions = np.repeat(base[None], n_frames, axis=0)
        phase = np.linspace(0, 8 * np.pi, n_frames)
        positions[:, 1, 1] += np.sin(phase)
        positions[:, 4, 2] += 0.5 * np.cos(phase / 2)
        positions[:, 6:, 0] += 0.3 * np.sin(phase / 3)[:, None]
        self.reference = Trajectory(positions, time, atoms, "ref")
        self.prediction = Trajectory(positions + 0.01 * np.sin(phase)[:, None, None], time, atoms, "pred")
        self.masks = Masks(
            protein=np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
            backbone=np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
            ligand=np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool),
            alignment=np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        )

    def test_js_divergence_identity(self) -> None:
        self.assertAlmostEqual(js_divergence(np.asarray([1, 2]), np.asarray([1, 2])), 0.0)

    def test_zero_transition_rows_become_self_loops(self) -> None:
        matrix = transition_matrix(np.asarray([[2, 1], [0, 0]], dtype=float))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        self.assertEqual(matrix[1, 1], 1.0)

    def test_slow_state_outputs_reference_fitted_tables(self) -> None:
        metrics, tables, status = slow_state_metrics(
            self.reference,
            self.prediction,
            self.masks,
            {"tica_lag_frames": 2, "tica_input_components": 4, "tica_components": 2,
             "n_states": 3, "msm_lag_frames": [2, 4], "fes_bins": 8},
        )
        self.assertIn(status["status"], {"ok", "partial"})
        self.assertTrue(np.isfinite(metrics["tica_gaussian_w2"]))
        self.assertEqual(len(tables["tica_projections"]), 160)
        self.assertEqual(len(tables["state_occupancy"]), 3)

    def test_adaptive_state_count_is_reported(self) -> None:
        metrics, tables, status = slow_state_metrics(
            self.reference,
            self.prediction,
            self.masks,
            {
                "tica_lag_frames": 2,
                "tica_input_components": 4,
                "tica_components": 2,
                "n_states": "auto",
                "n_states_min": 2,
                "n_states_max": 5,
                "min_frames_per_state": 10,
                "msm_lag_frames": [2, 4],
                "fes_bins": 8,
            },
        )
        self.assertEqual(metrics["state_selection_auto"], 1)
        self.assertGreaterEqual(metrics["n_states_resolved"], 2)
        self.assertLessEqual(metrics["n_states_resolved"], 5)
        self.assertEqual(len(tables["state_occupancy"]), metrics["n_states_resolved"])
        self.assertIn("state_selection", status)

    def test_thermodynamic_and_efficiency_adapters(self) -> None:
        observable = {
            "potential_energy_kJ_mol": np.linspace(-10, -8, len(self.reference.positions)),
            "temperature_K": np.linspace(299, 301, len(self.reference.positions)),
        }
        reference = Trajectory(
            self.reference.positions, self.reference.time_ns, self.reference.atoms, "ref", "", observable,
            {"runtime_s": 100.0, "hardware": "cpu"},
        )
        prediction = Trajectory(
            self.prediction.positions, self.prediction.time_ns, self.prediction.atoms, "pred", "", observable,
            {"runtime_s": 10.0, "hardware": "gpu"},
        )
        thermo, _, thermo_status = thermodynamic_metrics(reference, prediction)
        efficiency, _, efficiency_status = efficiency_metrics(reference, prediction)
        self.assertEqual(thermo_status["status"], "partial")
        self.assertAlmostEqual(thermo["potential_energy_wasserstein"], 0.0)
        self.assertEqual(efficiency_status["status"], "ok")
        self.assertAlmostEqual(efficiency["speedup_vs_reference_wall"], 10.0)

    def test_reference_replica_baseline(self) -> None:
        table, status = _reference_baselines(
            "X", [self.reference, self.prediction], self.masks,
            {"pca_components": 2, "contact_cutoff_A": 4.5, "contact_frequency_threshold": 0.1},
            {"slow_modes": False},
        )
        self.assertEqual(status["status"], "ok")
        self.assertEqual(len(table), 1)
        self.assertIn("pca_gaussian_w2", table)

    def test_replica_normalized_trust_requires_multiple_references(self) -> None:
        metrics, table, status = replica_normalized_trust_metrics(
            [self.reference], self.prediction, self.masks
        )
        self.assertEqual(status["status"], "not_estimable")
        self.assertTrue(table.empty)
        self.assertTrue(np.isnan(metrics["trust_horizon_ns"]))

    def test_case_bootstrap_is_deterministic(self) -> None:
        import pandas as pd

        table = pd.DataFrame({"model": ["m"] * 4, "case_id": list("abcd"), "x": [1, 2, 3, 4]})
        first = bootstrap_case_medians(table, ["x"], 100, 7, 0.95)
        second = bootstrap_case_medians(table, ["x"], 100, 7, 0.95)
        self.assertTrue(first.equals(second))

    def test_dihedral_shape(self) -> None:
        values = dihedral_angles(self.reference.positions, np.asarray([[0, 1, 2, 3]]))
        self.assertEqual(values.shape, (len(self.reference.positions), 1))

    def test_solvent_module_with_explicit_water(self) -> None:
        names = np.append(self.reference.atoms.names, "OW")
        resnames = np.append(self.reference.atoms.resnames, "HOH")
        resids = np.append(self.reference.atoms.resids, 901)
        elements = np.append(self.reference.atoms.elements, "O")
        chains = np.append(self.reference.atoms.chainids, "W")
        atoms = AtomTable(names, resnames, resids, elements, self.reference.atoms.bonds, chains, bond_source="test")
        water = np.repeat(np.asarray([[[1.0, 1.0, 1.0]]]), len(self.reference.positions), axis=0)
        positions = np.concatenate([self.reference.positions, water], axis=1)
        trajectory = Trajectory(positions, self.reference.time_ns, atoms)
        masks = Masks(
            protein=np.append(self.masks.protein, False), backbone=np.append(self.masks.backbone, False),
            ligand=np.append(self.masks.ligand, False), alignment=np.append(self.masks.alignment, False),
        )
        metrics, tables, status = solvent_metrics(trajectory, trajectory, masks, {})
        self.assertEqual(status["status"], "partial")
        self.assertEqual(metrics["water_bridge_occupancy_error"], 0.0)
        self.assertIn("solvent_timeseries", tables)


if __name__ == "__main__":
    unittest.main()
