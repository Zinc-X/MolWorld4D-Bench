import unittest

import numpy as np

from complexmd_eval.models import AtomTable, Trajectory
from complexmd_eval.timebase import (
    TimebaseError,
    harmonize_timebases,
    resolve_time_parameters,
    resolve_windows,
)


def trajectory(dt_ns: float, duration_ns: float, source: str) -> Trajectory:
    n_frames = int(round(duration_ns / dt_ns)) + 1
    atoms = AtomTable(
        np.asarray(["CA", "C1"]),
        np.asarray(["ALA", "MOL"]),
        np.asarray([1, 2]),
        np.asarray(["C", "C"]),
    )
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames) * dt_ns
    return Trajectory(positions, np.arange(n_frames) * dt_ns, atoms, source)


class TimebaseTests(unittest.TestCase):
    def test_commensurate_clocks_are_exactly_downsampled(self) -> None:
        reference = trajectory(0.1, 1.0, "reference")
        prediction = trajectory(0.2, 1.0, "prediction")
        refs, pred, audit, summary = harmonize_timebases(
            [reference], prediction, {"comparison_horizon": "common"}
        )
        self.assertAlmostEqual(summary["common_dt_ns"], 0.2)
        np.testing.assert_allclose(refs[0].time_ns, pred.time_ns)
        self.assertEqual(audit["resample_factor"].tolist(), [2, 1])

    def test_excessive_coarsening_is_rejected(self) -> None:
        with self.assertRaisesRegex(TimebaseError, "coarsening"):
            harmonize_timebases(
                [trajectory(0.017, 1.7, "reference")],
                trajectory(0.023, 1.702, "prediction"),
                {"max_timebase_coarsening": 20},
            )

    def test_reference_horizon_requires_prediction_coverage(self) -> None:
        with self.assertRaisesRegex(TimebaseError, "requests"):
            harmonize_timebases(
                [trajectory(0.1, 2.0, "reference")],
                trajectory(0.1, 1.0, "prediction"),
                {"comparison_horizon": "reference"},
            )

    def test_physical_parameters_record_rounding(self) -> None:
        resolved, audit = resolve_time_parameters(
            {
                "tica_lag_ns": 0.35,
                "msm_lags_ns": [0.35, 0.8],
                "acf_max_lag_ns": 0.9,
                "bootstrap_block_ns": 0.4,
            },
            0.2,
            10.0,
        )
        self.assertEqual(resolved["tica_lag_frames"], 2)
        self.assertEqual(resolved["msm_lag_frames"], [2, 4])
        row = audit[audit["parameter"] == "tica_lag_ns"].iloc[0]
        self.assertAlmostEqual(row["resolved_ns"], 0.4)
        self.assertAlmostEqual(row["rounding_error_ns"], 0.05)

    def test_automatic_fraction_windows_follow_available_duration(self) -> None:
        windows, audit = resolve_windows(
            {
                "windows_ns": "auto",
                "auto_windows": {"mode": "fractions", "fractions": [0.1, 0.5, 1.0]},
            },
            10.0,
            0.2,
        )
        self.assertEqual(windows, [1.0, 5.0, 10.0])
        self.assertEqual(len(audit), 3)


if __name__ == "__main__":
    unittest.main()
