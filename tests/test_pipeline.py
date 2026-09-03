import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from complexmd_eval.pipeline import evaluate_protocol


ROOT = Path(__file__).parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run([sys.executable, str(ROOT / "examples" / "generate_demo_data.py")], check=True)

    def test_quick_pipeline_distinguishes_frozen_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = evaluate_protocol(ROOT / "examples" / "protocol.quick.yaml", Path(tmp) / "out")
            table = pd.read_csv(out / "per_case.csv").set_index("model")
            self.assertGreater(table.loc["calibrated", "rmsf_backbone_amplitude_ratio"], 0.5)
            self.assertLess(table.loc["frozen", "rmsf_backbone_amplitude_ratio"], 0.1)
            self.assertTrue((out / "run_manifest.json").is_file())
            self.assertTrue((out / "report.md").is_file())

    def test_full_pipeline_writes_physical_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = evaluate_protocol(ROOT / "examples" / "protocol.full.yaml", Path(tmp) / "out")
            windows = pd.read_csv(out / "window_metrics.csv")
            self.assertEqual(set(windows["window_end_ns"]), {5.0, 10.0, 20.0, 29.5})
            self.assertTrue((out / "figures" / "window_overmove_bb.png").is_file())
            self.assertTrue((out / "metric_coverage.csv").is_file())
            self.assertTrue((out / "module_status.csv").is_file())
            self.assertTrue((out / "tica_projections.csv").is_file())
            self.assertTrue((out / "state_transitions.csv").is_file())
            self.assertTrue((out / "window_advanced.csv").is_file())
            self.assertTrue((out / "timebase_audit.csv").is_file())
            self.assertTrue((out / "parameter_resolution.csv").is_file())
            self.assertTrue((out / "analysis_resolution.json").is_file())
            frame_metrics = pd.read_csv(out / "frame_metrics.csv")
            self.assertIn("physical_soft_ok", frame_metrics)
            per_case = pd.read_csv(out / "per_case.csv")
            self.assertTrue((per_case["physical_soft_valid_frame_rate"] >= per_case["physical_valid_frame_rate"]).all())

    def test_pipeline_harmonizes_different_sampling_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / "different-dt.yaml"
            protocol.write_text(
                f"""
protocol_version: "1.0"
suite: different-dt
profile: screen
selections: {{ligand_resname: MOL, alignment: protein_backbone}}
analysis: {{comparison_horizon: common}}
cases:
  - id: DEMO
    topology: self_describing.npz
    reference:
      dt_ns: 0.5
      trajectories: [{(ROOT / 'examples/demo_data/reference.npz').as_posix()}]
    predictions:
      - name: calibrated
        dt_ns: 1.0
        trajectories: [{(ROOT / 'examples/demo_data/prediction_good.npz').as_posix()}]
""",
                encoding="utf-8",
            )
            out = evaluate_protocol(protocol, Path(tmp) / "out")
            audit = pd.read_csv(out / "timebase_audit.csv")
            self.assertEqual(audit["resample_factor"].tolist(), [2, 1])
            result = pd.read_csv(out / "per_replica.csv").iloc[0]
            self.assertEqual(result["common_dt_ns"], 1.0)
            self.assertEqual(result["resolved_common_horizon_ns"], 29.0)


if __name__ == "__main__":
    unittest.main()
