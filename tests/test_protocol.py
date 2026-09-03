from pathlib import Path
import tempfile
import unittest

from complexmd_eval.protocol import ProtocolError, load_protocol, scoped_profile
from complexmd_eval.profiles import resolve_profile


class ProtocolTests(unittest.TestCase):
    def test_full_defaults_remain_compatible_and_auto_is_opt_in(self) -> None:
        full = resolve_profile("full")["analysis"]
        self.assertEqual(full["n_states"], 6)
        self.assertEqual(full["tica_lag_frames"], 10)
        self.assertIsNone(full["tica_lag_ns"])
        self.assertEqual(full["windows_ns"], [])
        self.assertEqual(resolve_profile("kinetics")["analysis"]["n_states"], "auto")

    def test_example_protocol_validates_without_paths(self) -> None:
        path = Path(__file__).parents[1] / "examples" / "protocol.full.yaml"
        protocol = load_protocol(path, check_paths=False)
        self.assertEqual(protocol["profile"], "full")
        self.assertTrue(protocol["resolved_profile"]["metrics"]["acf"])

    def test_duplicate_case_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
protocol_version: "1.0"
suite: duplicate
cases:
  - &case
    id: X
    topology: self.npz
    reference: {dt_ns: 1.0, trajectories: [ref.npz]}
    predictions:
      - {name: m, dt_ns: 1.0, trajectories: [pred.npz]}
  - *case
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "duplicate case"):
                load_protocol(path, check_paths=False)

    def test_case_and_model_analysis_overrides_are_deep_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoped.yaml"
            path.write_text(
                """
protocol_version: "1.0"
suite: scoped
profile: kinetics
analysis:
  tica_lag_ns: 1.0
  physical: {bond_relative_error_max: 0.2, clash_distance_A: 1.0}
cases:
  - id: X
    topology: self.npz
    analysis:
      tica_lag_ns: 2.0
      physical: {clash_distance_A: 1.2}
    reference: {dt_ns: 0.1, trajectories: [ref.npz]}
    predictions:
      - name: m
        dt_ns: 0.2
        trajectories: [pred.npz]
        analysis:
          n_states: 4
          physical: {bond_relative_error_max: 0.1}
""",
                encoding="utf-8",
            )
            protocol = load_protocol(path, check_paths=False)
            resolved = scoped_profile(protocol, protocol["cases"][0], protocol["cases"][0]["predictions"][0])
            self.assertEqual(resolved["analysis"]["tica_lag_ns"], 2.0)
            self.assertEqual(resolved["analysis"]["n_states"], 4)
            self.assertEqual(resolved["analysis"]["physical"]["bond_relative_error_max"], 0.1)
            self.assertEqual(resolved["analysis"]["physical"]["clash_distance_A"], 1.2)


if __name__ == "__main__":
    unittest.main()
