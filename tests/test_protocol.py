from pathlib import Path
import tempfile
import unittest

from complexmd_eval.protocol import ProtocolError, load_protocol


class ProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
