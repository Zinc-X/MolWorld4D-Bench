import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from complexmd_eval.codex_integration import (
    SKILL_ENGINE_RANGE,
    doctor_report,
    install_skill,
    locate_skill_source,
    skill_status,
)


ROOT = Path(__file__).parents[1]


class CodexIntegrationTests(unittest.TestCase):
    def test_bundled_skill_is_complete_and_engine_is_compatible(self) -> None:
        source = locate_skill_source()
        self.assertIsNotNone(source)
        self.assertTrue((source / "SKILL.md").is_file())
        report = doctor_report(profile="screen")
        self.assertTrue(report["ok"])
        self.assertEqual(report["skill_engine_range"], SKILL_ENGINE_RANGE)

    def test_repo_install_is_non_destructive_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            dry_run = install_skill("repo", repo_root=repo, dry_run=True)
            self.assertEqual(dry_run["status"], "would_install")
            self.assertFalse(Path(dry_run["target"]).exists())
            installed = install_skill("repo", repo_root=repo)
            self.assertEqual(installed["status"], "installed")
            second = install_skill("repo", repo_root=repo)
            self.assertEqual(second["status"], "already_installed")
            self.assertFalse(second["changed"])
            status = skill_status(repo)
            self.assertTrue(status["targets"]["repo"]["matches_bundled"])

    def test_repo_install_refuses_to_overwrite_different_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            target = repo / ".agents" / "skills" / "complexmd-eval"
            target.mkdir(parents=True)
            marker = target / "SKILL.md"
            marker.write_text("user-owned content", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                install_skill("repo", repo_root=repo)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user-owned content")

    def test_cli_doctor_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "complexmd_eval.cli", "doctor", "--profile", "screen", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profile"], "screen")

    def test_plugin_manifest_matches_release(self) -> None:
        manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "molworld4d-bench")
        self.assertEqual(manifest["version"], "0.3.1")
        self.assertEqual(manifest["skills"], "./skills/")

    def test_plugin_bundle_contains_manifest_and_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_plugin_bundle.py"), "--out-dir", tmp],
                check=True,
                capture_output=True,
                text=True,
            )
            archive_path = Path(completed.stdout.strip())
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn(".codex-plugin/plugin.json", names)
            self.assertIn("skills/complexmd-eval/SKILL.md", names)


if __name__ == "__main__":
    unittest.main()
