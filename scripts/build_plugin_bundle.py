#!/usr/bin/env python3
"""Build a deterministic, skill-only MolWorld4D-Bench Codex plugin archive."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".codex-plugin" / "plugin.json"
SKILLS = ROOT / "skills"


def bundle_members() -> list[tuple[Path, str]]:
    if not MANIFEST.is_file():
        raise RuntimeError("missing .codex-plugin/plugin.json")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("name") != "molworld4d-bench" or not payload.get("version"):
        raise RuntimeError("plugin manifest name/version is invalid")
    members = [(MANIFEST, ".codex-plugin/plugin.json")]
    members.extend(
        (path, path.relative_to(ROOT).as_posix())
        for path in sorted(SKILLS.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not any(name.endswith("/SKILL.md") for _, name in members):
        raise RuntimeError("plugin has no SKILL.md")
    return members


def build_bundle(out_dir: Path) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    destination = out_dir / f"molworld4d-bench-plugin-{payload['version']}.zip"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing plugin archive: {destination}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in bundle_members():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plugin-dist")
    args = parser.parse_args()
    print(build_bundle(args.out_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
