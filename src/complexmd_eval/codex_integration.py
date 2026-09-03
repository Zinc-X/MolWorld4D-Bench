"""Runtime checks and non-destructive Codex skill installation helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Any

from ._version import __version__
from .profiles import PROFILES


SKILL_NAME = "complexmd-eval"
SKILL_ENGINE_RANGE = ">=0.3.1,<0.4.0"
_MIN_ENGINE_VERSION = (0, 3, 1)
_MAX_ENGINE_VERSION = (0, 4, 0)
_SKILL_FILES = (
    "SKILL.md",
    "agents/openai.yaml",
    "references/advanced-guidance.md",
    "references/protocol-guidance.md",
)
_CORE_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "PyYAML": "yaml",
    "matplotlib": "matplotlib",
}
_PROFILE_OPTIONAL_IMPORTS = {
    "structural": {"mdtraj": "SASA and DSSP", "networkx": "ligand ring and pi interactions"},
    "long_horizon": {"mdtraj": "SASA and DSSP", "networkx": "ligand ring and pi interactions"},
    "full": {"mdtraj": "SASA and DSSP", "networkx": "ligand ring and pi interactions"},
}


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split("+", 1)[0].split("-", 1)[0].split(".")
    try:
        return tuple(int(part) for part in (parts + ["0", "0"])[:3])  # type: ignore[return-value]
    except ValueError:
        return (0, 0, 0)


def locate_skill_source() -> Path | None:
    """Locate the bundled skill in a checkout or an installed distribution."""
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "skills" / SKILL_NAME,
        module_path.parents[1] / "share" / SKILL_NAME / "skills" / SKILL_NAME,
        Path(sysconfig.get_path("data")) / "share" / SKILL_NAME / "skills" / SKILL_NAME,
        Path(sys.prefix) / "share" / SKILL_NAME / "skills" / SKILL_NAME,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if all((normalized / relative).is_file() for relative in _SKILL_FILES):
            return normalized
    return None


def find_repo_root(start: str | Path | None = None) -> Path | None:
    path = Path(start or Path.cwd()).expanduser().resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def skill_target(scope: str, repo_root: str | Path | None = None) -> Path:
    if scope == "user":
        return Path.home() / ".agents" / "skills" / SKILL_NAME
    if scope == "repo":
        root = find_repo_root(repo_root)
        if root is None:
            raise RuntimeError("cannot locate a Git repository; pass --repo PATH")
        return root / ".agents" / "skills" / SKILL_NAME
    raise ValueError("skill scope must be 'user' or 'repo'")


def _bundle_digest(root: Path) -> str | None:
    root = root.resolve()
    if not all((root / relative).is_file() for relative in _SKILL_FILES):
        return None
    digest = hashlib.sha256()
    for relative in _SKILL_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def skill_status(repo_root: str | Path | None = None) -> dict[str, Any]:
    source = locate_skill_source()
    source_digest = _bundle_digest(source) if source else None
    targets: dict[str, Any] = {}
    for scope in ("user", "repo"):
        try:
            target = skill_target(scope, repo_root)
        except RuntimeError as exc:
            targets[scope] = {"available": False, "reason": str(exc)}
            continue
        lexists = os.path.lexists(target)
        target_digest = _bundle_digest(target) if lexists else None
        targets[scope] = {
            "available": True,
            "path": str(target),
            "installed": bool(target_digest),
            "matches_bundled": bool(source_digest and target_digest == source_digest),
            "is_symlink": target.is_symlink(),
        }
    return {
        "skill": SKILL_NAME,
        "source": str(source) if source else None,
        "source_complete": source is not None,
        "targets": targets,
    }


def install_skill(
    scope: str,
    *,
    repo_root: str | Path | None = None,
    mode: str = "copy",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the bundled skill without overwriting any existing target."""
    source = locate_skill_source()
    if source is None:
        raise RuntimeError("bundled Codex skill is incomplete or unavailable")
    target = skill_target(scope, repo_root)
    result = {
        "skill": SKILL_NAME,
        "scope": scope,
        "mode": mode,
        "source": str(source),
        "target": str(target),
        "dry_run": dry_run,
    }
    if os.path.lexists(target):
        if _bundle_digest(target) == _bundle_digest(source):
            return {**result, "status": "already_installed", "changed": False}
        raise RuntimeError(
            f"refusing to overwrite existing skill at {target}; it differs from the bundled "
            "skill and no files were changed"
        )
    if mode not in {"copy", "symlink"}:
        raise ValueError("skill install mode must be 'copy' or 'symlink'")
    if dry_run:
        return {**result, "status": "would_install", "changed": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(source, target)
    else:
        target.symlink_to(source, target_is_directory=True)
    return {**result, "status": "installed", "changed": True}


def _check(name: str, status: str, required: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "required": required, "message": message}


def _has_non_npz_trajectories(protocol: dict[str, Any]) -> bool:
    for case in protocol["cases"]:
        for group in (case["reference"], *case["predictions"]):
            for item in group["trajectories"]:
                value = item.get("path") if isinstance(item, dict) else item
                if Path(str(value)).suffix.lower() != ".npz":
                    return True
    return False


def doctor_report(
    *,
    profile: str | None = None,
    protocol_path: str | Path | None = None,
    check_paths: bool = True,
) -> dict[str, Any]:
    """Return machine-readable engine, dependency, protocol, and skill checks."""
    checks: list[dict[str, Any]] = []
    version = _version_tuple(__version__)
    compatible = _MIN_ENGINE_VERSION <= version < _MAX_ENGINE_VERSION
    checks.append(
        _check(
            "engine_version",
            "ok" if compatible else "error",
            True,
            f"installed {__version__}; skill expects {SKILL_ENGINE_RANGE}",
        )
    )
    for distribution, import_name in _CORE_IMPORTS.items():
        available = importlib.util.find_spec(import_name) is not None
        checks.append(
            _check(
                f"dependency:{distribution}",
                "ok" if available else "error",
                True,
                "available" if available else f"missing required dependency {distribution}",
            )
        )

    resolved_protocol = None
    selected_profile = profile
    needs_mdanalysis = False
    if protocol_path is not None:
        try:
            from .protocol import load_protocol

            resolved_protocol = load_protocol(
                protocol_path,
                profile_override=profile,
                check_paths=check_paths,
            )
            selected_profile = str(resolved_protocol["profile"])
            needs_mdanalysis = _has_non_npz_trajectories(resolved_protocol)
            checks.append(_check("protocol", "ok", True, "protocol is valid"))
        except Exception as exc:
            checks.append(_check("protocol", "error", True, str(exc)))
    if selected_profile is not None and selected_profile not in PROFILES:
        checks.append(
            _check(
                "profile",
                "error",
                True,
                f"unknown profile {selected_profile!r}; choose one of {sorted(PROFILES)}",
            )
        )
    elif selected_profile is not None:
        checks.append(_check("profile", "ok", True, f"selected {selected_profile}"))

    mda_available = importlib.util.find_spec("MDAnalysis") is not None
    checks.append(
        _check(
            "dependency:MDAnalysis",
            "ok" if mda_available else ("error" if needs_mdanalysis else "skipped"),
            needs_mdanalysis,
            "available"
            if mda_available
            else (
                "required by non-NPZ trajectory inputs; install complexmd-eval[trajectory]"
                if needs_mdanalysis
                else "not required for NPZ-only or unspecified inputs"
            ),
        )
    )
    for import_name, capability in _PROFILE_OPTIONAL_IMPORTS.get(selected_profile or "", {}).items():
        available = importlib.util.find_spec(import_name) is not None
        checks.append(
            _check(
                f"optional:{import_name}",
                "ok" if available else "warning",
                False,
                "available"
                if available
                else f"missing optional dependency for {capability}; install complexmd-eval[all]",
            )
        )
    skill_source = locate_skill_source()
    checks.append(
        _check(
            "skill_bundle",
            "ok" if skill_source else "error",
            True,
            str(skill_source) if skill_source else "bundled skill files are incomplete",
        )
    )
    ok = not any(item["status"] == "error" and item["required"] for item in checks)
    complete = ok and not any(item["status"] == "warning" for item in checks)
    return {
        "ok": ok,
        "complete": complete,
        "engine_version": __version__,
        "skill_engine_range": SKILL_ENGINE_RANGE,
        "profile": selected_profile,
        "protocol": str(Path(protocol_path).resolve()) if protocol_path is not None else None,
        "checks": checks,
        "skill": skill_status(
            Path(protocol_path).resolve().parent if protocol_path is not None else None
        ),
    }


def render_doctor(report: dict[str, Any]) -> str:
    lines = [
        f"ComplexMD-Eval {report['engine_version']} preflight",
        f"runnable: {'yes' if report['ok'] else 'no'}",
        f"complete selected capability: {'yes' if report['complete'] else 'no'}",
    ]
    for item in report["checks"]:
        label = item["status"].upper()
        lines.append(f"[{label}] {item['name']}: {item['message']}")
    return "\n".join(lines)


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
