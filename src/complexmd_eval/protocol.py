"""Load, validate, and resolve portable evaluation protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .profiles import PROFILES, deep_merge, normalize_analysis_overrides, resolve_profile


class ProtocolError(ValueError):
    pass


def _require(mapping: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ProtocolError(f"{where} missing required fields: {', '.join(missing)}")


def _trajectory_paths(group: dict, where: str) -> list[str]:
    paths = group.get("trajectories")
    if not isinstance(paths, list) or not paths:
        raise ProtocolError(f"{where}.trajectories must be a non-empty list")
    out = []
    for item in paths:
        path = item.get("path") if isinstance(item, dict) else item
        if not isinstance(path, str) or not path.strip():
            raise ProtocolError(f"{where}.trajectories contains an invalid path")
        out.append(path)
        if isinstance(item, dict):
            if "observables" in item and (not isinstance(item["observables"], str) or not item["observables"].strip()):
                raise ProtocolError(f"{where}.trajectories observables must be a CSV path")
            if "runtime_s" in item and float(item["runtime_s"]) <= 0:
                raise ProtocolError(f"{where}.trajectories runtime_s must be > 0")
    return out


def _validate_analysis(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError(f"{where} must be an object")
    horizon = value.get("comparison_horizon")
    if horizon is not None and horizon not in {"common", "reference", "prediction", "full_each"}:
        raise ProtocolError(
            f"{where}.comparison_horizon must be common, reference, prediction, or full_each"
        )
    windows = value.get("windows_ns")
    if isinstance(windows, str) and windows != "auto":
        raise ProtocolError(f"{where}.windows_ns string value must be 'auto'")
    for key in ("tica_lag_ns", "acf_max_lag_ns", "bootstrap_block_ns"):
        setting = value.get(key)
        if setting is not None and setting != "auto" and float(setting) <= 0:
            raise ProtocolError(f"{where}.{key} must be positive or 'auto'")
    msm_lags = value.get("msm_lags_ns")
    if msm_lags is not None and msm_lags != "auto":
        if not isinstance(msm_lags, list) or not msm_lags or any(float(item) <= 0 for item in msm_lags):
            raise ProtocolError(f"{where}.msm_lags_ns must be a non-empty positive list or 'auto'")
    n_states = value.get("n_states")
    if n_states is not None and n_states != "auto" and int(n_states) < 2:
        raise ProtocolError(f"{where}.n_states must be at least 2 or 'auto'")
    return value


def scoped_profile(protocol: dict, case: dict | None = None, prediction: dict | None = None) -> dict:
    """Resolve global, case, and prediction analysis overrides in that order."""
    analysis = normalize_analysis_overrides(protocol.get("analysis", {}) or {})
    if case is not None:
        analysis = deep_merge(analysis, normalize_analysis_overrides(case.get("analysis", {}) or {}))
    if prediction is not None:
        analysis = deep_merge(
            analysis, normalize_analysis_overrides(prediction.get("analysis", {}) or {})
        )
    return resolve_profile(str(protocol["profile"]), analysis)


def load_protocol(path: str | Path, profile_override: str | None = None, check_paths: bool = True) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    if not protocol_path.is_file():
        raise ProtocolError(f"protocol not found: {protocol_path}")
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ProtocolError("protocol must be a YAML/JSON object")
    _require(raw, ("protocol_version", "suite", "cases"), "protocol")
    if str(raw["protocol_version"]) != "1.0":
        raise ProtocolError("this release supports protocol_version: '1.0'")
    profile_name = profile_override or raw.get("profile", "full")
    if profile_name not in PROFILES:
        raise ProtocolError(f"profile must be one of {sorted(PROFILES)}")
    _validate_analysis(raw.get("analysis"), "analysis")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise ProtocolError("cases must be a non-empty list")
    seen_cases: set[str] = set()
    expected_models: set[str] | None = None
    for i, case in enumerate(cases):
        where = f"cases[{i}]"
        if not isinstance(case, dict):
            raise ProtocolError(f"{where} must be an object")
        _require(case, ("id", "topology", "reference", "predictions"), where)
        _validate_analysis(case.get("analysis"), f"{where}.analysis")
        case_id = str(case["id"])
        if case_id in seen_cases:
            raise ProtocolError(f"duplicate case id: {case_id}")
        seen_cases.add(case_id)
        ref = case["reference"]
        _require(ref, ("trajectories", "dt_ns"), f"{where}.reference")
        _trajectory_paths(ref, f"{where}.reference")
        if float(ref["dt_ns"]) <= 0:
            raise ProtocolError(f"{where}.reference.dt_ns must be > 0")
        preds = case["predictions"]
        if not isinstance(preds, list) or not preds:
            raise ProtocolError(f"{where}.predictions must be a non-empty list")
        names: list[str] = []
        for j, pred in enumerate(preds):
            pwhere = f"{where}.predictions[{j}]"
            _require(pred, ("name", "trajectories", "dt_ns"), pwhere)
            _validate_analysis(pred.get("analysis"), f"{pwhere}.analysis")
            _trajectory_paths(pred, pwhere)
            if float(pred["dt_ns"]) <= 0:
                raise ProtocolError(f"{pwhere}.dt_ns must be > 0")
            names.append(str(pred["name"]))
        if len(names) != len(set(names)):
            raise ProtocolError(f"duplicate prediction names in {case_id}: {names}")
        model_set = set(names)
        if expected_models is None:
            expected_models = model_set
        elif raw.get("strict_case_pairing", True) and model_set != expected_models:
            raise ProtocolError(
                f"prediction models for {case_id} do not match the paired case set: "
                f"expected {sorted(expected_models)}, got {sorted(model_set)}"
            )
    resolved = dict(raw)
    resolved["profile"] = profile_name
    resolved["resolved_profile"] = resolve_profile(profile_name, raw.get("analysis"))
    resolved["_protocol_path"] = str(protocol_path)
    if check_paths:
        _check_paths(resolved, protocol_path.parent)
    canonical = {k: v for k, v in resolved.items() if not k.startswith("_")}
    resolved["_protocol_hash"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return resolved


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _check_paths(protocol: dict, base: Path) -> None:
    missing: list[str] = []
    efficiency = (protocol.get("resolved_profile", {}).get("analysis", {}).get("efficiency", {}) or {})
    if efficiency.get("runtime_table"):
        runtime_path = resolve_path(str(efficiency["runtime_table"]), base)
        if not runtime_path.is_file():
            missing.append(str(runtime_path))
    for case in protocol["cases"]:
        topology = resolve_path(str(case["topology"]), base)
        # NPZ is self-describing and may intentionally use a placeholder topology.
        all_paths = _trajectory_paths(case["reference"], "reference")
        for pred in case["predictions"]:
            all_paths.extend(_trajectory_paths(pred, "prediction"))
        if any(Path(p).suffix.lower() != ".npz" for p in all_paths) and not topology.is_file():
            missing.append(str(topology))
        for value in all_paths:
            path = resolve_path(value, base)
            if not path.is_file():
                missing.append(str(path))
        for group in [case["reference"], *case["predictions"]]:
            for item in group["trajectories"]:
                if isinstance(item, dict) and item.get("observables"):
                    observable_path = resolve_path(str(item["observables"]), base)
                    if not observable_path.is_file():
                        missing.append(str(observable_path))
    if missing:
        preview = "\n  ".join(sorted(set(missing))[:20])
        raise ProtocolError(f"missing input paths:\n  {preview}")


def public_protocol(protocol: dict) -> dict:
    return {k: v for k, v in protocol.items() if not k.startswith("_")}
