"""Physical-time harmonization and auditable parameter resolution."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import gcd, lcm
from typing import Any

import numpy as np
import pandas as pd

from .models import Trajectory


class TimebaseError(ValueError):
    """Raised when trajectories cannot be compared on a safe common clock."""


def trajectory_dt_ns(trajectory: Trajectory, tolerance_ns: float = 1e-8) -> float:
    if len(trajectory.time_ns) < 2:
        raise TimebaseError(f"{trajectory.source}: at least two frames are required")
    differences = np.diff(np.asarray(trajectory.time_ns, dtype=float))
    dt = float(np.median(differences))
    if not np.isfinite(dt) or dt <= 0:
        raise TimebaseError(f"{trajectory.source}: time values must be strictly increasing")
    tolerance = max(float(tolerance_ns), abs(dt) * 1e-7)
    if not np.allclose(differences, dt, rtol=1e-7, atol=tolerance):
        raise TimebaseError(f"{trajectory.source}: non-uniform time grid is not safely resampleable")
    return dt


def _common_step(values: list[float], tolerance_ns: float) -> float:
    fractions = [Fraction(str(round(value, 12))).limit_denominator(10**9) for value in values]
    for value, fraction in zip(values, fractions):
        if abs(float(fraction) - value) > max(tolerance_ns, abs(value) * 1e-7):
            raise TimebaseError(f"sampling interval {value:g} ns cannot be represented reliably")
    numerator = 1
    denominator = fractions[0].denominator
    for fraction in fractions:
        numerator = lcm(numerator, fraction.numerator)
        denominator = gcd(denominator, fraction.denominator)
    return float(Fraction(numerator, denominator))


def _take(trajectory: Trajectory, indices: np.ndarray) -> Trajectory:
    return replace(
        trajectory,
        positions=trajectory.positions[indices],
        time_ns=trajectory.time_ns[indices],
        observables={name: values[indices] for name, values in trajectory.observables.items()},
    )


def harmonize_timebases(
    references: list[Trajectory],
    prediction: Trajectory,
    config: dict,
) -> tuple[list[Trajectory], Trajectory, pd.DataFrame, dict[str, float | str]]:
    """Exactly downsample trajectories onto a shared grid and apply a horizon policy.

    Coordinates are never interpolated. A commensurate grid that would require
    excessive coarsening is rejected instead of silently changing the kinetics.
    """
    trajectories = [*references, prediction]
    tolerance = float(config.get("timebase_tolerance_ns", 1e-8))
    dts = [trajectory_dt_ns(item, tolerance) for item in trajectories]
    starts = [float(item.time_ns[0]) for item in trajectories]
    if max(starts) - min(starts) > tolerance:
        raise TimebaseError("trajectory start times differ; align clocks before evaluation")
    common_dt = _common_step(dts, tolerance)
    factors = [common_dt / dt for dt in dts]
    rounded_factors = [int(round(value)) for value in factors]
    if any(abs(value - rounded) > 1e-7 for value, rounded in zip(factors, rounded_factors)):
        raise TimebaseError("sampling intervals do not have an exact common downsampling grid")
    max_coarsening = float(config.get("max_timebase_coarsening", 20.0))
    if max(factors) > max_coarsening + 1e-9:
        raise TimebaseError(
            f"safe common grid requires {max(factors):.3g}x coarsening, above "
            f"max_timebase_coarsening={max_coarsening:g}"
        )

    policy = str(config.get("comparison_horizon", "common"))
    if policy not in {"common", "reference", "prediction", "full_each"}:
        raise TimebaseError(f"unknown comparison_horizon {policy!r}")
    durations = [float(item.time_ns[-1] - item.time_ns[0]) for item in trajectories]
    requested_horizon: float | None
    if policy == "common":
        requested_horizon = min(durations)
    elif policy == "reference":
        requested_horizon = durations[0]
    elif policy == "prediction":
        requested_horizon = durations[-1]
    else:
        requested_horizon = None
    if requested_horizon is not None:
        too_short = [duration for duration in durations if duration + tolerance < requested_horizon]
        if too_short:
            raise TimebaseError(
                f"comparison_horizon={policy} requests {requested_horizon:g} ns but at least "
                f"one trajectory ends at {min(too_short):g} ns"
            )
        horizon_steps = int(np.floor((requested_horizon + tolerance) / common_dt))
        actual_horizon = horizon_steps * common_dt
    else:
        actual_horizon = min(
            int(np.floor((duration + tolerance) / common_dt)) * common_dt for duration in durations
        )

    outputs: list[Trajectory] = []
    rows: list[dict[str, Any]] = []
    for index, (trajectory, dt, factor) in enumerate(zip(trajectories, dts, rounded_factors)):
        indices = np.arange(0, len(trajectory.positions), factor, dtype=int)
        if requested_horizon is not None:
            relative = trajectory.time_ns[indices] - trajectory.time_ns[0]
            indices = indices[relative <= actual_horizon + tolerance]
        if len(indices) < 2:
            raise TimebaseError(f"{trajectory.source}: fewer than two frames remain after harmonization")
        output = _take(trajectory, indices)
        outputs.append(output)
        rows.append(
            {
                "source_kind": "reference" if index < len(references) else "prediction",
                "source_replica": index if index < len(references) else 0,
                "source": trajectory.source,
                "comparison_horizon": policy,
                "input_dt_ns": dt,
                "common_dt_ns": common_dt,
                "resample_factor": factor,
                "input_frames": len(trajectory.positions),
                "output_frames": len(output.positions),
                "input_duration_ns": durations[index],
                "output_duration_ns": float(output.time_ns[-1] - output.time_ns[0]),
                "requested_horizon_ns": requested_horizon,
                "resolved_common_horizon_ns": actual_horizon,
                "horizon_rounding_error_ns": (
                    actual_horizon - requested_horizon if requested_horizon is not None else 0.0
                ),
            }
        )
    summary: dict[str, float | str] = {
        "comparison_horizon": policy,
        "common_dt_ns": common_dt,
        "resolved_common_horizon_ns": actual_horizon,
        "max_resample_factor": float(max(rounded_factors)),
    }
    return outputs[:-1], outputs[-1], pd.DataFrame(rows), summary


def _resolve_scalar_time(
    config: dict,
    ns_key: str,
    frames_key: str,
    dt_ns: float,
    duration_ns: float,
    auto_fraction: float,
) -> tuple[int, dict[str, Any]]:
    requested = config.get(ns_key)
    if requested is None:
        frames = max(1, int(config.get(frames_key, 1)))
        desired_ns = frames * dt_ns
        source = "legacy_frames"
        requested_label: str | float = desired_ns
    else:
        if str(requested).lower() == "auto":
            desired_ns = max(dt_ns, duration_ns * auto_fraction)
            source = "auto_duration_fraction"
            requested_label = "auto"
        else:
            desired_ns = float(requested)
            if desired_ns <= 0:
                raise TimebaseError(f"{ns_key} must be positive or 'auto'")
            source = "physical_time"
            requested_label = desired_ns
        frames = max(1, int(np.floor(desired_ns / dt_ns + 0.5)))
    actual_ns = frames * dt_ns
    return frames, {
        "parameter": ns_key,
        "requested": requested_label,
        "desired_ns": desired_ns,
        "resolved_frames": frames,
        "resolved_ns": actual_ns,
        "rounding_error_ns": actual_ns - desired_ns,
        "resolution_source": source,
        "resolution_status": "resolved",
    }


def resolve_time_parameters(
    config: dict, dt_ns: float, duration_ns: float, enabled: dict | None = None
) -> tuple[dict, pd.DataFrame]:
    """Resolve physical-time controls to frame counts on the harmonized grid."""
    resolved = dict(config)
    rows: list[dict[str, Any]] = []
    mappings = (
        ("tica_lag_ns", "tica_lag_frames", 0.02, "slow_modes"),
        ("acf_max_lag_ns", "acf_max_lag_frames", 0.10, "acf"),
        ("bootstrap_block_ns", "bootstrap_block_frames", 0.05, "uncertainty"),
    )
    for ns_key, frames_key, fraction, module in mappings:
        if enabled is not None and not enabled.get(module):
            continue
        frames, row = _resolve_scalar_time(
            config, ns_key, frames_key, dt_ns, duration_ns, fraction
        )
        resolved[frames_key] = frames
        rows.append(row)

    if enabled is not None and not enabled.get("slow_modes"):
        return resolved, pd.DataFrame(rows)
    requested_lags = config.get("msm_lags_ns")
    if requested_lags is None:
        legacy = [max(1, int(value)) for value in config.get("msm_lag_frames", [resolved["tica_lag_frames"]])]
        lag_specs: list[tuple[str | float, float, str]] = [
            (frames * dt_ns, frames * dt_ns, "legacy_frames") for frames in legacy
        ]
    elif isinstance(requested_lags, str) and requested_lags.lower() == "auto":
        base = resolved["tica_lag_frames"] * dt_ns
        lag_specs = [("auto", base * multiplier, "auto_tica_multiples") for multiplier in (1, 2, 5)]
    else:
        lag_specs = [(float(value), float(value), "physical_time") for value in requested_lags]
    msm_frames: list[int] = []
    max_frames = max(1, int(np.floor(duration_ns / dt_ns)) // 3)
    for requested_label, desired_ns, source in lag_specs:
        if desired_ns <= 0:
            raise TimebaseError("msm_lags_ns values must be positive")
        frames = max(1, int(np.floor(desired_ns / dt_ns + 0.5)))
        if frames > max_frames:
            rows.append(
                {
                    "parameter": "msm_lags_ns",
                    "requested": requested_label,
                    "desired_ns": desired_ns,
                    "resolved_frames": np.nan,
                    "resolved_ns": np.nan,
                    "rounding_error_ns": np.nan,
                    "resolution_source": source,
                    "resolution_status": "skipped_insufficient_horizon",
                }
            )
            continue
        if frames in msm_frames:
            continue
        msm_frames.append(frames)
        actual_ns = frames * dt_ns
        rows.append(
            {
                "parameter": "msm_lags_ns",
                "requested": requested_label,
                "desired_ns": desired_ns,
                "resolved_frames": frames,
                "resolved_ns": actual_ns,
                "rounding_error_ns": actual_ns - desired_ns,
                "resolution_source": source,
                "resolution_status": "resolved",
            }
        )
    if not msm_frames:
        msm_frames = [max(1, min(resolved["tica_lag_frames"], max_frames))]
    resolved["msm_lag_frames"] = msm_frames
    return resolved, pd.DataFrame(rows)


def resolve_windows(config: dict, duration_ns: float, dt_ns: float) -> tuple[list[float], pd.DataFrame]:
    """Resolve explicit or automatic cumulative windows on the common grid."""
    requested = config.get("windows_ns", [])
    auto = dict(config.get("auto_windows", {}) or {})
    specifications: list[tuple[str, float]] = []
    if isinstance(requested, str):
        if requested.lower() != "auto":
            raise TimebaseError("windows_ns must be a list or 'auto'")
        mode = str(auto.get("mode", "fractions"))
        if mode == "fractions":
            for value in auto.get("fractions", [0.1, 0.25, 0.5, 0.75, 1.0]):
                fraction = float(value)
                if 0 < fraction <= 1:
                    specifications.append((f"fraction:{fraction:g}", duration_ns * fraction))
        elif mode == "log":
            count = max(2, int(auto.get("count", 5)))
            lower = max(2 * dt_ns, duration_ns * float(auto.get("min_fraction", 0.02)))
            if lower <= duration_ns:
                specifications.extend(
                    ("log:auto", float(value)) for value in np.geomspace(lower, duration_ns, count)
                )
        else:
            raise TimebaseError(f"unknown auto_windows.mode {mode!r}")
    else:
        specifications = [("explicit", float(value)) for value in (requested or [])]

    max_step = int(np.floor((duration_ns + 1e-9) / dt_ns))
    rows: list[dict[str, Any]] = []
    windows: list[float] = []
    for source, desired_ns in specifications:
        if desired_ns <= 0 or desired_ns > duration_ns + 1e-9:
            rows.append(
                {
                    "parameter": "windows_ns",
                    "requested": source,
                    "desired_ns": desired_ns,
                    "resolved_frames": np.nan,
                    "resolved_ns": np.nan,
                    "rounding_error_ns": np.nan,
                    "resolution_source": "automatic" if source != "explicit" else "physical_time",
                    "resolution_status": "skipped_outside_horizon",
                }
            )
            continue
        frames = min(max_step, max(2, int(np.floor(desired_ns / dt_ns + 0.5))))
        actual_ns = frames * dt_ns
        if actual_ns in windows:
            continue
        windows.append(actual_ns)
        rows.append(
            {
                "parameter": "windows_ns",
                "requested": source,
                "desired_ns": desired_ns,
                "resolved_frames": frames,
                "resolved_ns": actual_ns,
                "rounding_error_ns": actual_ns - desired_ns,
                "resolution_source": "automatic" if source != "explicit" else "physical_time",
                "resolution_status": "resolved",
            }
        )
    return sorted(windows), pd.DataFrame(rows)
