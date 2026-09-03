"""Task-oriented evaluation profiles and deterministic configuration merging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


METRIC_NAMES = (
    "physical", "paired_short", "motion", "rmsf", "pca", "contacts", "acf",
    "long_windows", "slow_modes", "structural", "chemical_interactions",
    "thermodynamics", "solvent", "efficiency", "uncertainty",
    "reference_baseline", "advanced_windows",
)

COMMON_DEFAULTS: dict[str, Any] = {
    "max_frames": None,
    "frame_stride": 1,
    "comparison_horizon": "common",
    "max_timebase_coarsening": 20.0,
    "timebase_tolerance_ns": 1e-8,
    "pca_components": 3,
    "windows_ns": [],
    "auto_windows": {
        "mode": "fractions",
        "fractions": [0.1, 0.25, 0.5, 0.75, 1.0],
        "count": 5,
        "min_fraction": 0.02,
    },
    "tica_lag_ns": "auto",
    "msm_lags_ns": "auto",
    "acf_max_lag_ns": "auto",
    "bootstrap_block_ns": "auto",
    "tica_input_components": 10,
    "tica_components": 3,
    "n_states": "auto",
    "n_states_min": 2,
    "n_states_max": 12,
    "min_frames_per_state": 20,
    "min_lagged_samples_per_state": 5,
    "min_transition_events": 5,
    "min_mfpt_pairs": 1,
    "max_ck_rmse": 0.15,
    "rare_transition_max_count": 3,
    "fes_bins": 24,
    "temperature_K": 300.0,
    "structural_stride": 10,
    "interaction_stride": 2,
    "solvent_stride": 2,
    "torsion_bins": 36,
    "bootstrap_samples": 500,
    "bootstrap_confidence": 0.95,
    "random_seed": 20260903,
    "trust_threshold_ratio": 2.0,
    "trust_persistence_frames": 5,
}


def _metrics(*enabled: str) -> dict[str, bool]:
    selected = set(enabled)
    return {name: name in selected for name in METRIC_NAMES}


def deep_merge(base: dict | None, override: dict | None) -> dict:
    """Recursively merge mappings without mutating either input."""
    merged = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_analysis_overrides(analysis: dict | None) -> dict:
    """Make legacy frame controls override inherited physical-time controls."""
    normalized = deepcopy(analysis or {})
    for physical_key, legacy_key in (
        ("tica_lag_ns", "tica_lag_frames"),
        ("msm_lags_ns", "msm_lag_frames"),
        ("acf_max_lag_ns", "acf_max_lag_frames"),
        ("bootstrap_block_ns", "bootstrap_block_frames"),
    ):
        if legacy_key in normalized and physical_key not in normalized:
            normalized[physical_key] = None
    return normalized


def _profile(description: str, enabled: tuple[str, ...], defaults: dict | None = None) -> dict:
    analysis = deep_merge(COMMON_DEFAULTS, defaults)
    return {"description": description, "defaults": analysis, "metrics": _metrics(*enabled)}


PROFILES: dict[str, dict] = {
    # Retained for compatibility. New protocols should prefer ``screen``.
    "quick": _profile(
        "Compatibility training-time sentinel using at most 100 retained frames.",
        ("physical", "paired_short", "motion", "rmsf", "pca", "contacts"),
        {"max_frames": 100, "pca_components": 2, "windows_ns": []},
    ),
    "screen": _profile(
        "Low-cost physical, paired, and motion screening on a physical horizon.",
        ("physical", "paired_short", "motion", "rmsf", "contacts"),
    ),
    "structural": _profile(
        "Structural ensembles, torsions, contacts, interactions, solvent, and observables.",
        (
            "physical", "motion", "rmsf", "pca", "contacts", "structural",
            "chemical_interactions", "thermodynamics", "solvent", "uncertainty",
        ),
    ),
    "ensemble": _profile(
        "Reference-fitted PCA/TICA, state occupancy, and free-energy surfaces.",
        ("physical", "motion", "rmsf", "pca", "contacts", "slow_modes", "uncertainty"),
    ),
    "kinetics": _profile(
        "Time-aware ACF, TICA/VAMP, MSM, ITS, MFPT, rare-event, and baseline evaluation.",
        (
            "motion", "pca", "contacts", "acf", "slow_modes", "long_windows",
            "reference_baseline", "uncertainty", "advanced_windows",
        ),
        {"windows_ns": "auto"},
    ),
    "long_horizon": _profile(
        "Cumulative-window structural, ensemble, interaction, and kinetic recovery.",
        (
            "physical", "motion", "rmsf", "pca", "contacts", "acf", "long_windows",
            "slow_modes", "structural", "chemical_interactions", "thermodynamics",
            "solvent", "efficiency", "uncertainty", "reference_baseline",
            "advanced_windows",
        ),
        {"windows_ns": "auto"},
    ),
    "full": _profile(
        "Complete compatibility-stable research evaluation without a fixed case count.",
        tuple(METRIC_NAMES),
        {
            "windows_ns": [],
            "tica_lag_ns": None,
            "tica_lag_frames": 10,
            "msm_lags_ns": None,
            "msm_lag_frames": [10, 20, 50],
            "acf_max_lag_ns": None,
            "acf_max_lag_frames": 100,
            "bootstrap_block_ns": None,
            "bootstrap_block_frames": 20,
            "n_states": 6,
        },
    ),
}


def resolve_profile(name: str, analysis: dict | None = None) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    template = deepcopy(PROFILES[name])
    overrides = normalize_analysis_overrides(analysis)
    metric_overrides = overrides.pop("metrics", {}) or {}
    template["metrics"] = deep_merge(template["metrics"], metric_overrides)
    template["analysis"] = deep_merge(template["defaults"], overrides)
    template.pop("defaults", None)
    template["name"] = name
    return template
