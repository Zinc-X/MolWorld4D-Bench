"""Evaluation profiles share metric implementations and differ only in cost."""

from __future__ import annotations

from copy import deepcopy


PROFILES: dict[str, dict] = {
    "quick": {
        "description": "Training-time protein-ligand sentinel evaluation.",
        "defaults": {
            "max_frames": 100,
            "frame_stride": 1,
            "pca_components": 2,
            "acf_max_lag_frames": 20,
            "windows_ns": [],
        },
        "metrics": {
            "physical": True,
            "paired_short": True,
            "motion": True,
            "rmsf": True,
            "pca": True,
            "contacts": True,
            "acf": False,
            "long_windows": False,
            "slow_modes": False,
            "structural": False,
            "chemical_interactions": False,
            "thermodynamics": False,
            "solvent": False,
            "efficiency": False,
            "uncertainty": False,
            "reference_baseline": False,
            "advanced_windows": False,
        },
    },
    "full": {
        "description": "Research evaluation from tens of ns to microseconds.",
        "defaults": {
            "max_frames": None,
            "frame_stride": 1,
            "pca_components": 3,
            "acf_max_lag_frames": 100,
            "windows_ns": [],
            "tica_lag_frames": 10,
            "tica_input_components": 10,
            "tica_components": 3,
            "n_states": 6,
            "msm_lag_frames": [10, 20, 50],
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
            "bootstrap_block_frames": 20,
            "random_seed": 20260903,
            "trust_threshold_ratio": 2.0,
            "trust_persistence_frames": 5,
        },
        "metrics": {
            "physical": True,
            "paired_short": True,
            "motion": True,
            "rmsf": True,
            "pca": True,
            "contacts": True,
            "acf": True,
            "long_windows": True,
            "slow_modes": True,
            "structural": True,
            "chemical_interactions": True,
            "thermodynamics": True,
            "solvent": True,
            "efficiency": True,
            "uncertainty": True,
            "reference_baseline": True,
            "advanced_windows": True,
        },
    },
}


def resolve_profile(name: str, analysis: dict | None = None) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    resolved = deepcopy(PROFILES[name])
    analysis = dict(analysis or {})
    metrics = dict(resolved["metrics"])
    metrics.update(analysis.pop("metrics", {}) or {})
    defaults = dict(resolved["defaults"])
    defaults.update(analysis)
    resolved["metrics"] = metrics
    resolved["analysis"] = defaults
    resolved["name"] = name
    return resolved
