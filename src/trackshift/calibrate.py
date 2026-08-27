"""Alpha/beta calibration. Scale knobs only — not F1 constants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .pipeline import is_calibration_lap


def calibration_mask(lap_table: pd.DataFrame) -> pd.Series:
    return lap_table.apply(is_calibration_lap, axis=1)


def calibrate_alpha_beta(lap_table: pd.DataFrame) -> dict:
    mask = calibration_mask(lap_table)
    cal = lap_table.loc[mask].copy()
    if cal.empty:
        raise RuntimeError("Calibration set is empty.")
    d_med = float(cal["D_lap"].median())
    h_med = float(cal["H_lap"].median())
    if not np.isfinite(d_med) or d_med <= 0:
        raise RuntimeError(f"median D_lap is not usable: {d_med}")
    if not np.isfinite(h_med) or h_med <= 0:
        raise RuntimeError(
            f"median H_lap is not usable: {h_med}. Harvest pipeline is broken; do not invent beta."
        )
    target = config.TARGET_MEDIAN_ABS_NET
    alpha = target / d_med
    beta = target / h_med
    return {
        "NOT_F1_CONSTANTS": True,
        "comment": (
            "alpha and beta are simulation scale parameters. "
            "They are not battery capacity, MGU-K limits, ERS limits, "
            "electrical efficiency, or actual SOC parameters."
        ),
        "target_median_abs_net": target,
        "d_min_ms2": config.D_MIN_MS2,
        "a_clip_ms2": list(config.A_CLIP_MS2),
        "dt_gap_seconds": config.DT_GAP_SECONDS,
        "yellow_factor": config.YELLOW_FACTOR,
        "winsor_quantile": config.WINSOR_QUANTILE,
        "e0": config.E0,
        "calibration_set": {
            "definition": "green (TrackStatus==1 only), non-box, IsAccurate=True, not thin, not freeze",
            "n_laps": int(len(cal)),
            "median_D_lap": d_med,
            "median_H_lap": h_med,
        },
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_reduced": False,
        "beta_scale_applied": 1.0,
    }


def apply_scale(cfg: dict, factor: float) -> dict:
    out = dict(cfg)
    out["alpha"] = float(cfg["alpha"] * factor)
    out["beta"] = float(cfg["beta"] * factor)
    out["alpha_reduced"] = True
    out["beta_scale_applied"] = float(cfg.get("beta_scale_applied", 1.0) * factor)
    return out


def save_config(cfg: dict, path: Path | None = None) -> Path:
    path = path or config.ENERGY_CONFIG_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path
