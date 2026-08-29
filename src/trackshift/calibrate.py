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
    if "calibration_set" in lap_table.columns:
        mask = lap_table["calibration_set"].astype(bool)
    else:
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
    target_net = config.TARGET_MEDIAN_ABS_NET
    imbalance = (cal["H_lap"] / h_med) - (cal["D_lap"] / d_med)
    median_abs_imbalance = float(imbalance.abs().median())
    if not np.isfinite(median_abs_imbalance) or median_abs_imbalance <= 0:
        raise RuntimeError(
            f"median |H/medianH - D/medianD| is not usable: {median_abs_imbalance}"
        )
    # T scales the typical relative NET imbalance to target_net.
    # Previous formula used T = target_net, which scaled each flow to 0.03
    # instead of the median absolute net change.
    flow_scale_T = target_net / median_abs_imbalance
    alpha = flow_scale_T / d_med
    beta = flow_scale_T / h_med
    return {
        "NOT_F1_CONSTANTS": True,
        "comment": (
            "alpha and beta are simulation scale parameters. "
            "They are not battery capacity, MGU-K limits, ERS limits, "
            "electrical efficiency, or actual SOC parameters. "
            "The previous formula scaled each flow to 0.03 individually. "
            "The corrected formula scales the median absolute NET imbalance "
            "to approximately 0.03, matching the stated design objective."
        ),
        "target_median_abs_net": target_net,
        "flow_scale_T": float(flow_scale_T),
        "median_abs_imbalance": median_abs_imbalance,
        "calibration_method": (
            "T = 0.03 / median_cal(|H_lap/median(H_lap) - D_lap/median(D_lap)|); "
            "alpha = T / median(D_lap); beta = T / median(H_lap)"
        ),
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
            "median_abs_imbalance": median_abs_imbalance,
            "mean_imbalance": float(imbalance.mean()),
            "median_imbalance": float(imbalance.median()),
        },
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_reduced": False,
        "beta_scale_applied": 1.0,
        "floor_reduction_reason": None,
    }


def apply_scale(cfg: dict, factor: float) -> dict:
    out = dict(cfg)
    out["alpha"] = float(cfg["alpha"] * factor)
    out["beta"] = float(cfg["beta"] * factor)
    out["alpha_reduced"] = True
    out["beta_scale_applied"] = float(cfg.get("beta_scale_applied", 1.0) * factor)
    out["flow_scale_T"] = float(cfg.get("flow_scale_T", 0.0) * factor)
    out["floor_reduction_reason"] = (
        "floor-touch lap fraction exceeded 0.20; "
        "applied existing joint 0.5 reduction to both alpha and beta "
        "(design report §17 step 4). Not a new reduction factor."
    )
    return out


def save_config(cfg: dict, path: Path | None = None) -> Path:
    path = path or config.ENERGY_CONFIG_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path
