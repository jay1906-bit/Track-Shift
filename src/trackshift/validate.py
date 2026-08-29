"""Section 18 validation pack. No SOC ground truth is used."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .race_state import is_box_lap, is_freeze_status


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def occupancy_metrics(sample_e: pd.Series, end_e: pd.Series, energy_laps: pd.DataFrame) -> dict:
    """Design-report §18.6 occupancy, reported at sample and end-of-lap level."""
    sample_e = pd.to_numeric(sample_e, errors="coerce")
    end_e = pd.to_numeric(end_e, errors="coerce")
    tmp = energy_laps.sort_values(["Driver", "LapNumber"]).copy()
    tmp["dE"] = tmp.groupby("Driver")["EstimatedEnergyIndex_end"].diff()
    floor_touch = pd.to_numeric(energy_laps.get("n_clip_low", 0), errors="coerce").fillna(0) > 0
    ceil_recover = pd.to_numeric(energy_laps.get("n_clip_high", 0), errors="coerce").fillna(0) > 0
    return {
        "sample_frac_in_0p05_0p95": float(((sample_e > 0.05) & (sample_e < 0.95)).mean()),
        "sample_frac_ge_0p95": float((sample_e >= 0.95).mean()),
        "sample_frac_le_0p05": float((sample_e <= 0.05).mean()),
        "end_frac_in_0p05_0p95": float(((end_e > 0.05) & (end_e < 0.95)).mean()),
        "end_frac_ge_0p95": float((end_e >= 0.95).mean()),
        "end_frac_le_0p05": float((end_e <= 0.05).mean()),
        "end_min": float(end_e.min()),
        "end_max": float(end_e.max()),
        "end_mean": float(end_e.mean()),
        "end_median": float(end_e.median()),
        "sample_min": float(sample_e.min()),
        "sample_max": float(sample_e.max()),
        "floor_touch_lap_fraction": float(floor_touch.mean()) if len(energy_laps) else 0.0,
        "ceiling_recover_lap_fraction": float(ceil_recover.mean()) if len(energy_laps) else 0.0,
        "median_abs_dE": float(tmp["dE"].abs().median()) if tmp["dE"].notna().any() else float("nan"),
        "median_dE": float(tmp["dE"].median()) if tmp["dE"].notna().any() else float("nan"),
    }


def run_validation(
    lap_table: pd.DataFrame,
    samples: pd.DataFrame,
    ref_samples: pd.DataFrame,
    energy_laps: pd.DataFrame,
    raw_hash_before: str,
    clean_hash_before: str,
) -> dict:
    checks = []

    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Spatial sanity on reference flying lap
    if ref_samples.empty:
        add("spatial_deploy_harvest", False, "no reference samples")
    else:
        dist = ref_samples["Distance"].to_numpy(dtype=float)
        dep = ref_samples["DeployProxy"].to_numpy(dtype=float)
        har = ref_samples["HarvestProxy"].to_numpy(dtype=float)
        # Main-straight-ish: first/last 15% of distance vs middle braking
        q15, q85 = np.nanquantile(dist, [0.15, 0.85])
        straightish = (dist <= q15) | (dist >= q85)
        mid = (dist > q15) & (dist < q85)
        dep_straight = float(np.nanmean(dep[straightish])) if straightish.any() else np.nan
        har_mid = float(np.nanmean(har[mid])) if mid.any() else np.nan
        har_straight = float(np.nanmean(har[straightish])) if straightish.any() else np.nan
        deploy_ok = np.isfinite(dep_straight) and dep_straight > 0
        harvest_ok = np.isfinite(har_mid) and har_mid >= har_straight
        add(
            "spatial_deploy_harvest",
            deploy_ok and harvest_ok,
            f"mean DeployProxy on ends={dep_straight:.4f}; mean HarvestProxy mid={har_mid:.4f} vs ends={har_straight:.4f}",
        )

    # 2. Harvest ≈ 0 when Brake=False
    if samples.empty:
        add("harvest_brake_gate", False, "no samples")
    else:
        brake_false = ~samples["Brake"].fillna(False).astype(bool)
        h = samples.loc[brake_false, "HarvestProxy"].fillna(0.0)
        max_h = float(h.max()) if len(h) else 0.0
        add("harvest_brake_gate", max_h == 0.0, f"max HarvestProxy when Brake=False: {max_h}")

    # 3. Pit/SC freeze: ~0 net E change
    freeze = energy_laps["freeze_flag"].astype(bool)
    if "EstimatedEnergyIndex_end" in energy_laps.columns:
        # net change vs previous lap for same driver
        tmp = energy_laps.sort_values(["Driver", "LapNumber"]).copy()
        tmp["E_prev"] = tmp.groupby("Driver")["EstimatedEnergyIndex_end"].shift(1)
        tmp["dE"] = tmp["EstimatedEnergyIndex_end"] - tmp["E_prev"]
        freeze_de = tmp.loc[tmp["freeze_flag"] == True, "dE"].dropna()  # noqa: E712
        max_abs = float(freeze_de.abs().max()) if len(freeze_de) else 0.0
        add(
            "pit_sc_vsc_freeze",
            max_abs < 1e-9 or len(freeze_de) == 0,
            f"n freeze laps with previous E={len(freeze_de)}; max |dE|={max_abs}",
        )
    else:
        add("pit_sc_vsc_freeze", False, "missing energy column")

    # 4. max Δt used <= 1 s
    max_dt = float(pd.to_numeric(lap_table["max_dt_used"], errors="coerce").max())
    add("max_dt_used_le_1s", np.isfinite(max_dt) and max_dt <= config.DT_GAP_SECONDS + 1e-9, f"max_dt_used={max_dt}")

    # 5. E in [0,1]
    e = energy_laps["EstimatedEnergyIndex_end"]
    in_bounds = bool(e.between(config.E_MIN, config.E_MAX).all())
    add("energy_bounds", in_bounds, f"min={float(e.min())} max={float(e.max())}")

    # 6. Not pegged — design report §18.6: most of the race, most cars in (0.05, 0.95).
    # "Most" means majority occupancy, not the weakened "<80% exactly at 0 or 1" check.
    sample_e = samples["EstimatedEnergyIndex"] if "EstimatedEnergyIndex" in samples.columns else e
    occ = occupancy_metrics(sample_e, e, energy_laps)
    majority_sample = occ["sample_frac_in_0p05_0p95"] > 0.50
    majority_end = occ["end_frac_in_0p05_0p95"] > 0.50
    add(
        "energy_not_mostly_pegged",
        majority_sample and majority_end,
        (
            f"sample in (0.05,0.95)={occ['sample_frac_in_0p05_0p95']:.3f}; "
            f"end-of-lap in (0.05,0.95)={occ['end_frac_in_0p05_0p95']:.3f}; "
            f"end >=0.95={occ['end_frac_ge_0p95']:.3f}; end <=0.05={occ['end_frac_le_0p05']:.3f}; "
            f"end min={occ['end_min']:.5f} max={occ['end_max']:.5f} mean={occ['end_mean']:.5f} "
            f"median={occ['end_median']:.5f}; floor-touch laps={occ['floor_touch_lap_fraction']:.3f}; "
            f"ceiling-recover laps={occ['ceiling_recover_lap_fraction']:.3f}; "
            f"median |dE|={occ['median_abs_dE']:.5f}"
        ),
    )

    # 7. Higher mean throttle → higher D_lap (calibration laps, per driver sign of correlation)
    cal = lap_table.loc[lap_table["calibration_set"] == True].copy() if "calibration_set" in lap_table else lap_table  # noqa: E712
    if "calibration_set" not in lap_table:
        cal = lap_table.loc[
            (lap_table["TrackStatus"].astype(str) == "1")
            & lap_table["PitInTime"].isna()
            & lap_table["PitOutTime"].isna()
            & (lap_table["IsAccurate"] == True)  # noqa: E712
        ].copy()
    corrs = []
    for _, grp in cal.groupby("Driver"):
        if grp["mean_throttle"].nunique() < 2 or grp["D_lap"].nunique() < 2:
            continue
        corrs.append(float(grp["mean_throttle"].corr(grp["D_lap"])))
    mean_corr = float(np.nanmean(corrs)) if corrs else np.nan
    add(
        "throttle_vs_D_lap",
        np.isfinite(mean_corr) and mean_corr > 0,
        f"mean per-driver corr(mean_throttle, D_lap) on calibration laps={mean_corr}",
    )

    # 8–11 join integrity
    add("clean_laps_995", len(lap_table) == 995, f"lap_table rows={len(lap_table)}")
    add("output_rows_995", len(energy_laps) == 995, f"energy_laps rows={len(energy_laps)}")
    add(
        "zero_duplicate_driver_lap",
        int(energy_laps.duplicated(["Driver", "LapNumber"]).sum()) == 0,
        f"duplicates={int(energy_laps.duplicated(['Driver','LapNumber']).sum())}",
    )
    add(
        "label_estimated_energy_index",
        "EstimatedEnergyIndex_end" in energy_laps.columns,
        "column is EstimatedEnergyIndex_end (end-of-lap simulated state, not SOC)",
    )

    # 12–13 files unchanged
    raw_now = file_sha256(config.RAW_LAPS_CSV)
    clean_now = file_sha256(config.CLEAN_LAPS_CSV)
    add("raw_unchanged", raw_now == raw_hash_before, f"raw sha256={raw_now}")
    add("clean_unchanged", clean_now == clean_hash_before, f"clean sha256={clean_now}")

    # extra: freeze laps have ~0 D/H
    box_or_freeze = energy_laps["freeze_flag"].astype(bool)
    freeze_d = pd.to_numeric(energy_laps.loc[box_or_freeze, "D_lap"], errors="coerce").fillna(0.0)
    freeze_h = pd.to_numeric(energy_laps.loc[box_or_freeze, "H_lap"], errors="coerce").fillna(0.0)
    add(
        "freeze_proxies_near_zero",
        float(freeze_d.abs().max() or 0) < 1e-9 and float(freeze_h.abs().max() or 0) < 1e-9,
        f"max |D_lap| freeze={float(freeze_d.abs().max() or 0)}; max |H_lap| freeze={float(freeze_h.abs().max() or 0)}",
    )

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "n_passed": sum(c["passed"] for c in checks),
        "n_checks": len(checks),
        "checks": checks,
        "occupancy": occ,
    }
