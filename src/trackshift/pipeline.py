"""Per-lap processing and all-laps unscaled integrals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .proxies import add_proxies
from .race_state import is_green_only, is_box_lap, lap_race_state
from .session_io import pick_session_lap
from .telemetry import add_delta_time, add_kinematics, add_validity, extract_car_data, in_lap_window


def prepare_lap_telemetry(session, lap_row: pd.Series) -> tuple[pd.DataFrame, dict]:
    driver = str(lap_row["Driver"])
    lap_number = int(lap_row["LapNumber"])
    state = lap_race_state(lap_row["TrackStatus"], lap_row["PitInTime"], lap_row["PitOutTime"])
    meta = {
        "Driver": driver,
        "LapNumber": lap_number,
        "DriverNumber": lap_row.get("DriverNumber"),
        "IsAccurate": bool(lap_row["IsAccurate"]),
        "TrackStatus": str(lap_row["TrackStatus"]),
        "PitInTime": lap_row["PitInTime"],
        "PitOutTime": lap_row["PitOutTime"],
        **state,
        "thin_lap": False,
        "n_raw_samples": 0,
        "n_in_window": 0,
        "n_valid_samples": 0,
        "error": None,
    }
    try:
        session_lap = pick_session_lap(session, driver, lap_number)
        tel = extract_car_data(session_lap)
    except Exception as exc:
        meta["error"] = str(exc)
        empty = pd.DataFrame()
        return empty, meta

    meta["n_raw_samples"] = int(len(tel))
    tel = add_validity(tel)
    tel = add_delta_time(tel)
    tel = add_kinematics(tel)

    window = in_lap_window(tel, session_lap)
    tel["InLapWindow"] = window
    tel = add_proxies(tel, state["race_state_factor"])
    # Keep padded kinematics for the first in-window derivative, then drop pad.
    in_lap = tel.loc[window].copy().reset_index(drop=True)
    if len(in_lap):
        # Recompute Δt on the in-lap clock: first sample has undefined Δt.
        in_lap["DeltaTimeSeconds"] = in_lap["Date"].diff().dt.total_seconds()
        dt = in_lap["DeltaTimeSeconds"]
        in_lap["DeltaTimeValid"] = np.isfinite(dt) & (dt > 0) & (dt <= config.DT_GAP_SECONDS)
        in_lap["DeltaTimeGap"] = np.isfinite(dt) & (dt > config.DT_GAP_SECONDS)
        in_lap["DeltaTimeNonPositive"] = np.isfinite(dt) & (dt <= 0)
        # First sample: no energy/integral update
        in_lap.loc[0, "DeltaTimeValid"] = False

    meta["n_in_window"] = int(len(in_lap))
    meta["n_valid_samples"] = int(in_lap["SampleValid"].sum()) if len(in_lap) else 0
    if meta["n_valid_samples"] < config.THIN_LAP_MIN_VALID_SAMPLES:
        meta["thin_lap"] = True
        if len(in_lap):
            in_lap["DeployProxy"] = 0.0
            in_lap["HarvestProxy"] = 0.0
    in_lap["Driver"] = driver
    in_lap["LapNumber"] = lap_number
    in_lap["TrackStatus"] = str(lap_row["TrackStatus"])
    in_lap["freeze_flag"] = state["freeze_flag"]
    in_lap["yellow_flag"] = state["yellow_flag"]
    in_lap["IsAccurate"] = bool(lap_row["IsAccurate"])
    return in_lap, meta


def integrate_lap(samples: pd.DataFrame, meta: dict) -> dict:
    if samples is None or samples.empty or meta.get("thin_lap"):
        d_lap = np.nan if meta.get("thin_lap") or samples is None or samples.empty else 0.0
        h_lap = np.nan if meta.get("thin_lap") or samples is None or samples.empty else 0.0
        if meta.get("thin_lap"):
            d_lap, h_lap = np.nan, np.nan
        return {
            "D_lap": d_lap,
            "H_lap": h_lap,
            "max_dt_used": np.nan,
            "n_dt_gaps": 0,
            "mean_throttle": np.nan,
        }
    usable = samples["DeltaTimeValid"].fillna(False) & samples["SampleValid"].fillna(False)
    dt = samples.loc[usable, "DeltaTimeSeconds"].astype(float)
    d_lap = float((samples.loc[usable, "DeployProxy"] * dt).sum())
    h_lap = float((samples.loc[usable, "HarvestProxy"] * dt).sum())
    max_dt = float(dt.max()) if len(dt) else np.nan
    n_gaps = int(samples["DeltaTimeGap"].fillna(False).sum())
    mean_thr = float(pd.to_numeric(samples.loc[samples["SampleValid"], "Throttle"], errors="coerce").mean())
    return {
        "D_lap": d_lap,
        "H_lap": h_lap,
        "max_dt_used": max_dt,
        "n_dt_gaps": n_gaps,
        "mean_throttle": mean_thr,
    }


def is_calibration_lap(row: pd.Series) -> bool:
    return (
        is_green_only(row["TrackStatus"])
        and (not is_box_lap(row.get("PitInTime"), row.get("PitOutTime")))
        and bool(row["IsAccurate"])
        and (not bool(row.get("thin_lap", False)))
        and (not bool(row.get("freeze_flag", False)))
    )


def process_all_laps(session, laps_clean: pd.DataFrame, progress_every: int = 50):
    sample_frames = []
    rows = []
    n = len(laps_clean)
    for i, (_, lap_row) in enumerate(laps_clean.iterrows(), start=1):
        samples, meta = prepare_lap_telemetry(session, lap_row)
        integ = integrate_lap(samples, meta)
        row = {**meta, **integ}
        rows.append(row)
        if samples is not None and not samples.empty:
            keep_cols = [
                c
                for c in [
                    "Driver",
                    "LapNumber",
                    "Date",
                    "SessionTime",
                    "Distance",
                    "Speed",
                    "Throttle",
                    "Brake",
                    "tau",
                    "SpeedMSSmoothed",
                    "AccelerationMS2",
                    "DecelerationMS2",
                    "DeltaTimeSeconds",
                    "DeltaTimeValid",
                    "DeltaTimeGap",
                    "SampleValid",
                    "DeployProxy",
                    "HarvestProxy",
                    "LiftDecelProxy",
                    "HarvestGate",
                    "freeze_flag",
                    "yellow_flag",
                    "IsAccurate",
                    "TrackStatus",
                    "InLapWindow",
                ]
                if c in samples.columns
            ]
            sample_frames.append(samples.loc[:, keep_cols].copy())
        if progress_every and i % progress_every == 0:
            print(f"  processed {i}/{n} laps", flush=True)

    lap_table = pd.DataFrame(rows)
    samples_all = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return lap_table, samples_all
