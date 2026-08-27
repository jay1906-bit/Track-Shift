"""Original car-data extract, distance, Δt, smoothing, acceleration.

Integrals use get_car_data() only. Never get_telemetry().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def extract_car_data(lap):
    """Return padded original car stream with FastF1-derived Distance."""
    car = lap.get_car_data(pad=1, pad_side="both")
    car = car.add_distance()
    tel = car.copy()
    tel = tel.sort_values("Date").reset_index(drop=True)
    return tel


def in_lap_window(tel: pd.DataFrame, lap) -> pd.Series:
    """Keep samples in [LapStartTime, lap end], dropping pad samples."""
    start = getattr(lap, "LapStartTime", None)
    duration = getattr(lap, "Time", None)
    if "SessionTime" in tel.columns and start is not None and duration is not None:
        try:
            end = start + duration
            st = tel["SessionTime"]
            return (st >= start) & (st <= end)
        except Exception:
            pass
    # Fallback: drop the first and last padded rows if pad=1 both sides.
    mask = pd.Series(True, index=tel.index)
    if len(tel) >= 3:
        mask.iloc[0] = False
        mask.iloc[-1] = False
    return mask


def add_delta_time(tel: pd.DataFrame) -> pd.DataFrame:
    out = tel.copy()
    out["DeltaTimeSeconds"] = out["Date"].diff().dt.total_seconds()
    return out


def add_kinematics(tel: pd.DataFrame) -> pd.DataFrame:
    """Speed conversion, rolling-median smoother, clipped acceleration, deceleration."""
    out = tel.copy()
    speed = pd.to_numeric(out["Speed"], errors="coerce")
    speed = speed.clip(lower=config.SPEED_CLIP_KMH[0], upper=config.SPEED_CLIP_KMH[1])
    out["SpeedKmhClipped"] = speed
    out["SpeedMS"] = speed / 3.6

    window = config.ROLLING_MEDIAN_WINDOW
    out["SpeedMSSmoothed"] = (
        out["SpeedMS"].rolling(window=window, center=True, min_periods=1).median()
    )

    dt = out["DeltaTimeSeconds"]
    valid_dt = np.isfinite(dt) & (dt > 0) & (dt <= config.DT_GAP_SECONDS)
    dv = out["SpeedMSSmoothed"].diff()
    accel = dv / dt
    accel = accel.where(valid_dt, np.nan)
    lo, hi = config.A_CLIP_MS2
    out["AccelerationMS2Raw"] = accel
    out["AccelerationMS2"] = accel.clip(lower=lo, upper=hi)
    out["DecelerationMS2"] = np.where(
        out["AccelerationMS2"].isna(),
        np.nan,
        np.maximum(0.0, -out["AccelerationMS2"]),
    )
    out["DeltaTimeValid"] = valid_dt
    out["DeltaTimeGap"] = np.isfinite(dt) & (dt > config.DT_GAP_SECONDS)
    out["DeltaTimeNonPositive"] = np.isfinite(dt) & (dt <= 0)
    return out


def add_validity(tel: pd.DataFrame) -> pd.DataFrame:
    out = tel.copy()
    throttle = pd.to_numeric(out["Throttle"], errors="coerce")
    speed = pd.to_numeric(out["Speed"], errors="coerce")
    throttle_ok = throttle.between(config.THROTTLE_MIN, config.THROTTLE_MAX, inclusive="both")
    speed_ok = np.isfinite(speed)
    out["ThrottleRaw"] = throttle
    out["ThrottleValid"] = throttle_ok
    out["SpeedValid"] = speed_ok
    out["tau"] = np.where(throttle_ok, np.clip(throttle / 100.0, 0.0, 1.0), np.nan)
    out["SampleValid"] = throttle_ok & speed_ok
    return out
