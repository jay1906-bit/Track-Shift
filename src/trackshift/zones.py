"""CircuitInfo zones and 400 m lookahead (Approach C). Does not own energy state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _circuit_corners(session) -> tuple[pd.DataFrame, str]:
    """Corner markers with along-track Distance.

    FastF1 fills Distance from the fastest lap's merged telemetry. That raises
    when the fastest lap has empty position data (Monaco 2026 ANT lap 76).
    The detector is not involved; this only recovers CircuitInfo so zone_type
    and lookahead can be built.
    """
    from fastf1.mvapi.data import get_circuit_info as mv_get_circuit_info

    circuit_key = session.session_info["Meeting"]["Circuit"]["Key"]
    if circuit_key == 149 and session.session_info["Meeting"]["Circuit"].get("ShortName") == "Mugello":
        circuit_key = 146
    info = mv_get_circuit_info(year=int(session.event.year), circuit_key=int(circuit_key))
    if info is None:
        return pd.DataFrame(columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]), "unavailable"

    def _try(lap, source: str) -> str | None:
        try:
            pos = lap.get_pos_data()
        except Exception:
            return None
        if pos is None or pos.empty or "Date" not in pos.columns:
            return None
        try:
            info.add_marker_distance(reference_lap=lap)
        except Exception:
            return None
        if info.corners["Distance"].notna().any():
            return source
        return None

    try:
        source = _try(session.laps.pick_fastest(), "fastest_lap")
        if source:
            return info.corners.copy(), source
    except Exception:
        pass

    n = 0
    for _, lap in session.laps.iterlaps():
        n += 1
        source = _try(lap, f"fallback_lap_{n}")
        if source:
            return info.corners.copy(), source
        if n >= 200:
            break
    return info.corners.copy(), "distance_missing"


def build_zone_table(session, track_length_m: float) -> pd.DataFrame:
    corners, distance_source = _circuit_corners(session)
    if "Distance" in corners.columns:
        corners = corners.dropna(subset=["Distance"])
    corners = corners.sort_values("Distance").reset_index(drop=True) if len(corners) else corners
    L = float(track_length_m)
    segments = []
    cursor = 0.0
    straight_i = 1
    if corners.empty:
        segments.append(_seg("straight_01", "straight", 0.0, L, None))
        zones = pd.DataFrame(segments)
        zones["length_m"] = zones["end_m"] - zones["start_m"]
        zones.attrs["corner_distance_source"] = distance_source
        return zones
    for _, crow in corners.iterrows():
        cdist = float(crow["Distance"])
        cnum = int(crow["Number"])
        brake_start = max(0.0, cdist - config.BRAKE_ZONE_BEFORE_M)
        brake_end = max(0.0, cdist - config.CORNER_HALF_WIDTH_M)
        corner_start = max(0.0, cdist - config.CORNER_HALF_WIDTH_M)
        corner_end = min(L, cdist + config.CORNER_HALF_WIDTH_M)

        if brake_start < cursor:
            brake_start = cursor
        if brake_end < cursor:
            brake_end = cursor
        if corner_start < cursor:
            corner_start = cursor

        if brake_start > cursor + 1e-6:
            segments.append(_seg(f"straight_{straight_i:02d}", "straight", cursor, min(brake_start, L), None))
            straight_i += 1
            cursor = min(brake_start, L)

        if brake_end > cursor + 1e-6 and brake_end > brake_start:
            segments.append(_seg(f"brake_T{cnum}", "brake", cursor, min(brake_end, L), cnum))
            cursor = min(brake_end, L)

        if corner_end > cursor + 1e-6:
            start = max(cursor, corner_start)
            if corner_end > start:
                segments.append(_seg(f"corner_T{cnum}", "corner", start, corner_end, cnum))
                cursor = corner_end

    if cursor < L - 1e-6:
        segments.append(_seg(f"straight_{straight_i:02d}", "straight", cursor, L, None))

    zones = pd.DataFrame(segments)
    zones["length_m"] = zones["end_m"] - zones["start_m"]
    zones = zones.loc[zones["length_m"] > 1e-6].reset_index(drop=True)
    zones.attrs["corner_distance_source"] = distance_source
    return zones


def _seg(zone_id, zone_type, start, end, corner_number):
    return {
        "zone_id": zone_id,
        "zone_type": zone_type,
        "start_m": float(start),
        "end_m": float(end),
        "corner_number": corner_number,
    }


def assign_zone(distance: np.ndarray, zones: pd.DataFrame) -> np.ndarray:
    d = np.asarray(distance, dtype=float)
    out = np.array(["unassigned"] * len(d), dtype=object)
    starts = zones["start_m"].to_numpy(dtype=float)
    ends = zones["end_m"].to_numpy(dtype=float)
    ids = zones["zone_id"].to_numpy()
    for i in range(len(zones)):
        mask = (d >= starts[i]) & (d < ends[i])
        out[mask] = ids[i]
    last = len(zones) - 1
    out[(d >= starts[last]) & (d <= ends[last])] = ids[last]
    out[d < starts[0]] = ids[0]
    out[d > ends[last]] = ids[last]
    return out


def zone_baselines(samples: pd.DataFrame, zones: pd.DataFrame, cal_keys: pd.DataFrame) -> pd.DataFrame:
    """Calibration-set zone statistics. Observed telemetry summaries, not ERS maps."""
    key = samples.merge(cal_keys, on=["Driver", "LapNumber"], how="inner")
    if key.empty:
        raise RuntimeError("No calibration samples for zone baselines.")
    grouped = key.groupby("zone_id", dropna=False)
    stats = grouped.agg(
        n_samples=("zone_id", "size"),
        mean_speed=("Speed", "mean"),
        median_speed=("Speed", "median"),
        mean_throttle=("Throttle", "mean"),
        throttle_ge_0_8_frac=("tau", lambda s: float((s >= 0.8).mean()) if len(s) else np.nan),
        brake_frac=("Brake", lambda s: float(s.astype(bool).mean()) if len(s) else np.nan),
        mean_DeployProxy=("DeployProxy", "mean"),
        mean_HarvestProxy=("HarvestProxy", "mean"),
    ).reset_index()
    out = zones.merge(stats, on="zone_id", how="left")
    for col in [
        "n_samples",
        "mean_speed",
        "median_speed",
        "mean_throttle",
        "throttle_ge_0_8_frac",
        "brake_frac",
        "mean_DeployProxy",
        "mean_HarvestProxy",
    ]:
        if col not in out:
            out[col] = np.nan
    out["n_samples"] = out["n_samples"].fillna(0).astype(int)
    out["mean_DeployProxy"] = out["mean_DeployProxy"].fillna(0.0)
    out["mean_HarvestProxy"] = out["mean_HarvestProxy"].fillna(0.0)
    return out


def lookahead_from_distance(distance_m: float, zones: pd.DataFrame, horizon_m: float | None = None) -> dict:
    """Session-baseline lookahead over the next horizon metres. No wrap-around in v1."""
    horizon_m = config.LOOKAHEAD_M if horizon_m is None else horizon_m
    start = float(distance_m)
    end = start + float(horizon_m)
    deploy = 0.0
    harvest = 0.0
    covered = 0.0
    for _, z in zones.iterrows():
        a = max(start, float(z["start_m"]))
        b = min(end, float(z["end_m"]))
        if b > a:
            length = b - a
            deploy += float(z["mean_DeployProxy"]) * length
            harvest += float(z["mean_HarvestProxy"]) * length
            covered += length
    return {
        "LookaheadDeployProxy": deploy,
        "LookaheadHarvestProxy": harvest,
        "LookaheadCoveredM": covered,
    }


def add_sample_lookahead(samples: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    vals = [lookahead_from_distance(d, zones) for d in out["Distance"].to_numpy(dtype=float)]
    look = pd.DataFrame(vals, index=out.index)
    return pd.concat([out, look], axis=1)


def lap_start_lookahead(samples: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Lookahead at the first in-lap sample (start of lap)."""
    first = (
        samples.sort_values(["Driver", "LapNumber", "Date"])
        .groupby(["Driver", "LapNumber"], as_index=False)
        .head(1)
    )
    look = add_sample_lookahead(first.loc[:, ["Distance"]], zones)
    out = first.loc[:, ["Driver", "LapNumber", "Distance"]].copy()
    out["LookaheadDeployProxy"] = look["LookaheadDeployProxy"].to_numpy()
    out["LookaheadHarvestProxy"] = look["LookaheadHarvestProxy"].to_numpy()
    out["LookaheadCoveredM"] = look["LookaheadCoveredM"].to_numpy()
    out = out.rename(columns={"Distance": "LapStartDistance"})
    return out
