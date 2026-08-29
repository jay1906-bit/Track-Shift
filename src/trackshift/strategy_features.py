"""Strategy feature integration: derived laps + FastF1 car-data driver-ahead.

This module does not compute or modify EstimatedEnergyIndex, proxies, zones,
lookahead, or ATTACK/SAVE/DEFEND rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .session_io import pick_session_lap
from .telemetry import in_lap_window


AHEAD_NOTE = (
    "DistanceToDriverAhead is a FastF1-derived, speed-integrated on-track "
    "distance in metres (Telemetry.add_driver_ahead / calculate_driver_ahead). "
    "It is not an official FIA timing gap, Interval, or GapToLeader. "
    "DriverAhead is the car-number string of the nearest car with a positive "
    "integrated-distance delta. Empty DriverAhead means no car ahead. "
    "Cars in the pit lane are not excluded (FastF1 documented limitation). "
    "Computed per lap on original get_car_data() timestamps; get_telemetry() "
    "is not used."
)

LAP_SUMMARY_NOTE = (
    "Lap-level ahead columns are aggregations of the same per-lap car-data "
    "DriverAhead / DistanceToDriverAhead series. They are not a second gap "
    "model. DriverAhead_end / DistanceToDriverAhead_end are the last in-lap "
    "car-data sample. DistanceToDriverAhead_median is the median of finite "
    "in-lap DistanceToDriverAhead values."
)


def build_derived_laps(laps_clean: pd.DataFrame) -> pd.DataFrame:
    """Persist PositionChange, LapTimeSeconds, LapTimeDelta on the 995-row spine.

    First recorded lap per driver is NaN (groupby.diff). NaNs are not filled.
    Negative PositionChange means gained a place.
    """
    out = laps_clean.copy()
    out["LapNumber"] = out["LapNumber"].astype(int)
    out["LapTimeSeconds"] = pd.to_timedelta(out["LapTime"]).dt.total_seconds()
    out = out.sort_values(["Driver", "LapNumber"], kind="mergesort").reset_index(drop=True)
    out["LapTimeDelta"] = out.groupby("Driver", sort=False)["LapTimeSeconds"].diff()
    out["PositionChange"] = out.groupby("Driver", sort=False)["Position"].diff()
    cols = [
        "Driver",
        "DriverNumber",
        "LapNumber",
        "Position",
        "LapTimeSeconds",
        "LapTimeDelta",
        "PositionChange",
    ]
    return out.loc[:, cols]


def _normalize_driver_ahead(value) -> str:
    """Return a car-number string, or '' if nobody is ahead.

    CSV round-trips may yield 41.0 / '41.0'; FastF1 uses '41'.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass
    if isinstance(value, (float, int, np.floating, np.integer)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer() and float(value) >= 0:
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "<na>"}:
        return ""
    try:
        as_float = float(text)
        if np.isfinite(as_float) and as_float.is_integer() and as_float >= 0:
            return str(int(as_float))
    except (TypeError, ValueError):
        pass
    return text


def extract_lap_driver_ahead(session, lap_row: pd.Series) -> pd.DataFrame:
    """Per-lap original car data + add_driver_ahead(). No get_telemetry()."""
    driver = str(lap_row["Driver"])
    lap_number = int(lap_row["LapNumber"])
    empty = pd.DataFrame(
        columns=["Driver", "LapNumber", "Date", "SessionTime", "DriverAhead", "DistanceToDriverAhead"]
    )
    try:
        session_lap = pick_session_lap(session, driver, lap_number)
        car = session_lap.get_car_data(pad=1, pad_side="both")
    except Exception:
        return empty

    if car is None or len(car) < 3:
        return empty

    # FastF1 get_telemetry() computes ahead on unpadded car_data (iloc[1:-1])
    # before any pos merge. We stop there: original car clock, no interpolation.
    try:
        car_in = car.iloc[1:-1].add_driver_ahead()
    except Exception:
        return empty

    if car_in is None or car_in.empty:
        return empty
    window = in_lap_window(car_in, session_lap)
    car_in = car_in.loc[window]
    if car_in is None or car_in.empty:
        return empty

    out = pd.DataFrame(
        {
            "Driver": driver,
            "LapNumber": lap_number,
            "Date": car_in["Date"].to_numpy(),
            "SessionTime": car_in["SessionTime"].to_numpy() if "SessionTime" in car_in.columns else pd.NaT,
            "DriverAhead": car_in["DriverAhead"].map(_normalize_driver_ahead).to_numpy(),
            "DistanceToDriverAhead": pd.to_numeric(car_in["DistanceToDriverAhead"], errors="coerce").to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def extract_all_driver_ahead(session, laps_clean: pd.DataFrame, progress_every: int = 50) -> pd.DataFrame:
    frames = []
    n = len(laps_clean)
    for i, (_, lap_row) in enumerate(laps_clean.iterrows(), start=1):
        frame = extract_lap_driver_ahead(session, lap_row)
        if frame is not None and not frame.empty:
            frames.append(frame)
        if progress_every and i % progress_every == 0:
            print(f"  driver-ahead {i}/{n} laps", flush=True)
    if not frames:
        return pd.DataFrame(
            columns=["Driver", "LapNumber", "Date", "SessionTime", "DriverAhead", "DistanceToDriverAhead"]
        )
    return pd.concat(frames, ignore_index=True)


def summarize_driver_ahead(samples: pd.DataFrame, laps_clean: pd.DataFrame) -> pd.DataFrame:
    """Last-sample and median aggregations of the FastF1 car-data ahead columns."""
    spine = laps_clean.loc[:, ["Driver", "LapNumber"]].copy()
    spine["LapNumber"] = spine["LapNumber"].astype(int)
    if samples is None or samples.empty:
        spine["DriverAhead_end"] = ""
        spine["DistanceToDriverAhead_end"] = np.nan
        spine["DistanceToDriverAhead_median"] = np.nan
        spine["n_samples_with_car_ahead"] = 0
        spine["n_car_samples_ahead_extract"] = 0
        return spine

    work = samples.copy()
    work["LapNumber"] = work["LapNumber"].astype(int)
    work["has_ahead"] = work["DriverAhead"].fillna("").astype(str).str.strip().ne("")
    work["DistanceToDriverAhead"] = pd.to_numeric(work["DistanceToDriverAhead"], errors="coerce")
    work = work.sort_values(["Driver", "LapNumber", "Date"], kind="mergesort")

    last = work.groupby(["Driver", "LapNumber"], sort=False).tail(1)
    last = last.loc[:, ["Driver", "LapNumber", "DriverAhead", "DistanceToDriverAhead"]].rename(
        columns={
            "DriverAhead": "DriverAhead_end",
            "DistanceToDriverAhead": "DistanceToDriverAhead_end",
        }
    )
    last["DriverAhead_end"] = last["DriverAhead_end"].map(_normalize_driver_ahead)

    med = (
        work.groupby(["Driver", "LapNumber"], sort=False)["DistanceToDriverAhead"]
        .median()
        .rename("DistanceToDriverAhead_median")
        .reset_index()
    )
    counts = (
        work.groupby(["Driver", "LapNumber"], sort=False)
        .agg(
            n_samples_with_car_ahead=("has_ahead", "sum"),
            n_car_samples_ahead_extract=("DriverAhead", "size"),
        )
        .reset_index()
    )
    counts["n_samples_with_car_ahead"] = counts["n_samples_with_car_ahead"].astype(int)
    counts["n_car_samples_ahead_extract"] = counts["n_car_samples_ahead_extract"].astype(int)

    out = spine.merge(last, on=["Driver", "LapNumber"], how="left")
    out = out.merge(med, on=["Driver", "LapNumber"], how="left")
    out = out.merge(counts, on=["Driver", "LapNumber"], how="left")
    out["DriverAhead_end"] = out["DriverAhead_end"].map(_normalize_driver_ahead)
    out["n_samples_with_car_ahead"] = out["n_samples_with_car_ahead"].fillna(0).astype(int)
    out["n_car_samples_ahead_extract"] = out["n_car_samples_ahead_extract"].fillna(0).astype(int)
    return out


def join_strategy_features(
    laps_clean: pd.DataFrame,
    derived: pd.DataFrame,
    trackaware: pd.DataFrame,
    ahead_laps: pd.DataFrame,
) -> pd.DataFrame:
    """995-row strategy table. Energy/lookahead columns are copied, not recomputed."""
    spine = laps_clean.loc[:, ["Driver", "DriverNumber", "LapNumber", "Position"]].copy()
    spine["LapNumber"] = spine["LapNumber"].astype(int)
    spine = spine.sort_values(["Driver", "LapNumber"], kind="mergesort").reset_index(drop=True)

    derived_cols = ["Driver", "LapNumber", "PositionChange", "LapTimeSeconds", "LapTimeDelta"]
    energy_cols = [
        "Driver",
        "LapNumber",
        "EstimatedEnergyIndex",
        "LookaheadDeployProxy",
        "LookaheadHarvestProxy",
        "LookaheadCoveredM",
    ]
    out = spine.merge(derived.loc[:, derived_cols], on=["Driver", "LapNumber"], how="left")
    out = out.merge(trackaware.loc[:, energy_cols], on=["Driver", "LapNumber"], how="left")
    ahead_keep = [
        c
        for c in [
            "Driver",
            "LapNumber",
            "DriverAhead_end",
            "DistanceToDriverAhead_end",
            "DistanceToDriverAhead_median",
            "n_samples_with_car_ahead",
            "n_car_samples_ahead_extract",
        ]
        if c in ahead_laps.columns
    ]
    out = out.merge(ahead_laps.loc[:, ahead_keep], on=["Driver", "LapNumber"], how="left")
    return out


def validate_strategy_features(
    *,
    laps_clean: pd.DataFrame,
    derived: pd.DataFrame,
    samples: pd.DataFrame,
    strategy: pd.DataFrame,
    trackaware_before: pd.DataFrame,
    energy_v1_before: pd.DataFrame,
    trackaware_after: pd.DataFrame,
    energy_v1_after: pd.DataFrame,
    session_drivers: list[str],
    raw_hash_before: str,
    clean_hash_before: str,
    raw_hash_after: str,
    clean_hash_after: str,
    energy_hashes_before: dict[str, str],
    energy_hashes_after: dict[str, str],
    reference_car_dates: pd.Series | None,
) -> dict:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    tmp = laps_clean.copy()
    tmp["LapNumber"] = tmp["LapNumber"].astype(int)
    tmp = tmp.sort_values(["Driver", "LapNumber"], kind="mergesort")
    tmp["expected_pc"] = tmp.groupby("Driver", sort=False)["Position"].diff()
    pc_join = derived.merge(tmp.loc[:, ["Driver", "LapNumber", "expected_pc"]], on=["Driver", "LapNumber"], how="left")
    pos_ok = bool(
        np.isclose(
            pd.to_numeric(pc_join["PositionChange"], errors="coerce"),
            pd.to_numeric(pc_join["expected_pc"], errors="coerce"),
            equal_nan=True,
        ).all()
    )
    add("position_change_matches_diff", pos_ok, "PositionChange = current Position - previous Position per driver")

    first_idx = derived.groupby("Driver", sort=False).head(1).index
    first_nan = bool(derived.loc[first_idx, "PositionChange"].isna().all())
    add("first_lap_position_change_nan", first_nan, f"n first laps={len(first_idx)}")

    for name, df in [("derived", derived), ("strategy", strategy), ("clean", laps_clean)]:
        dups = int(df.duplicated(["Driver", "LapNumber"]).sum())
        add(f"{name}_driver_lap_unique", dups == 0, f"duplicates={dups}")

    add("derived_995", len(derived) == 995, f"derived rows={len(derived)}")
    add("strategy_995", len(strategy) == 995, f"strategy rows={len(strategy)}")
    add("clean_995", len(laps_clean) == 995, f"clean rows={len(laps_clean)}")

    add("raw_unchanged", raw_hash_before == raw_hash_after, f"raw sha256={raw_hash_after}")
    add("clean_unchanged", clean_hash_before == clean_hash_after, f"clean sha256={clean_hash_after}")

    valid_nums = {str(x).strip() for x in session_drivers}
    ahead = samples["DriverAhead"].map(_normalize_driver_ahead) if len(samples) else pd.Series(dtype=str)
    bad_ahead = ahead[(ahead != "") & ~ahead.isin(valid_nums)]
    add(
        "driver_ahead_car_number_or_empty",
        len(bad_ahead) == 0,
        f"n_invalid={len(bad_ahead)}; unique_ahead={sorted(ahead[ahead != ''].unique().tolist())[:30]}",
    )

    dist = pd.to_numeric(samples["DistanceToDriverAhead"], errors="coerce") if len(samples) else pd.Series(dtype=float)
    present = dist.notna()
    n_neg = int((dist[present] < 0).sum()) if present.any() else 0
    add("distance_to_ahead_non_negative", n_neg == 0, f"n_negative={n_neg}; n_present={int(present.sum())}")
    add("no_impossible_negative_gaps", n_neg == 0, f"n_negative={n_neg}")

    date_ok = True
    date_detail = "no samples"
    if len(samples) and "Date" in samples.columns:
        dates = pd.to_datetime(samples["Date"], errors="coerce")
        date_ok = bool(dates.notna().all())
        date_detail = f"n_dates={int(dates.notna().sum())}; n_null={int(dates.isna().sum())}"
        if reference_car_dates is not None and len(reference_car_dates):
            ref = pd.to_datetime(pd.Series(reference_car_dates), errors="coerce").dropna()
            got = dates[
                (samples["Driver"] == "NOR") & (samples["LapNumber"].astype(int) == 5)
            ].dropna()
            if len(ref) and len(got):
                ref_ns = set(pd.to_datetime(ref).astype("int64").tolist())
                got_ns = pd.to_datetime(got).astype("int64")
                missing = int((~got_ns.isin(ref_ns)).sum())
                date_ok = date_ok and missing == 0
                date_detail += f"; NOR lap 5 dates not in get_car_data={missing}"
    add("sample_ahead_on_car_data_timestamps", date_ok, date_detail)

    energy_files_ok = energy_hashes_before == energy_hashes_after
    add(
        "energy_files_untouched",
        energy_files_ok,
        f"before={energy_hashes_before}; after={energy_hashes_after}",
    )

    e_cols = ["EstimatedEnergyIndex", "LookaheadDeployProxy", "LookaheadHarvestProxy", "LookaheadCoveredM"]
    before_e = trackaware_before.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    after_e = trackaware_after.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    energy_values_ok = True
    energy_detail = "compared trackaware columns"
    for col in e_cols:
        if col not in before_e.columns or col not in after_e.columns:
            energy_values_ok = False
            energy_detail = f"missing {col}"
            break
        if not np.allclose(
            pd.to_numeric(before_e[col], errors="coerce"),
            pd.to_numeric(after_e[col], errors="coerce"),
            equal_nan=True,
        ):
            energy_values_ok = False
            energy_detail = f"{col} changed"
            break
    add("energy_values_identical", energy_values_ok, energy_detail)

    v1_ok = energy_v1_before.equals(energy_v1_after)
    add("energy_v1_identical", v1_ok, f"energy_v1 rows={len(energy_v1_after)}")

    joined_e = strategy.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    src_e = before_e.loc[:, ["Driver", "LapNumber", *e_cols]].reset_index(drop=True)
    merged = joined_e.loc[:, ["Driver", "LapNumber", *e_cols]].merge(
        src_e, on=["Driver", "LapNumber"], suffixes=("_join", "_src")
    )
    join_e_ok = True
    join_e_detail = "join copies match trackaware"
    for col in e_cols:
        if not np.allclose(
            pd.to_numeric(merged[f"{col}_join"], errors="coerce"),
            pd.to_numeric(merged[f"{col}_src"], errors="coerce"),
            equal_nan=True,
        ):
            join_e_ok = False
            join_e_detail = f"join mismatch on {col}"
            break
    add("joined_energy_lookahead_identical_to_source", join_e_ok, join_e_detail)
    add("lookahead_values_identical", energy_values_ok and join_e_ok, energy_detail)

    add(
        "no_attack_save_defend_columns",
        not any(c in strategy.columns for c in ["ATTACK", "SAVE", "DEFEND", "Mode", "StrategyMode"]),
        f"columns={list(strategy.columns)}",
    )

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "n_passed": int(sum(c["passed"] for c in checks)),
        "n_checks": len(checks),
        "checks": checks,
    }
