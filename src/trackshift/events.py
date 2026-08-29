"""Australia overtake-event detection (Phases 1-7).

Builds pairwise attacker/target proximity episodes from the existing
sample-level DriverAhead extract and the existing sample-level energy walk.

This module does not compute or modify EstimatedEnergyIndex, DeployProxy,
HarvestProxy, alpha/beta, zones, or the lookahead formula. It reads the
frozen outputs and reuses ``zones.zone_baselines`` / ``zones.lookahead_from_distance``.

No ATTACK / SAVE / DEFEND logic and no ML lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config
from .zones import lookahead_from_distance, zone_baselines

SAMPLE_KEY = ["Driver", "LapNumber", "Date"]

EVENT_TYPES = [
    "ON_TRACK_PASS",
    "CLOSE_INTERACTION_NO_PASS",
    "PIT_RELATED",
    "RETIREMENT_RELATED",
    "LAPPED_UNLAPPED",
    "UNCERTAIN",
    "INVALID",
]

GAP_UNITS_NOTE = (
    "gap_at_start_m and DistanceToDriverAhead are FastF1 speed-integrated "
    "distances in METRES (Telemetry.add_driver_ahead). They are not FIA "
    "timing gaps and not seconds. gap_at_start_s_est is a derived estimate "
    "only, computed as metres / attacker speed at t0."
)


@dataclass(frozen=True)
class EventParams:
    """Detection geometry. Chosen from sampling rate and track scale only."""

    gap_max_m: float = config.EVENT_GAP_MAX_M
    gap_min_m: float = config.EVENT_GAP_MIN_M
    min_persist_s: float = config.EVENT_MIN_PERSIST_S
    horizon_s: float = config.EVENT_HORIZON_S
    merge_gap_s: float = config.EVENT_MERGE_GAP_S
    closing_window_s: float = config.EVENT_CLOSING_WINDOW_S
    asof_tolerance_ms: int = config.EVENT_ASOF_TOLERANCE_MS
    lapped_fraction: float = config.EVENT_LAPPED_FRACTION
    gap_jump_fraction: float = config.EVENT_GAP_JUMP_FRACTION
    exclude_lap_numbers: tuple = config.EVENT_EXCLUDE_LAP_NUMBERS
    horizon_anchor: str = config.EVENT_HORIZON_ANCHOR
    audit_horizon_s: float = config.EVENT_AUDIT_HORIZON_S

    def to_dict(self) -> dict:
        out = asdict(self)
        out["exclude_lap_numbers"] = list(self.exclude_lap_numbers)
        return out


def horizon_end(t0: pd.Timestamp, event_end_time, params: EventParams) -> pd.Timestamp:
    """End of the window the label is allowed to look at.

    ``"t0"`` is the final definition: the window is [t0, t0 + horizon_s] and its
    length does not depend on how long the two cars happened to stay together.
    t0 is the decision point, so the question the label answers is fixed.

    ``"event_end"`` reproduces the v1-v3 behaviour, where the window ran to the
    end of the engagement plus horizon_s and therefore stretched to 423 s on the
    longest battle. Kept only so the historical runs stay reproducible.
    """
    if params.horizon_anchor == "t0":
        return pd.Timestamp(t0) + pd.Timedelta(seconds=params.horizon_s)
    if params.horizon_anchor == "event_end":
        return pd.Timestamp(event_end_time) + pd.Timedelta(seconds=params.horizon_s)
    raise ValueError(f"unknown horizon_anchor {params.horizon_anchor!r}")


# ----------------------------------------------------------------------
# Phase 1-2: inputs and joins
# ----------------------------------------------------------------------


def build_driver_number_map(laps_clean: pd.DataFrame) -> dict[str, str]:
    """Car-number string -> driver abbreviation, from the cleaned lap spine."""
    pairs = laps_clean.loc[:, ["DriverNumber", "Driver"]].dropna().drop_duplicates()
    mapping: dict[str, str] = {}
    for num, drv in pairs.itertuples(index=False):
        key = str(int(float(num)))
        existing = mapping.get(key)
        if existing is not None and existing != drv:
            raise RuntimeError(f"Car number {key} maps to both {existing} and {drv}")
        mapping[key] = str(drv)
    return mapping


def session_epoch(samples: pd.DataFrame) -> pd.Timestamp:
    """Absolute timestamp of SessionTime zero, so lap/pit timedeltas become Dates."""
    offsets = pd.to_datetime(samples["Date"]) - pd.to_timedelta(samples["SessionTime"])
    spread = (offsets.max() - offsets.min()).total_seconds()
    if abs(spread) > 1.0:
        raise RuntimeError(f"Date - SessionTime is not constant (spread {spread} s).")
    return pd.Timestamp(offsets.median())


def join_attacker_samples(
    ahead: pd.DataFrame,
    energy_samples: pd.DataFrame,
    laps_clean: pd.DataFrame,
    params: EventParams,
) -> tuple[pd.DataFrame, dict]:
    """Join the ahead extract onto the energy samples and the lap spine.

    Primary key is Driver + LapNumber + Date. (Driver, Date) alone is not
    unique because lap-boundary timestamps appear on two adjacent laps.
    """
    stats: dict = {}
    ahead = ahead.copy()
    ahead["LapNumber"] = ahead["LapNumber"].astype(int)
    ahead["Date"] = pd.to_datetime(ahead["Date"])
    ahead["DriverAhead"] = ahead["DriverAhead"].fillna("").astype(str).str.strip()

    energy = energy_samples.copy()
    energy["LapNumber"] = energy["LapNumber"].astype(int)
    energy["Date"] = pd.to_datetime(energy["Date"])

    stats["n_ahead_rows"] = int(len(ahead))
    stats["n_energy_rows"] = int(len(energy))
    stats["ahead_duplicate_keys"] = int(ahead.duplicated(SAMPLE_KEY).sum())
    stats["energy_duplicate_keys"] = int(energy.duplicated(SAMPLE_KEY).sum())

    energy_cols = SAMPLE_KEY + [
        "SessionTime",
        "Distance",
        "Speed",
        "EstimatedEnergyIndex",
        "zone_id",
        "freeze_flag",
        "yellow_flag",
        "TrackStatus",
        "SampleValid",
    ]
    energy_slim = energy.loc[:, [c for c in energy_cols if c in energy.columns]]

    joined = ahead.merge(energy_slim, on=SAMPLE_KEY, how="left", suffixes=("", "_energy"))
    matched = joined["Distance"].notna()
    stats["exact_sample_matches"] = int(matched.sum())
    stats["unmatched_samples_before_asof"] = int((~matched).sum())
    stats["exact_match_rate"] = float(matched.mean())

    # Causal backward asof only for rows the exact key missed. Never forward:
    # a forward match would import a sample from after the attacker timestamp.
    n_asof = 0
    if (~matched).any():
        missing = joined.loc[~matched, ["Driver", "Date"]].copy()
        filled = []
        for drv, grp in missing.groupby("Driver", sort=False):
            src = energy_slim.loc[energy_slim["Driver"] == drv].sort_values("Date")
            if src.empty:
                continue
            got = pd.merge_asof(
                grp.sort_values("Date"),
                src.drop(columns=["Driver"]),
                on="Date",
                direction="backward",
                tolerance=pd.Timedelta(milliseconds=params.asof_tolerance_ms),
            )
            got["Driver"] = drv
            filled.append(got)
        if filled:
            fill = pd.concat(filled, ignore_index=True)
            n_asof = int(fill["Distance"].notna().sum())
        stats["asof_backward_matches"] = n_asof
    else:
        stats["asof_backward_matches"] = 0

    stats["unmatched_samples_final"] = int(joined["Distance"].isna().sum()) - n_asof

    lap_cols = [
        "Driver",
        "LapNumber",
        "DriverNumber",
        "Position",
        "Compound",
        "TyreLife",
        "Stint",
        "PitInTime",
        "PitOutTime",
        "LapStartTime",
        "Time",
    ]
    laps = laps_clean.copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    joined = joined.merge(
        laps.loc[:, [c for c in lap_cols if c in laps.columns]],
        on=["Driver", "LapNumber"],
        how="left",
    )
    stats["lap_join_match_rate"] = float(joined["Position"].notna().mean())
    joined["gap_m"] = pd.to_numeric(joined["DistanceToDriverAhead"], errors="coerce")
    return joined, stats


def add_cumulative_distance(df: pd.DataFrame, track_length_m: float) -> pd.DataFrame:
    """Race distance covered by the driver, used for pairwise ordering."""
    out = df.copy()
    out["cum_distance_m"] = (out["LapNumber"].astype(float) - 1.0) * float(
        track_length_m
    ) + out["Distance"].astype(float)
    return out


def build_lap_lengths(samples: pd.DataFrame) -> pd.DataFrame:
    """Measured length of each lap. Pass the COMPLETE telemetry, not a subset.

    Derived frames such as the DriverAhead extract can be missing an entire lap,
    and a missing lap would silently remove a whole lap from every later offset.
    """
    return (
        samples.groupby(["Driver", "LapNumber"])["Distance"]
        .max()
        .rename("lap_length_m")
        .reset_index()
    )


def build_lap_offsets(lap_lengths: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Race distance completed before each lap starts, per driver.

    Each driver's lap index is reindexed over the full 1..N range so a lap that
    is absent from the source cannot shorten the running total. Absent laps are
    filled with that driver's median lap length.
    """
    frames = []
    filled: dict[str, list[int]] = {}
    for driver, grp in lap_lengths.groupby("Driver", sort=True):
        series = grp.set_index("LapNumber")["lap_length_m"].sort_index()
        full = pd.RangeIndex(1, int(series.index.max()) + 1, name="LapNumber")
        reindexed = series.reindex(full)
        gaps = [int(lap) for lap in reindexed.index[reindexed.isna()]]
        if gaps:
            filled[str(driver)] = gaps
        reindexed = reindexed.fillna(series.median())
        offsets = reindexed.cumsum() - reindexed
        frames.append(
            pd.DataFrame(
                {
                    "Driver": driver,
                    "LapNumber": reindexed.index.astype(int),
                    "lap_offset_m": offsets.to_numpy(dtype=float),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    return table, {"laps_gap_filled": filled, "n_laps_gap_filled": sum(len(v) for v in filled.values())}


def add_race_distance(
    df: pd.DataFrame, lap_lengths: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Monotone per-driver race distance built from each lap's own length.

    ``cum_distance_m`` multiplies LapNumber by a single track length, which
    overshoots every real lap and so steps by a median of ~244 m at each line
    crossing. That is harmless for a same-instant comparison but fatal for any
    separation differenced over time.

    ``lap_lengths`` should come from the complete telemetry via
    ``build_lap_lengths``. Deriving it from ``df`` is only safe when ``df``
    contains every lap.
    """
    out = df.copy()
    source = lap_lengths if lap_lengths is not None else build_lap_lengths(out)
    offsets, _ = build_lap_offsets(source)
    out = out.merge(offsets, on=["Driver", "LapNumber"], how="left")
    out["race_distance_m"] = out["lap_offset_m"] + out["Distance"].astype(float)
    return out.drop(columns=["lap_offset_m"])


def speed_at_track_position(
    frame: pd.DataFrame,
    distance_m: float,
    t0: pd.Timestamp,
    lookback_s: float,
) -> tuple[float, float]:
    """Target speed the last time it crossed ``distance_m``, at or before t0.

    The target is ahead, so it already drove the attacker's current piece of
    track. Reading its speed there compares the two cars over the same corner or
    straight instead of at the same instant in different places.

    Returns (speed_kmh, lag_s) where lag_s is how long before t0 that crossing
    happened. Both bracketing samples lie at or before t0, so this is causal.
    """
    if frame is None or frame.empty or not np.isfinite(distance_m):
        return float("nan"), float("nan")
    lo = t0 - pd.Timedelta(seconds=lookback_s)
    window = frame.loc[(frame["Date"] >= lo) & (frame["Date"] <= t0)]
    if len(window) < 2:
        return float("nan"), float("nan")

    dist = window["Distance"].to_numpy(dtype=float)
    speed = window["Speed"].to_numpy(dtype=float)
    times = window["Date"].to_numpy()

    # Upward crossings only, so a lap reset (a downward step) never matches.
    hits = np.flatnonzero((dist[:-1] <= distance_m) & (dist[1:] > distance_m))
    if hits.size == 0:
        return float("nan"), float("nan")
    i = int(hits[-1])

    span = dist[i + 1] - dist[i]
    frac = float((distance_m - dist[i]) / span) if span > 0 else 0.0
    frac = min(max(frac, 0.0), 1.0)
    speed_at = float(speed[i] + frac * (speed[i + 1] - speed[i]))
    t_cross = times[i] + (times[i + 1] - times[i]) * frac
    lag_s = float((t0 - pd.Timestamp(t_cross)).total_seconds())
    return speed_at, lag_s


def closing_speed_from_separation(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    t0: pd.Timestamp,
    window_s: float,
) -> tuple[float, int]:
    """Pre-t0 approach rate from the two cars' own race distances.

    Positive means the attacker is closing. This never consults DriverAhead, so
    it survives the identity flicker that made the v1 feature missing for half
    the passes. The target track is linearly interpolated onto the attacker's
    timestamps, and the window is trimmed to the target's last sample at or
    before t0 so no interpolation ever reaches past t0.
    """
    if a_frame is None or b_frame is None or a_frame.empty or b_frame.empty:
        return float("nan"), 0

    b_hist = b_frame.loc[b_frame["Date"] <= t0]
    if len(b_hist) < 2:
        return float("nan"), 0
    b_last = b_hist["Date"].iloc[-1]

    lo = t0 - pd.Timedelta(seconds=window_s)
    a_win = a_frame.loc[(a_frame["Date"] >= lo) & (a_frame["Date"] <= min(t0, b_last))]
    a_win = a_win.dropna(subset=["race_distance_m"])
    if len(a_win) < 3:
        return float("nan"), int(len(a_win))

    a_times = a_win["Date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    b_times = b_hist["Date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    b_dist = b_hist["race_distance_m"].to_numpy(dtype=float)
    b_interp = np.interp(a_times, b_times, b_dist)

    separation = b_interp - a_win["race_distance_m"].to_numpy(dtype=float)
    seconds = (a_times - a_times[0]) / 1e9
    if seconds[-1] - seconds[0] <= 0:
        return float("nan"), int(len(a_win))

    # Least-squares slope over the whole window rather than an endpoint
    # difference, so per-sample distance noise does not dominate.
    slope = float(np.polyfit(seconds, separation, 1)[0])
    return float(-slope * 3.6), int(len(a_win))


def add_track_position_speed_features(
    events: pd.DataFrame,
    by_driver: dict[str, pd.DataFrame],
    lookback_s: float,
    window_s: float,
    max_lag_s: float,
) -> pd.DataFrame:
    """v2 competitor-speed features, aligned by track position rather than clock.

    Depends only on t0, the attacker's t0 track distance, and the two cars' own
    telemetry, so it is unaffected by how the detector labelled the event.
    """
    records = []
    for e in events.itertuples(index=False):
        t0 = pd.Timestamp(e.event_start_time)
        a_frame = by_driver.get(e.attacker)
        b_frame = by_driver.get(e.target) if isinstance(e.target, str) and e.target else None
        distance = (
            float(e.track_distance_at_start_m) if pd.notna(e.track_distance_at_start_m) else np.nan
        )
        attacker_speed = (
            float(e.attacker_speed_at_start_kmh) if pd.notna(e.attacker_speed_at_start_kmh) else np.nan
        )

        target_speed_pos, lag_s = speed_at_track_position(b_frame, distance, t0, lookback_s)
        closing_kmh, n_win = closing_speed_from_separation(a_frame, b_frame, t0, window_s)

        alignment_ok = bool(np.isfinite(lag_s) and 0.0 <= lag_s <= max_lag_s)
        rel_pos = (
            attacker_speed - target_speed_pos
            if alignment_ok and np.isfinite(attacker_speed) and np.isfinite(target_speed_pos)
            else np.nan
        )
        records.append(
            {
                "event_id": e.event_id,
                "target_speed_at_attacker_position_kmh": target_speed_pos if alignment_ok else np.nan,
                "relative_speed_same_position_kmh": rel_pos,
                "spatial_time_gap_s": lag_s if alignment_ok else np.nan,
                "spatial_alignment_ok": alignment_ok,
                "spatial_alignment_lag_raw_s": lag_s,
                "closing_speed_at_start_kmh": closing_kmh,
                "closing_speed_n_samples": n_win,
                "closing_speed_available": bool(np.isfinite(closing_kmh)),
                "closing_speed_window_s": window_s,
            }
        )
    return pd.DataFrame(records)


def build_zone_stats(
    energy_samples: pd.DataFrame, zones: pd.DataFrame, trackaware: pd.DataFrame
) -> pd.DataFrame:
    """Recreate the frozen Approach C zone baselines with the existing function.

    CP10 computed these in memory and persisted only the zone geometry, so they
    are rebuilt here from the same calibration laps and the same samples. The
    run script asserts they reproduce the stored lookahead values.
    """
    cal_keys = trackaware.loc[
        trackaware["calibration_set"] == True, ["Driver", "LapNumber"]  # noqa: E712
    ]
    return zone_baselines(energy_samples, zones, cal_keys)


def distance_to_next_zone_type(distance_m: float, zones: pd.DataFrame, zone_type: str) -> float:
    """Metres to the next zone of this type. NaN if none remains this lap.

    Mirrors the existing lookahead convention: no wrap-around past the line.
    """
    ahead_zones = zones.loc[
        (zones["zone_type"] == zone_type) & (zones["start_m"] > float(distance_m)), "start_m"
    ]
    if ahead_zones.empty:
        return float("nan")
    return float(ahead_zones.min() - float(distance_m))


# ----------------------------------------------------------------------
# Phase 6: proximity episodes
# ----------------------------------------------------------------------


def detect_episodes(joined: pd.DataFrame, params: EventParams) -> pd.DataFrame:
    """Collapse sustained close running by the same pair into single episodes.

    A battle that lasts eight seconds becomes ONE row, not one row per sample.
    """
    work = joined
    eligible = (
        work["DriverAhead"].astype(str).str.strip().ne("")
        & work["gap_m"].notna()
        & ~work["LapNumber"].isin(list(params.exclude_lap_numbers))
    )
    # An "engagement" is continuous proximity within gap_max for one pair,
    # including any sub-gap_min wheel-to-wheel phase. Blocking on the wider
    # band first stops a battle that dips below gap_min and comes back from
    # being emitted twice as two separate events.
    contact = work.loc[eligible & (work["gap_m"] <= params.gap_max_m)].sort_values(
        ["Driver", "DriverAhead", "Date"]
    ).copy()
    if contact.empty:
        return pd.DataFrame()

    pair_changed = (
        contact["Driver"].ne(contact["Driver"].shift())
        | contact["DriverAhead"].ne(contact["DriverAhead"].shift())
    )
    dt = contact["Date"].diff().dt.total_seconds()
    time_broken = dt.isna() | (dt > params.merge_gap_s)
    contact["engagement_index"] = (pair_changed | time_broken).cumsum()

    engagement_end = contact.groupby("engagement_index")["Date"].max()
    jumps = contact.groupby("engagement_index")["gap_m"].apply(
        lambda s: float(s.diff().abs().max()) if len(s) > 1 else 0.0
    )

    # t0 must be a genuine decision point: attacker behind by at least gap_min.
    in_band = contact.loc[contact["gap_m"] >= params.gap_min_m]
    if in_band.empty:
        return pd.DataFrame()

    grouped = in_band.groupby("engagement_index", sort=True)
    episodes = grouped.agg(
        attacker=("Driver", "first"),
        target_number=("DriverAhead", "first"),
        event_start_time=("Date", "first"),
        in_band_end_time=("Date", "last"),
        event_start_lap=("LapNumber", "first"),
        n_samples=("Date", "size"),
        gap_min_m=("gap_m", "min"),
        gap_mean_m=("gap_m", "mean"),
    ).reset_index()
    episodes["event_end_time"] = episodes["engagement_index"].map(engagement_end)
    episodes["max_gap_jump_m"] = episodes["engagement_index"].map(jumps).astype(float)
    episodes["duration_s"] = (
        episodes["event_end_time"] - episodes["event_start_time"]
    ).dt.total_seconds()
    episodes["in_band_duration_s"] = (
        episodes["in_band_end_time"] - episodes["event_start_time"]
    ).dt.total_seconds()

    episodes = episodes.loc[episodes["in_band_duration_s"] >= params.min_persist_s]
    return _dedupe_pair_episodes(episodes, params)


def _dedupe_pair_episodes(episodes: pd.DataFrame, params: EventParams) -> pd.DataFrame:
    """Drop a pair's re-engagement that its own previous horizon still covers.

    If the same attacker/target come back together while the previous event is
    still being scored, any pass there is already attributed to that event.
    Keeping both would count one battle twice.

    The suppression window is the same window the label uses, so under the final
    ``t0`` anchor a pair that re-engages more than horizon_s after the previous
    t0 is a genuinely new opportunity and is kept. Nothing stays suppressed
    indefinitely.
    """
    if episodes.empty:
        return episodes
    ordered = episodes.sort_values(
        ["attacker", "target_number", "event_start_time"]
    ).reset_index(drop=True)
    keep = []
    last_key = None
    last_window_end = None
    for row in ordered.itertuples(index=False):
        key = (row.attacker, row.target_number)
        t0 = pd.Timestamp(row.event_start_time)
        if key != last_key or last_window_end is None or t0 >= last_window_end:
            keep.append(True)
            last_key = key
            last_window_end = horizon_end(t0, row.event_end_time, params)
        else:
            keep.append(False)
    kept = ordered.loc[keep]
    return kept.sort_values("event_start_time").reset_index(drop=True)


# ----------------------------------------------------------------------
# Phase 3-5: per-episode features measured at or before t0
# ----------------------------------------------------------------------


def _asof_row(frame: pd.DataFrame, when: pd.Timestamp, tolerance_ms: int) -> pd.Series | None:
    """Last row at or before `when`, within tolerance. Never looks forward."""
    if frame is None or frame.empty:
        return None
    idx = int(frame["Date"].searchsorted(when, side="right")) - 1
    if idx < 0:
        return None
    row = frame.iloc[idx]
    delta = (when - row["Date"]).total_seconds()
    if delta < 0 or delta > tolerance_ms / 1000.0:
        return None
    return row


def _closing_speed_kmh(
    attacker_ahead: pd.DataFrame,
    target_number: str,
    t0: pd.Timestamp,
    window_s: float,
) -> tuple[float, float]:
    """Closing speed from the gap trend strictly before t0.

    Positive means the attacker was closing. Uses only samples in
    [t0 - window_s, t0] against the same target, so it cannot contain the pass.
    """
    lo = t0 - pd.Timedelta(seconds=window_s)
    pre = attacker_ahead.loc[
        (attacker_ahead["Date"] >= lo)
        & (attacker_ahead["Date"] <= t0)
        & (attacker_ahead["DriverAhead"] == target_number)
    ]
    pre = pre.dropna(subset=["gap_m"])
    if len(pre) < 2:
        return float("nan"), float("nan")
    dt = (pre["Date"].iloc[-1] - pre["Date"].iloc[0]).total_seconds()
    if dt <= 0:
        return float("nan"), float("nan")
    dgap = float(pre["gap_m"].iloc[-1]) - float(pre["gap_m"].iloc[0])
    return float(-dgap / dt * 3.6), float(dt)


def _pair_ordering(
    attacker_samples: pd.DataFrame,
    target_samples: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tolerance_ms: int,
    distance_column: str = "cum_distance_m",
) -> pd.DataFrame:
    """Attacker-minus-target race distance on the attacker's clock."""
    a = attacker_samples.loc[
        (attacker_samples["Date"] >= start) & (attacker_samples["Date"] <= end),
        ["Date", distance_column, "LapNumber"],
    ].sort_values("Date")
    if a.empty or target_samples.empty:
        return pd.DataFrame()
    b = target_samples.loc[:, ["Date", distance_column, "LapNumber"]].sort_values("Date")
    merged = pd.merge_asof(
        a.rename(columns={distance_column: "cum_a", "LapNumber": "lap_a"}),
        b.rename(columns={distance_column: "cum_b", "LapNumber": "lap_b"}),
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta(milliseconds=tolerance_ms),
    )
    merged["cum_diff"] = merged["cum_a"] - merged["cum_b"]
    return merged.dropna(subset=["cum_diff"])


def build_events(
    joined: pd.DataFrame,
    episodes: pd.DataFrame,
    laps_clean: pd.DataFrame,
    zone_stats: pd.DataFrame,
    number_map: dict[str, str],
    track_length_m: float,
    params: EventParams,
    epoch: pd.Timestamp,
    distance_column: str = "cum_distance_m",
    swap_distance_column: str | None = None,
    swap_test: str = "absolute",
    race_id: str | None = None,
    circuit: str | None = None,
    year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature extraction at t0 plus pairwise outcome inside the horizon.

    The two race-distance consumers want different coordinates, because neither
    coordinate is good at both jobs:

    * ``distance_column`` (lapped test) compares two cars in absolute terms at
      one instant. ``cum_distance_m`` is right here: its synthetic lap length is
      identical for every driver, so the lap-count term cancels exactly and only
      the within-lap Distance difference survives. It reproduces the FastF1 gap
      to a median of 0.11 m. Measured-length accumulation is far worse for this,
      because each car's integrated lap differs by tens of metres and that error
      compounds over a race.

    * ``swap_distance_column`` with ``swap_test="displacement"`` measures ground
      each car gained against its OWN position at t0. Differencing a driver
      against itself cancels all cross-driver accumulation, so here the measured
      per-lap coordinate ``race_distance_m`` is correct and the synthetic one is
      not: constant lap length injects ~232 m of phantom distance whenever a car
      crosses the line inside the window.

    ``swap_test="absolute"`` reproduces v1, where the swap fired when
    cum_a - cum_b turned positive.
    """
    if swap_test not in {"absolute", "displacement"}:
        raise ValueError(f"unknown swap_test {swap_test!r}")
    swap_distance_column = swap_distance_column or distance_column
    # Race identity only. Defaults keep the single-race Australia scripts working
    # unchanged; the multi-race pipeline passes each race's own values.
    race_id = config.RACE_ID if race_id is None else race_id
    circuit = config.CIRCUIT_NAME if circuit is None else circuit
    year = config.YEAR if year is None else year
    by_driver = {
        drv: grp.sort_values("Date").reset_index(drop=True)
        for drv, grp in joined.groupby("Driver", sort=False)
    }
    laps = laps_clean.copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    lap_lookup = laps.set_index(["Driver", "LapNumber"])
    global_last_date = joined["Date"].max()

    rows = []
    audits = []
    for i, ep in enumerate(episodes.itertuples(index=False), start=1):
        attacker = str(ep.attacker)
        target_number = str(ep.target_number)
        target = number_map.get(target_number)
        t0 = pd.Timestamp(ep.event_start_time)
        # t0 is the decision point and the label answers "does a pass follow
        # within horizon_s of it". Features remain fixed at t0.
        window_end = horizon_end(t0, ep.event_end_time, params)
        # Evidence is collected past the horizon so a late pass can still be
        # reported as one. Nothing beyond window_end may touch the label.
        audit_end = max(window_end, t0 + pd.Timedelta(seconds=params.audit_horizon_s))

        a_frame = by_driver.get(attacker)
        b_frame = by_driver.get(target) if target else None
        a_row = _asof_row(a_frame, t0, params.asof_tolerance_ms)

        reasons: list[str] = []
        event_type = None

        if target is None:
            event_type = "INVALID"
            reasons.append("target car number not in driver map")
        if a_row is None:
            event_type = "INVALID"
            reasons.append("no attacker sample at t0")
        if b_frame is None or (b_frame is not None and b_frame.empty):
            event_type = "INVALID"
            reasons.append("no target sample stream")

        b_row = _asof_row(b_frame, t0, params.asof_tolerance_ms) if b_frame is not None else None
        if b_row is None and event_type is None:
            event_type = "INVALID"
            reasons.append("no causal target sample at t0")

        attacker_speed = float(a_row["Speed"]) if a_row is not None else float("nan")
        target_speed = float(b_row["Speed"]) if b_row is not None else float("nan")
        distance_at_start = float(a_row["Distance"]) if a_row is not None else float("nan")
        gap_at_start = float(a_row["gap_m"]) if a_row is not None else float("nan")

        closing_kmh, closing_dt = _closing_speed_kmh(
            a_frame, target_number, t0, params.closing_window_s
        )

        look = {"LookaheadDeployProxy": np.nan, "LookaheadHarvestProxy": np.nan, "LookaheadCoveredM": np.nan}
        if np.isfinite(distance_at_start):
            look = lookahead_from_distance(distance_at_start, zone_stats)

        # ---- race state at t0 -------------------------------------------------
        track_status = str(a_row["TrackStatus"]) if a_row is not None else ""
        freeze_flag = bool(a_row["freeze_flag"]) if a_row is not None else False
        target_freeze = bool(b_row["freeze_flag"]) if b_row is not None else False
        green_at_start = (track_status == "1") and not freeze_flag and not target_freeze

        # ---- pairwise ordering and swap evidence ------------------------------
        # Each signal is first located anywhere in [t0, audit_end], then censored
        # at window_end. Only the censored copies reach the classifier; the raw
        # times exist so a pass that lands past the horizon is still visible.
        cum_diff_at_start = float("nan")
        swap_time_distance_any = pd.NaT
        swap_time_ahead_any = pd.NaT
        lapped_at_start = False
        if a_row is not None and b_row is not None:
            cum_diff_at_start = float(a_row[distance_column]) - float(b_row[distance_column])
            lapped_at_start = abs(cum_diff_at_start) > params.lapped_fraction * track_length_m

            order = _pair_ordering(
                a_frame, b_frame, t0, audit_end, params.asof_tolerance_ms, swap_distance_column
            )
            if not order.empty:
                after = order.loc[order["Date"] > t0]
                if swap_test == "absolute":
                    crossed = after.loc[after["cum_diff"] > 0]
                else:
                    # Ground each car has covered since t0. Differencing a driver
                    # against its own t0 position cancels every cross-driver
                    # accumulation error, so the swap fires once the attacker has
                    # taken back more than the independently measured gap.
                    gained = (after["cum_a"] - float(a_row[swap_distance_column])) - (
                        after["cum_b"] - float(b_row[swap_distance_column])
                    )
                    threshold = gap_at_start if np.isfinite(gap_at_start) else 0.0
                    crossed = after.loc[gained > threshold]
                if not crossed.empty:
                    swap_time_distance_any = pd.Timestamp(crossed["Date"].iloc[0])

            # The target's own ahead stream naming the attacker is the direct
            # pairwise statement "A is now ahead of B".
            a_num = str(int(float(a_row["DriverNumber"]))) if pd.notna(a_row.get("DriverNumber")) else None
            if a_num is not None:
                b_sees_a = b_frame.loc[
                    (b_frame["Date"] > t0)
                    & (b_frame["Date"] <= audit_end)
                    & (b_frame["DriverAhead"] == a_num)
                ]
                if not b_sees_a.empty:
                    swap_time_ahead_any = pd.Timestamp(b_sees_a["Date"].iloc[0])

        def _censor(when):
            return when if pd.notna(when) and when <= window_end else pd.NaT

        swap_time_distance = _censor(swap_time_distance_any)
        swap_time_ahead = _censor(swap_time_ahead_any)
        has_distance_swap = pd.notna(swap_time_distance)
        has_ahead_swap = pd.notna(swap_time_ahead)
        swap_time = min(
            [t for t in [swap_time_distance, swap_time_ahead] if pd.notna(t)],
            default=pd.NaT,
        )

        # A pass is corroborated only once BOTH signals have fired, so the
        # corroboration instant is the later of the two.
        corroborated_time = (
            max(swap_time_distance_any, swap_time_ahead_any)
            if pd.notna(swap_time_distance_any) and pd.notna(swap_time_ahead_any)
            else pd.NaT
        )
        actual_swap_delay_s = (
            float((corroborated_time - t0).total_seconds()) if pd.notna(corroborated_time) else np.nan
        )
        pass_after_horizon = bool(pd.notna(corroborated_time) and corroborated_time > window_end)

        # ---- pit involvement ---------------------------------------------------
        outcome_end = swap_time if pd.notna(swap_time) else window_end
        pit_flags = _pit_involvement(
            attacker, target, a_frame, b_frame, lap_lookup, t0, outcome_end, window_end, epoch
        )

        # ---- retirement --------------------------------------------------------
        target_last_date = b_frame["Date"].max() if b_frame is not None and not b_frame.empty else pd.NaT
        target_retires = bool(
            pd.notna(target_last_date)
            and target_last_date < outcome_end
            and target_last_date < global_last_date - pd.Timedelta(seconds=60)
        )

        gap_jump_limit = params.gap_jump_fraction * track_length_m
        big_gap_jump = float(ep.max_gap_jump_m) > gap_jump_limit

        # ---- classification (precedence order matters) -------------------------
        if event_type is None:
            if pit_flags["either_in_box_lap_at_start"]:
                # Checked before the green test: a box lap also sets freeze_flag,
                # and "in the pits" is a pit event, not an unclassified invalid one.
                event_type = "PIT_RELATED"
                reasons.append("attacker or target on a box lap at t0")
            elif not green_at_start:
                event_type = "INVALID"
                reasons.append(f"race state not green at t0 (TrackStatus={track_status}, freeze={freeze_flag or target_freeze})")
            elif big_gap_jump:
                event_type = "INVALID"
                reasons.append(f"gap discontinuity {ep.max_gap_jump_m:.0f} m inside episode")
            elif not np.isfinite(gap_at_start) or not np.isfinite(attacker_speed):
                event_type = "INVALID"
                reasons.append("missing gap or speed at t0")
            elif lapped_at_start:
                event_type = "LAPPED_UNLAPPED"
                reasons.append(f"pair separated by {cum_diff_at_start:.0f} m of race distance at t0")
            elif pd.notna(swap_time) and pit_flags["target_pit_before_outcome"]:
                event_type = "PIT_RELATED"
                reasons.append("target pitted at or before the order change")
            elif pd.isna(swap_time) and pit_flags["either_pit_in_window"]:
                event_type = "PIT_RELATED"
                reasons.append("a pit stop occurred inside the outcome window")
            elif target_retires:
                event_type = "RETIREMENT_RELATED"
                reasons.append("target telemetry ends inside the outcome window")
            elif has_distance_swap and has_ahead_swap:
                event_type = "ON_TRACK_PASS"
                reasons.append("race-distance ordering and target DriverAhead both show the swap")
            elif has_distance_swap or has_ahead_swap:
                event_type = "UNCERTAIN"
                reasons.append(
                    "only one of the two order-change signals fired "
                    f"(distance={has_distance_swap}, driver_ahead={has_ahead_swap})"
                )
            else:
                event_type = "CLOSE_INTERACTION_NO_PASS"
                if pass_after_horizon:
                    reasons.append(
                        "valid close battle, no pairwise order change inside the horizon; "
                        f"the pair did swap {actual_swap_delay_s:.1f} s after t0, outside it"
                    )
                else:
                    reasons.append("valid close battle, no pairwise order change inside the horizon")

        overtake_success = (
            1 if event_type == "ON_TRACK_PASS" else (0 if event_type == "CLOSE_INTERACTION_NO_PASS" else np.nan)
        )

        lap_key = (attacker, int(ep.event_start_lap))
        lap_row = lap_lookup.loc[lap_key] if lap_key in lap_lookup.index else None
        target_position = float(b_row["Position"]) if b_row is not None and pd.notna(b_row.get("Position")) else np.nan

        gap_s_est = (
            gap_at_start / (attacker_speed / 3.6)
            if np.isfinite(gap_at_start) and np.isfinite(attacker_speed) and attacker_speed > 1.0
            else np.nan
        )

        rows.append(
            {
                "event_id": f"{race_id}_{i:05d}",
                "race_id": race_id,
                "year": year,
                "circuit": circuit,
                "attacker": attacker,
                "target": target if target else "",
                "target_number": target_number,
                "event_start_time": t0,
                "event_end_time": pd.Timestamp(ep.event_end_time),
                "event_duration_s": float(ep.duration_s),
                "in_band_duration_s": float(ep.in_band_duration_s),
                "outcome_window_end": window_end,
                "outcome_window_s": float((window_end - t0).total_seconds()),
                "event_start_lap": int(ep.event_start_lap),
                "n_samples_in_episode": int(ep.n_samples),
                "gap_at_start_m": gap_at_start,
                "gap_at_start_s_est": gap_s_est,
                "gap_min_in_episode_m": float(ep.gap_min_m),
                "attacker_speed_at_start_kmh": attacker_speed,
                "target_speed_at_start_kmh": target_speed,
                "relative_speed_at_start_kmh": attacker_speed - target_speed
                if np.isfinite(attacker_speed) and np.isfinite(target_speed)
                else np.nan,
                "closing_speed_at_start_kmh": closing_kmh,
                "closing_speed_window_s": closing_dt,
                # Missingness is label-correlated (a fresh engagement has no
                # pre-t0 history with this target), so the ML phase must model
                # the flag rather than impute the value away.
                "closing_speed_available": bool(np.isfinite(closing_kmh)),
                "attacker_position": float(lap_row["Position"]) if lap_row is not None and pd.notna(lap_row.get("Position")) else np.nan,
                "target_position": target_position,
                "track_distance_at_start_m": distance_at_start,
                "zone_id": str(a_row["zone_id"]) if a_row is not None else "",
                "distance_to_next_corner_m": distance_to_next_zone_type(distance_at_start, zone_stats, "corner")
                if np.isfinite(distance_at_start)
                else np.nan,
                "distance_to_next_straight_m": distance_to_next_zone_type(distance_at_start, zone_stats, "straight")
                if np.isfinite(distance_at_start)
                else np.nan,
                "compound": str(lap_row["Compound"]) if lap_row is not None else "",
                "tyre_age": float(lap_row["TyreLife"]) if lap_row is not None and pd.notna(lap_row.get("TyreLife")) else np.nan,
                "stint": float(lap_row["Stint"]) if lap_row is not None and pd.notna(lap_row.get("Stint")) else np.nan,
                "EstimatedEnergyIndex": float(a_row["EstimatedEnergyIndex"]) if a_row is not None else np.nan,
                "LookaheadDeployProxy": look["LookaheadDeployProxy"],
                "LookaheadHarvestProxy": look["LookaheadHarvestProxy"],
                "LookaheadCoveredM": look["LookaheadCoveredM"],
                "lookahead_truncated": bool(
                    np.isfinite(look["LookaheadCoveredM"]) and look["LookaheadCoveredM"] < config.LOOKAHEAD_M - 1e-6
                ),
                "TrackStatus": track_status,
                "freeze_flag": freeze_flag,
                "event_type": event_type,
                "overtake_success": overtake_success,
                # Audit only. Measured over [t0, t0 + audit_horizon_s] so a late
                # swap stays visible; it never feeds event_type or the label.
                "actual_swap_delay_s": actual_swap_delay_s,
                "pass_after_horizon": pass_after_horizon,
            }
        )

        audits.append(
            {
                "event_id": f"{race_id}_{i:05d}",
                "race_id": race_id,
                "attacker": attacker,
                "target": target if target else "",
                "event_start_time": t0,
                "event_start_lap": int(ep.event_start_lap),
                "event_type": event_type,
                "reason": "; ".join(reasons),
                "gap_at_start_m": gap_at_start,
                "duration_s": float(ep.duration_s),
                "cum_distance_diff_at_start_m": cum_diff_at_start,
                "lapped_at_start": lapped_at_start,
                "max_gap_jump_m": float(ep.max_gap_jump_m),
                "swap_by_distance": has_distance_swap,
                "swap_by_driver_ahead": has_ahead_swap,
                "swap_time": swap_time,
                "swap_delay_s": float((swap_time - t0).total_seconds()) if pd.notna(swap_time) else np.nan,
                "swap_distance_delay_any_s": float(
                    (swap_time_distance_any - t0).total_seconds()
                ) if pd.notna(swap_time_distance_any) else np.nan,
                "swap_ahead_delay_any_s": float(
                    (swap_time_ahead_any - t0).total_seconds()
                ) if pd.notna(swap_time_ahead_any) else np.nan,
                "actual_swap_delay_s": actual_swap_delay_s,
                "pass_after_horizon": pass_after_horizon,
                "target_speed_at_start_kmh": target_speed,
                "closing_speed_at_start_kmh": closing_kmh,
                "green_at_start": green_at_start,
                "target_retires_in_window": target_retires,
                "horizon_end": window_end,
                "horizon_s_effective": float((window_end - t0).total_seconds()),
                "audit_horizon_s": float(params.audit_horizon_s),
                **pit_flags,
            }
        )

    events = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    return events, audit


def _pit_involvement(
    attacker: str,
    target: str | None,
    a_frame: pd.DataFrame | None,
    b_frame: pd.DataFrame | None,
    lap_lookup: pd.DataFrame,
    t0: pd.Timestamp,
    outcome_end: pd.Timestamp,
    window_end: pd.Timestamp,
    epoch: pd.Timestamp,
) -> dict:
    """Pit exposure for both cars, scoped so a later stop cannot void a real pass."""

    def laps_in(frame, start, end):
        if frame is None or frame.empty:
            return []
        sl = frame.loc[(frame["Date"] >= start) & (frame["Date"] <= end)]
        return sorted({int(x) for x in sl["LapNumber"].unique()})

    def box_lap(driver, lap_number):
        key = (driver, lap_number)
        if key not in lap_lookup.index:
            return False
        row = lap_lookup.loc[key]
        return bool(pd.notna(row.get("PitInTime")) or pd.notna(row.get("PitOutTime")))

    def pit_in_date(driver, lap_number):
        key = (driver, lap_number)
        if key not in lap_lookup.index:
            return pd.NaT
        raw = lap_lookup.loc[key].get("PitInTime")
        if pd.isna(raw):
            return pd.NaT
        return epoch + pd.to_timedelta(raw)

    a_start_laps = laps_in(a_frame, t0, t0)
    b_start_laps = laps_in(b_frame, t0, t0)
    either_box_at_start = any(box_lap(attacker, ln) for ln in a_start_laps) or (
        target is not None and any(box_lap(target, ln) for ln in b_start_laps)
    )

    a_window_laps = laps_in(a_frame, t0, window_end)
    b_window_laps = laps_in(b_frame, t0, window_end)
    either_pit_in_window = any(box_lap(attacker, ln) for ln in a_window_laps) or (
        target is not None and any(box_lap(target, ln) for ln in b_window_laps)
    )

    target_pit_before_outcome = False
    if target is not None:
        for ln in laps_in(b_frame, t0, outcome_end):
            pit_dt = pit_in_date(target, ln)
            if pd.notna(pit_dt) and t0 <= pit_dt <= outcome_end:
                target_pit_before_outcome = True
                break

    return {
        "either_in_box_lap_at_start": bool(either_box_at_start),
        "either_pit_in_window": bool(either_pit_in_window),
        "target_pit_before_outcome": bool(target_pit_before_outcome),
    }
