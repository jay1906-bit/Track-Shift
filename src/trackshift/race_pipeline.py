"""Run the frozen Australia pipeline against an arbitrary 2026 race.

This module is an ORCHESTRATOR. It contains no new modelling. Every step calls
the existing frozen implementation:

* lap cleaning        - the rule used to build the Australia clean spine
                        (drop Deleted == True, drop missing LapTime)
* telemetry + proxies - pipeline.process_all_laps, proxies.winsorize_session_proxies
* alpha / beta        - calibrate.calibrate_alpha_beta, per race, plus the same
                        floor-clip reduction loop as scripts/run_energy_system.py
* energy walk         - energy.walk_all_drivers
* zones + lookahead   - zones.build_zone_table / zone_baselines
* driver ahead        - strategy_features.extract_all_driver_ahead
* event detection     - events.detect_episodes / events.build_events with the
                        frozen v4 parameters and the t0-anchored horizon

alpha and beta are calibrated per race because the repository's established
method derives them from that session's own calibration laps. Reusing
Australia's constants on another circuit would not be comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .calibrate import apply_scale, calibrate_alpha_beta, calibration_mask
from .energy import clip_lap_fraction, end_of_lap_energy, walk_all_drivers
from .events import (
    EventParams,
    add_cumulative_distance,
    add_race_distance,
    add_track_position_speed_features,
    build_driver_number_map,
    build_events,
    build_lap_lengths,
    detect_episodes,
    join_attacker_samples,
)
from .pipeline import integrate_lap, process_all_laps
from .proxies import winsorize_session_proxies
from .strategy_features import (
    _normalize_driver_ahead,
    build_derived_laps,
    extract_all_driver_ahead,
)
from .zones import assign_zone, build_zone_table, zone_baselines

DETECTOR_VERSION = "v4_t0_anchored_30s_horizon"


@dataclass(frozen=True)
class RaceSpec:
    """One race session to process. Race sessions only, never sprint or quali."""

    year: int
    event_name: str
    slug: str
    session_code: str = "R"

    @property
    def race_id(self) -> str:
        return f"{self.slug}_{self.year}_{self.session_code}"


@dataclass
class RaceArtifacts:
    spec: RaceSpec
    events: pd.DataFrame
    audit: pd.DataFrame
    laps_clean: pd.DataFrame
    zones: pd.DataFrame
    energy_config: dict
    quality: dict
    error: str | None = None
    stage: str | None = None
    timings: dict = field(default_factory=dict)


def clean_laps(laps_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The Australia cleaning rule, applied verbatim.

    Only two rows are removed: laps the stewards deleted, and rows with no
    LapTime. Nothing else is filtered here; race-state handling happens later.
    """
    deleted = laps_raw["Deleted"] == True  # noqa: E712
    missing_laptime = laps_raw["LapTime"].isna()
    out = laps_raw.loc[~deleted & ~missing_laptime].copy()
    stats = {
        "raw_rows": int(len(laps_raw)),
        "deleted_removed": int(deleted.sum()),
        "missing_laptime_removed": int(missing_laptime.sum()),
        "clean_rows": int(len(out)),
    }
    return out, stats


def _recompute_lap_integrals(lap_table: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Mirror of the helper in scripts/run_energy_system.py.

    Duplicated rather than imported so the frozen Australia script is not
    touched. Keep the two in step if either ever changes.
    """
    out = lap_table.copy()
    updated = []
    for (drv, ln), grp in samples.groupby(["Driver", "LapNumber"]):
        thin = bool(out.loc[(out["Driver"] == drv) & (out["LapNumber"] == ln), "thin_lap"].iloc[0])
        integ = integrate_lap(grp, {"thin_lap": thin})
        integ["Driver"] = drv
        integ["LapNumber"] = ln
        updated.append(integ)
    upd = pd.DataFrame(updated)
    drop_cols = [c for c in ["D_lap", "H_lap", "max_dt_used", "n_dt_gaps", "mean_throttle"] if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out.merge(upd, on=["Driver", "LapNumber"], how="left")


def calibrate_energy(lap_table: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-race alpha/beta plus the frozen floor-clip reduction loop."""
    cfg = calibrate_alpha_beta(lap_table)
    samples_e = walk_all_drivers(samples, cfg["alpha"], cfg["beta"])
    end_e = end_of_lap_energy(samples_e)
    floor_frac = float((end_e["n_clip_low"] > 0).mean())
    iters = 0
    while floor_frac > config.CLIP_LAP_FRACTION_LIMIT and iters < config.CLIP_RATE_MAX_ITERS:
        cfg = apply_scale(cfg, config.CLIP_RATE_REDUCE_FACTOR)
        iters += 1
        samples_e = walk_all_drivers(samples, cfg["alpha"], cfg["beta"])
        end_e = end_of_lap_energy(samples_e)
        floor_frac = float((end_e["n_clip_low"] > 0).mean())
    cfg["clip_lap_fraction"] = clip_lap_fraction(end_e)
    cfg["n_scale_reductions"] = iters
    cfg["floor_touch_lap_fraction"] = floor_frac
    return samples_e, cfg


def add_portable_track_features(
    events: pd.DataFrame, zones: pd.DataFrame, meta: dict
) -> pd.DataFrame:
    """Circuit-portable replacements for the raw zone_id, plus circuit geometry.

    zone_id is circuit-specific: brake_T11 exists at Albert Park and nowhere
    else, so a model trained on it cannot be evaluated on an unseen circuit.
    zone_type has three levels everywhere, and track_position_frac is the same
    quantity on every track.

    The circuit geometry columns are exogenous. No quantity derived from what
    happened in the race is attached here.
    """
    out = events.copy()
    zone_type = zones.set_index("zone_id")["zone_type"]
    out["zone_type"] = out["zone_id"].map(zone_type).fillna("unassigned")
    out["track_position_frac"] = pd.to_numeric(
        out["track_distance_at_start_m"], errors="coerce"
    ) / float(meta["circuit_length_m"])
    for col in ("circuit_length_m", "corner_count", "longest_straight_m",
                "n_long_straights_over_500m"):
        out[col] = meta[col]
    return out


def add_cluster_keys(events: pd.DataFrame) -> pd.DataFrame:
    """Grouping keys for later race-grouped evaluation.

    pair_id is unordered so that A->B and B->A share it: they describe the same
    two cars. reciprocal_event_id links the specific opposite-direction event
    that overlaps in time, which is the tighter correlation.
    """
    out = events.copy()
    out["pair_id"] = [
        f"{r}|" + "|".join(sorted([str(a), str(b)]))
        for r, a, b in zip(out["race_id"], out["attacker"], out["target"])
    ]
    out["cluster_id"] = (
        out["race_id"].astype(str) + "|L" + out["event_start_lap"].astype(int).astype(str)
        + "|" + out["pair_id"].astype(str)
    )

    # A reciprocal is the same pair running in the opposite direction with
    # overlapping 30 s windows.
    reciprocal = pd.Series([""] * len(out), index=out.index, dtype=object)
    t0 = pd.to_datetime(out["event_start_time"])
    horizon = pd.to_datetime(out["horizon_end_time"])
    for _, idx in out.groupby("pair_id").groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        for i in idx:
            for j in idx:
                if i == j or out.at[i, "attacker"] == out.at[j, "attacker"]:
                    continue
                if t0[j] <= horizon[i] and t0[i] <= horizon[j]:
                    reciprocal[i] = out.at[j, "event_id"]
                    break
    out["reciprocal_event_id"] = reciprocal
    return out


def measure_circuit_length(samples: pd.DataFrame) -> tuple[float, dict]:
    """Circuit length from measured lap Distance, robust to a reset failure.

    Australia originally used ``nanmax(Distance)``. That equals the true Albert
    Park length because every lap reset. On later races a lap whose Distance
    did not reset (China LAW lap 13: 8945 m against a 5408 m median) stretched
    the zone table, the lapped threshold and ``track_position_frac``. That is a
    telemetry defect, not a detector-threshold change.

    Method: drop per-lap maxima above ``Q75 + 3*IQR`` (Tukey fence with a 2%
    of-median IQR floor so a tight cluster cannot over-reject), then take the
    remaining maximum. Albert Park (max/median = 1.04) is unchanged.
    """
    per_lap = samples.groupby(["Driver", "LapNumber"])["Distance"].max()
    q25 = float(per_lap.quantile(0.25))
    median = float(per_lap.median())
    q75 = float(per_lap.quantile(0.75))
    iqr = max(q75 - q25, 0.02 * median if median > 0 else 0.0)
    cap = q75 + 3.0 * iqr
    inliers = per_lap[per_lap <= cap]
    if inliers.empty:
        length = float(per_lap.max())
    else:
        length = float(inliers.max())
    outliers = per_lap[per_lap > cap]
    return length, {
        "method": "max per-lap Distance among laps <= Q75 + 3*IQR",
        "n_laps": int(len(per_lap)),
        "q25_lap_length_m": q25,
        "median_lap_length_m": median,
        "q75_lap_length_m": q75,
        "iqr_m": float(iqr),
        "outlier_cap_m": float(cap),
        "raw_max_lap_length_m": float(per_lap.max()),
        "n_outlier_laps": int(len(outliers)),
        "outlier_laps": [
            {"driver": str(drv), "lap": int(lap), "length_m": float(val)}
            for (drv, lap), val in outliers.items()
        ][:20],
        "circuit_length_m": length,
        "raw_max_over_median": (
            float(per_lap.max() / median) if median > 0 else float("nan")
        ),
        "used_over_median": float(length / median) if median > 0 else float("nan"),
    }


def circuit_metadata(zones: pd.DataFrame, circuit_length_m: float) -> dict:
    """Exogenous circuit geometry. Never derived from the race outcome."""
    straights = zones.loc[zones["zone_type"] == "straight", "length_m"]
    return {
        "circuit_length_m": float(circuit_length_m),
        "corner_count": int(zones.loc[zones["zone_type"] == "corner"].shape[0]),
        "n_zones": int(len(zones)),
        "longest_straight_m": float(straights.max()) if len(straights) else float("nan"),
        "n_long_straights_over_500m": int((straights > 500.0).sum()) if len(straights) else 0,
        "mean_straight_m": float(straights.mean()) if len(straights) else float("nan"),
    }


def process_race(
    spec: RaceSpec,
    params: EventParams | None = None,
    progress_every: int = 200,
    verbose: bool = True,
) -> RaceArtifacts:
    """Full frozen pipeline for one race. Raises nothing; failures are reported."""
    import fastf1

    params = params or EventParams()
    timings: dict = {}
    quality: dict = {"race_id": spec.race_id, "event_name": spec.event_name}
    stage = "init"

    def log(msg: str) -> None:
        if verbose:
            print(f"    [{spec.slug}] {msg}", flush=True)

    def blank(err: str) -> RaceArtifacts:
        return RaceArtifacts(
            spec=spec,
            events=pd.DataFrame(),
            audit=pd.DataFrame(),
            laps_clean=pd.DataFrame(),
            zones=pd.DataFrame(),
            energy_config={},
            quality=quality,
            error=err,
            stage=stage,
            timings=timings,
        )

    try:
        stage = "load_session"
        t = time.time()
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(config.CACHE_DIR))
        session = fastf1.get_session(spec.year, spec.event_name, spec.session_code)
        session.load()
        timings["load_session_s"] = round(time.time() - t, 1)
        log(f"session loaded in {timings['load_session_s']}s")

        stage = "clean_laps"
        laps_raw = pd.DataFrame(session.laps)
        laps, clean_stats = clean_laps(laps_raw)
        laps["LapNumber"] = laps["LapNumber"].astype(int)
        quality["lap_cleaning"] = clean_stats
        quality["n_drivers"] = int(laps["Driver"].nunique())
        log(f"clean laps {clean_stats['clean_rows']} of {clean_stats['raw_rows']}, "
            f"{quality['n_drivers']} drivers")
        if clean_stats["clean_rows"] < 100:
            return blank(f"only {clean_stats['clean_rows']} clean laps; race unusable")

        stage = "telemetry_proxies"
        t = time.time()
        lap_table, samples = process_all_laps(session, laps, progress_every=progress_every)
        if samples.empty:
            return blank("no telemetry samples extracted")
        samples, win_report = winsorize_session_proxies(samples)
        lap_table = _recompute_lap_integrals(lap_table, samples)
        lap_table["calibration_set"] = calibration_mask(lap_table)
        timings["telemetry_s"] = round(time.time() - t, 1)
        quality["winsorize"] = win_report
        quality["n_samples"] = int(len(samples))
        quality["n_lap_rows"] = int(len(lap_table))
        quality["n_thin_laps"] = int(lap_table["thin_lap"].sum())
        quality["n_lap_errors"] = int(lap_table["error"].notna().sum())
        log(f"telemetry {len(samples)} samples in {timings['telemetry_s']}s")

        stage = "energy_calibration"
        t = time.time()
        samples_e, energy_cfg = calibrate_energy(lap_table, samples)
        end_e = end_of_lap_energy(samples_e)
        samples = samples.merge(
            samples_e.loc[:, ["Driver", "LapNumber", "Date", "EstimatedEnergyIndex"]],
            on=["Driver", "LapNumber", "Date"],
            how="left",
        )
        energy_laps = lap_table.merge(end_e, on=["Driver", "LapNumber"], how="left")
        timings["energy_s"] = round(time.time() - t, 1)
        quality["energy"] = {
            "alpha": energy_cfg["alpha"],
            "beta": energy_cfg["beta"],
            "flow_scale_T": energy_cfg.get("flow_scale_T"),
            "n_calibration_laps": energy_cfg["calibration_set"]["n_laps"],
            "n_scale_reductions": energy_cfg["n_scale_reductions"],
            "floor_touch_lap_fraction": energy_cfg["floor_touch_lap_fraction"],
            "clip_lap_fraction": energy_cfg["clip_lap_fraction"],
            "sample_energy_missing": int(samples["EstimatedEnergyIndex"].isna().sum()),
            "sample_energy_min": float(samples["EstimatedEnergyIndex"].min()),
            "sample_energy_max": float(samples["EstimatedEnergyIndex"].max()),
        }
        log(f"alpha={energy_cfg['alpha']:.4g} beta={energy_cfg['beta']:.4g} "
            f"({timings['energy_s']}s)")

        stage = "zones"
        circuit_length_m, length_stats = measure_circuit_length(samples)
        quality["circuit_length_measurement"] = length_stats
        if length_stats["n_outlier_laps"]:
            log(
                f"WARNING: {length_stats['n_outlier_laps']} lap(s) had Distance "
                f"above the IQR fence ({length_stats['outlier_cap_m']:.0f} m); "
                f"raw max {length_stats['raw_max_lap_length_m']:.0f} m ignored. "
                f"Using {circuit_length_m:.1f} m "
                f"(median {length_stats['median_lap_length_m']:.0f} m)"
            )
        zones = build_zone_table(session, circuit_length_m)
        corner_src = zones.attrs.get("corner_distance_source", "unknown")
        quality["circuit_info_distance_source"] = corner_src
        if corner_src != "fastest_lap":
            log(f"WARNING: CircuitInfo Distance from {corner_src}")
        samples["zone_id"] = assign_zone(samples["Distance"].to_numpy(dtype=float), zones)
        cal_keys = energy_laps.loc[energy_laps["calibration_set"] == True, ["Driver", "LapNumber"]]  # noqa: E712
        zone_stats = zone_baselines(samples, zones, cal_keys)
        quality["circuit"] = circuit_metadata(zones, circuit_length_m)
        quality["n_unassigned_zone_samples"] = int((samples["zone_id"] == "unassigned").sum())

        stage = "driver_ahead"
        t = time.time()
        ahead = extract_all_driver_ahead(session, laps, progress_every=progress_every)
        if ahead.empty:
            return blank("driver-ahead extract is empty")
        ahead["DriverAhead"] = ahead["DriverAhead"].map(_normalize_driver_ahead)
        timings["driver_ahead_s"] = round(time.time() - t, 1)
        quality["n_ahead_samples"] = int(len(ahead))
        log(f"driver-ahead {len(ahead)} samples in {timings['driver_ahead_s']}s")

        stage = "join"
        joined, join_stats = join_attacker_samples(ahead, samples, laps, params)
        joined = add_cumulative_distance(joined, circuit_length_m)
        joined = add_race_distance(joined, build_lap_lengths(samples))
        quality["join"] = join_stats
        race_dist = joined.sort_values(["Driver", "Date"])
        quality["race_distance_negative_steps"] = int(
            (race_dist.groupby("Driver")["race_distance_m"].diff() < 0).sum()
        )
        lap_lengths = samples.groupby(["Driver", "LapNumber"])["Distance"].max()
        quality["lap_length_m"] = {
            "min": float(lap_lengths.min()),
            "median": float(lap_lengths.median()),
            "max": float(lap_lengths.max()),
        }

        stage = "events"
        number_map = build_driver_number_map(laps)
        episodes = detect_episodes(joined, params)
        if episodes.empty:
            return blank("no proximity episodes detected")
        events, audit = build_events(
            joined, episodes, laps, zone_stats, number_map, circuit_length_m, params,
            epoch=_session_epoch(samples),
            distance_column="cum_distance_m",
            swap_distance_column="race_distance_m",
            swap_test="displacement",
            race_id=spec.race_id,
            circuit=spec.event_name,
            year=spec.year,
        )
        quality["n_episodes"] = int(len(episodes))
        quality["target_mapping"] = {
            "n_car_numbers": len(number_map),
            "n_events_with_unmapped_target": int((events["target"] == "").sum()),
        }

        stage = "features"
        by_driver = {
            drv: grp.sort_values("Date").reset_index(drop=True)
            for drv, grp in joined.groupby("Driver", sort=False)
        }
        derived = build_derived_laps(laps)
        events = events.merge(
            derived.loc[:, ["Driver", "LapNumber", "PositionChange"]].rename(
                columns={
                    "Driver": "attacker",
                    "LapNumber": "event_start_lap",
                    "PositionChange": "attacker_PositionChange_audit",
                }
            ),
            on=["attacker", "event_start_lap"],
            how="left",
        )
        speed = add_track_position_speed_features(
            events, by_driver,
            config.EVENT_SPATIAL_LOOKBACK_S,
            config.EVENT_CLOSING_WINDOW_V2_S,
            config.EVENT_MAX_ALIGNMENT_LAG_S,
        )
        events = events.rename(
            columns={
                "target_speed_at_start_kmh": "target_speed_same_time_kmh_audit",
                "relative_speed_at_start_kmh": "relative_speed_same_time_kmh_audit",
                "closing_speed_at_start_kmh": "closing_speed_v1_kmh_audit",
                "closing_speed_window_s": "closing_speed_v1_window_s_audit",
                "closing_speed_available": "closing_speed_v1_available_audit",
                "gap_at_start_s_est": "gap_at_start_s_crude_audit",
                "outcome_window_end": "horizon_end_time",
                "outcome_window_s": "horizon_s_effective",
            }
        ).merge(speed, on="event_id", how="left", validate="one_to_one")

        events = add_portable_track_features(events, zones, quality["circuit"])
        events = add_cluster_keys(events)
        events["window_crosses_start_finish"] = _crosses_line(events, by_driver)
        events["detector_version"] = DETECTOR_VERSION
        events["target_number"] = events["target_number"].astype(str)

        stage = "quality"
        quality.update(_event_quality(events, audit, params))
        quality["frozen_parameters"] = {
            "gap_max_m": params.gap_max_m,
            "gap_min_m": params.gap_min_m,
            "min_persist_s": params.min_persist_s,
            "merge_gap_s": params.merge_gap_s,
            "horizon_s": params.horizon_s,
            "horizon_anchor": params.horizon_anchor,
        }
        quality["timings_s"] = timings
        log(f"events {len(events)}  passes {int((events['event_type'] == 'ON_TRACK_PASS').sum())}")

        return RaceArtifacts(
            spec=spec,
            events=events,
            audit=audit,
            laps_clean=laps,
            zones=zones,
            energy_config=energy_cfg,
            quality=quality,
            timings=timings,
        )
    except Exception as exc:  # noqa: BLE001 - failures are data, not crashes
        import traceback

        quality["traceback"] = traceback.format_exc()[-2000:]
        return blank(f"{type(exc).__name__}: {exc}")


def _session_epoch(samples: pd.DataFrame) -> pd.Timestamp:
    from .events import session_epoch

    return session_epoch(samples)


def _crosses_line(events: pd.DataFrame, by_driver: dict) -> list[bool]:
    out = []
    for r in events.itertuples(index=False):
        t0 = pd.Timestamp(r.event_start_time)
        end = pd.Timestamp(r.horizon_end_time)
        hit = False
        for drv in (r.attacker, r.target):
            frame = by_driver.get(drv) if isinstance(drv, str) and drv else None
            if frame is None:
                continue
            win = frame.loc[(frame["Date"] >= t0) & (frame["Date"] <= end)]
            if win["LapNumber"].nunique() > 1:
                hit = True
                break
        out.append(hit)
    return out


def _event_quality(events: pd.DataFrame, audit: pd.DataFrame, params: EventParams) -> dict:
    from .events import EVENT_TYPES

    scoreable = events.loc[events["event_type"].isin(["ON_TRACK_PASS", "CLOSE_INTERACTION_NO_PASS"])]
    n_pass = int((events["event_type"] == "ON_TRACK_PASS").sum())
    n_no = int((events["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum())
    horizon_len = (
        pd.to_datetime(events["horizon_end_time"]) - pd.to_datetime(events["event_start_time"])
    ).dt.total_seconds()

    spaced = events.sort_values(["attacker", "target_number", "event_start_time"])
    same_pair = (
        spaced["attacker"].eq(spaced["attacker"].shift())
        & spaced["target_number"].eq(spaced["target_number"].shift())
    )
    sep = (spaced["event_start_time"] - spaced["event_start_time"].shift()).dt.total_seconds().where(same_pair)

    closing_missing = ~events["closing_speed_available"].astype(bool)
    by_class = {}
    for label, name in ((1.0, "positive"), (0.0, "negative")):
        sub = scoreable.loc[scoreable["overtake_success"] == label]
        by_class[name] = {
            "n": int(len(sub)),
            "closing_speed_missing_rate": float(
                (~sub["closing_speed_available"].astype(bool)).mean()
            ) if len(sub) else float("nan"),
            "spatial_alignment_ok_rate": float(sub["spatial_alignment_ok"].astype(bool).mean())
            if len(sub) else float("nan"),
        }

    return {
        "event_type_counts": {t: int((events["event_type"] == t).sum()) for t in EVENT_TYPES},
        "n_events": int(len(events)),
        "n_labelled": int(len(scoreable)),
        "n_positive": n_pass,
        "n_negative": n_no,
        "positive_rate": float(n_pass / (n_pass + n_no)) if (n_pass + n_no) else float("nan"),
        "horizon_s_effective": {
            "min": float(horizon_len.min()),
            "max": float(horizon_len.max()),
            "all_equal_to_horizon": bool(np.allclose(horizon_len, params.horizon_s)),
        },
        "event_duration_s": {
            "min": float(events["event_duration_s"].min()),
            "median": float(events["event_duration_s"].median()),
            "p75": float(events["event_duration_s"].quantile(0.75)),
            "max": float(events["event_duration_s"].max()),
        },
        "cross_start_finish": {
            "n": int(events["window_crosses_start_finish"].sum()),
            "fraction": float(events["window_crosses_start_finish"].mean()),
        },
        "min_same_pair_separation_s": float(sep.min()) if same_pair.any() else float("nan"),
        "closing_speed_missing_rate": float(closing_missing.mean()),
        "closing_speed_missing_by_class": by_class,
        "lookahead_truncated_rate": float(events["lookahead_truncated"].astype(bool).mean()),
        "energy_missing": int(events["EstimatedEnergyIndex"].isna().sum()),
        "in_horizon_pass_delay_s": {
            "median": float(
                pd.to_numeric(
                    events.loc[events["event_type"] == "ON_TRACK_PASS", "actual_swap_delay_s"],
                    errors="coerce",
                ).median()
            ) if n_pass else float("nan"),
            "max": float(
                pd.to_numeric(
                    events.loc[events["event_type"] == "ON_TRACK_PASS", "actual_swap_delay_s"],
                    errors="coerce",
                ).max()
            ) if n_pass else float("nan"),
        },
        "pass_after_horizon": int(events["pass_after_horizon"].sum()),
        "invalid_reasons": (
            audit.loc[audit["event_type"] == "INVALID", "reason"]
            .value_counts()
            .head(12)
            .to_dict()
            if "reason" in audit.columns
            else {}
        ),
    }


def write_race_artifacts(art: RaceArtifacts, out_dir: Path) -> dict:
    """Per-race outputs. Sample-level frames are intentionally not persisted."""
    race_dir = out_dir / art.spec.slug
    race_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if not art.events.empty:
        p = race_dir / f"{art.spec.slug}_events.csv"
        art.events.to_csv(p, index=False)
        written["events"] = str(p)
    if not art.audit.empty:
        p = race_dir / f"{art.spec.slug}_events_audit.csv"
        art.audit.to_csv(p, index=False)
        written["audit"] = str(p)
    if not art.laps_clean.empty:
        p = race_dir / f"{art.spec.slug}_laps_clean.csv"
        art.laps_clean.to_csv(p, index=False)
        written["laps_clean"] = str(p)
    if not art.zones.empty:
        p = race_dir / f"{art.spec.slug}_circuit_zones.csv"
        art.zones.to_csv(p, index=False)
        written["zones"] = str(p)
    return written
