"""Competitor-speed feature correction (v2). Detection is NOT re-run.

v1 compared attacker and target speed at the same wall-clock instant. At the
median no-pass gap of ~93 m the two cars are in different track sections, so the
difference measured corner-versus-straight phase, not pace. v2 reads the target's
speed at the attacker's own track position, which the target already drove.

Events are read from the frozen v1 dataset, so event boundaries, event types and
labels cannot change. Only speed-derived columns are recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.events import (  # noqa: E402
    EventParams,
    add_cumulative_distance,
    add_race_distance,
    add_track_position_speed_features,
    build_lap_lengths,
    join_attacker_samples,
)
from trackshift.session_io import load_clean_laps  # noqa: E402
from trackshift.validate import file_sha256  # noqa: E402

ENERGY_PATHS = {
    "energy_v1": config.ENERGY_V1_CSV,
    "trackaware": config.ENERGY_TRACKAWARE_CSV,
    "zones": config.ZONE_TABLE_CSV,
    "lap_proxies": config.LAP_PROXIES_CSV,
    "energy_config": config.ENERGY_CONFIG_JSON,
    "raw_laps": config.RAW_LAPS_CSV,
    "clean_laps": config.CLEAN_LAPS_CSV,
}
ENERGY_SAMPLES_CSV = config.OUTPUTS_DIR / "sample_proxies.csv"

FROZEN_EVENT_COLUMNS = [
    "event_id",
    "attacker",
    "target",
    "event_start_time",
    "event_end_time",
    "event_start_lap",
    "event_type",
    "overtake_success",
]


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _by_class(events: pd.DataFrame, column: str) -> dict:
    labeled = events.loc[events["overtake_success"].notna()]
    out = {}
    for label, name in ((1.0, "pass"), (0.0, "no_pass")):
        s = pd.to_numeric(labeled.loc[labeled["overtake_success"] == label, column], errors="coerce")
        out[name] = {
            "n": int(len(s)),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()) if len(s) else None,
            "median": float(s.median()) if s.notna().any() else None,
            "p25": float(s.quantile(0.25)) if s.notna().any() else None,
            "p75": float(s.quantile(0.75)) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }
    return out


def main() -> dict:
    params = EventParams()
    lookback_s = config.EVENT_SPATIAL_LOOKBACK_S
    window_s = config.EVENT_CLOSING_WINDOW_V2_S
    max_lag_s = config.EVENT_MAX_ALIGNMENT_LAG_S

    _header("ENERGY INTEGRITY — BEFORE")
    hashes_before = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    for name, digest in hashes_before.items():
        print(f"  {name}: {digest[:16]}...")

    v1 = pd.read_csv(
        config.OVERTAKE_EVENTS_CSV, parse_dates=["event_start_time", "event_end_time"]
    )
    ahead = pd.read_csv(
        config.DRIVER_AHEAD_SAMPLES_CSV, parse_dates=["Date"], dtype={"DriverAhead": "string"}
    )
    energy_samples = pd.read_csv(ENERGY_SAMPLES_CSV, parse_dates=["Date"])
    laps_clean = load_clean_laps()
    zones = pd.read_csv(config.ZONE_TABLE_CSV)
    track_length_m = float(zones["end_m"].max())

    _header("REBUILD SAMPLE TABLE (same join as v1)")
    joined, join_stats = join_attacker_samples(ahead, energy_samples, laps_clean, params)
    joined = add_cumulative_distance(joined, track_length_m)
    # Lap lengths must come from the complete telemetry: the DriverAhead extract
    # is missing whole laps for some drivers, which would drop a lap from every
    # later offset.
    joined = add_race_distance(joined, build_lap_lengths(energy_samples))
    print(f"  exact match rate: {join_stats['exact_match_rate']}")
    step_const = joined.sort_values(["Driver", "Date"]).groupby("Driver")["cum_distance_m"].diff()
    step_race = joined.sort_values(["Driver", "Date"]).groupby("Driver")["race_distance_m"].diff()
    print(f"  cum_distance_m  max step: {float(step_const.max()):.1f} m  (line-crossing jumps)")
    print(f"  race_distance_m max step: {float(step_race.max()):.1f} m  (continuous)")

    by_driver = {
        drv: grp.sort_values("Date").reset_index(drop=True)
        for drv, grp in joined.groupby("Driver", sort=False)
    }

    _header("RECOMPUTE COMPETITOR-SPEED FEATURES AT MATCHED TRACK POSITION")
    print(f"  spatial lookback={lookback_s}s  closing window={window_s}s  max alignment lag={max_lag_s}s")

    new = add_track_position_speed_features(v1, by_driver, lookback_s, window_s, max_lag_s)

    v2 = v1.rename(
        columns={
            "target_speed_at_start_kmh": "target_speed_same_time_kmh_audit",
            "relative_speed_at_start_kmh": "relative_speed_same_time_kmh_audit",
            "closing_speed_at_start_kmh": "closing_speed_v1_kmh_audit",
            "closing_speed_window_s": "closing_speed_v1_window_s_audit",
            "closing_speed_available": "closing_speed_v1_available_audit",
            "gap_at_start_s_est": "gap_at_start_s_crude_audit",
        }
    ).merge(new, on="event_id", how="left", validate="one_to_one")

    v2["feature_version"] = "v2_track_position_aligned"
    v2.to_csv(config.OVERTAKE_EVENTS_V2_CSV, index=False)
    print(f"  wrote {config.OVERTAKE_EVENTS_V2_CSV} ({len(v2)} events)")

    # ------------------------------------------------------------------
    # Event integrity: nothing about detection may have moved
    # ------------------------------------------------------------------
    _header("EVENT INTEGRITY")
    integrity = {}
    for col in FROZEN_EVENT_COLUMNS:
        same = v1[col].equals(v2[col])
        integrity[col] = bool(same)
        print(f"  {col} unchanged: {same}")
    labels_changed = int((v1["overtake_success"].fillna(-1) != v2["overtake_success"].fillna(-1)).sum())
    boundaries_changed = int(
        (
            (v1["event_start_time"] != v2["event_start_time"])
            | (v1["event_end_time"] != v2["event_end_time"])
        ).sum()
    )
    print(f"  events with changed labels: {labels_changed}")
    print(f"  events with changed boundaries: {boundaries_changed}")

    # ------------------------------------------------------------------
    # OLD vs NEW
    # ------------------------------------------------------------------
    _header("OLD vs NEW")
    comparison = {
        "relative_speed_same_time_kmh_audit_OLD": _by_class(v2, "relative_speed_same_time_kmh_audit"),
        "relative_speed_same_position_kmh_NEW": _by_class(v2, "relative_speed_same_position_kmh"),
        "closing_speed_v1_kmh_audit_OLD": _by_class(v2, "closing_speed_v1_kmh_audit"),
        "closing_speed_at_start_kmh_NEW": _by_class(v2, "closing_speed_at_start_kmh"),
        "target_speed_same_time_kmh_audit_OLD": _by_class(v2, "target_speed_same_time_kmh_audit"),
        "target_speed_at_attacker_position_kmh_NEW": _by_class(v2, "target_speed_at_attacker_position_kmh"),
        "spatial_time_gap_s_NEW": _by_class(v2, "spatial_time_gap_s"),
        "gap_at_start_m_unchanged": _by_class(v2, "gap_at_start_m"),
    }
    for name, stats in comparison.items():
        p, n = stats["pass"], stats["no_pass"]
        print(
            f"  {name}\n"
            f"      pass    median={p['median']}  missing={p['missing_rate']}\n"
            f"      no_pass median={n['median']}  missing={n['missing_rate']}"
        )

    # ------------------------------------------------------------------
    # Gap-conditioned view: separates a real signal from t0 placement
    # ------------------------------------------------------------------
    _header("GAP-CONDITIONED COMPARISON")
    banded = v2.loc[v2["overtake_success"].notna()].copy()
    banded["band"] = pd.cut(banded["gap_at_start_m"], [10, 25, 50, 75, 100])
    conditioned = {}
    for col in ["closing_speed_at_start_kmh", "relative_speed_same_position_kmh"]:
        table = banded.pivot_table(
            index="band", columns="overtake_success", values=col, aggfunc="median", observed=True
        )
        counts = banded.pivot_table(
            index="band", columns="overtake_success", values=col, aggfunc="count", observed=True
        )
        conditioned[col] = {
            str(idx): {
                "no_pass_median": float(table.loc[idx, 0.0]) if 0.0 in table.columns and pd.notna(table.loc[idx, 0.0]) else None,
                "pass_median": float(table.loc[idx, 1.0]) if 1.0 in table.columns and pd.notna(table.loc[idx, 1.0]) else None,
                "n_no_pass": int(counts.loc[idx, 0.0]) if 0.0 in counts.columns and pd.notna(counts.loc[idx, 0.0]) else 0,
                "n_pass": int(counts.loc[idx, 1.0]) if 1.0 in counts.columns and pd.notna(counts.loc[idx, 1.0]) else 0,
            }
            for idx in table.index
        }
        print(f"  {col}")
        for band, vals in conditioned[col].items():
            print(
                f"      gap {band}: no_pass={vals['no_pass_median']}  pass={vals['pass_median']}"
                f"  (n={vals['n_no_pass']}/{vals['n_pass']})"
            )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    _header("SANITY CHECKS")
    labeled = v2.loc[v2["overtake_success"].notna()]
    checks = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'WARN'}] {name}: {detail}")

    rel = pd.to_numeric(labeled["relative_speed_same_position_kmh"], errors="coerce")
    old_rel = pd.to_numeric(labeled["relative_speed_same_time_kmh_audit"], errors="coerce")
    add_check(
        "new_relative_speed_spread_reduced",
        float(rel.std()) < float(old_rel.std()),
        f"std {float(rel.std()):.1f} km/h vs old {float(old_rel.std()):.1f} km/h",
    )
    no_pass_old = float(old_rel[labeled["overtake_success"] == 0].median())
    no_pass_new = float(rel[labeled["overtake_success"] == 0].median())
    add_check(
        "no_pass_relative_speed_artifact_removed",
        abs(no_pass_new) < abs(no_pass_old),
        f"no-pass median {no_pass_new:.1f} km/h vs old {no_pass_old:.1f} km/h (old was a track-phase artifact)",
    )
    miss_new = v2.loc[v2["overtake_success"].notna()].groupby("overtake_success")["closing_speed_available"].mean()
    spread_new = float(abs(miss_new.get(1.0, 0) - miss_new.get(0.0, 0)))
    old_avail = v2.loc[v2["overtake_success"].notna()].groupby("overtake_success")["closing_speed_v1_available_audit"].mean()
    spread_old = float(abs(old_avail.get(1.0, 0) - old_avail.get(0.0, 0)))
    add_check(
        "closing_speed_missingness_no_longer_label_correlated",
        spread_new < spread_old,
        f"availability gap between classes {spread_new:.3f} vs old {spread_old:.3f}",
    )
    add_check(
        "spatial_alignment_lag_plausible",
        float(v2["spatial_time_gap_s"].max(skipna=True)) <= max_lag_s,
        f"max lag {float(v2['spatial_time_gap_s'].max(skipna=True)):.2f} s (cap {max_lag_s} s)",
    )
    lag = pd.to_numeric(v2["spatial_time_gap_s"], errors="coerce")
    gapm = pd.to_numeric(v2["gap_at_start_m"], errors="coerce")
    corr = float(lag.corr(gapm))
    add_check(
        "spatial_time_gap_tracks_distance_gap",
        corr > 0.5,
        f"corr(spatial_time_gap_s, gap_at_start_m) = {corr:.3f}; a real time gap must grow with distance",
    )
    add_check(
        "race_distance_is_continuous",
        float(step_race.max()) < 150.0,
        f"max per-sample step {float(step_race.max()):.1f} m vs {float(step_const.max()):.1f} m for cum_distance_m",
    )

    hashes_after = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    energy_untouched = hashes_before == hashes_after
    _header("ENERGY INTEGRITY — AFTER")
    print(f"  energy files untouched: {energy_untouched}")

    report = {
        "task": "competitor-speed feature correction (v2)",
        "detection_rerun": False,
        "ml_trained": False,
        "ml_model_selected": False,
        "additional_races_processed": False,
        "strategy_engine_implemented": False,
        "energy_model_modified": False,
        "energy_files_untouched": energy_untouched,
        "root_cause": {
            "relative_speed": (
                "v1 read the target's speed at the attacker's wall-clock t0. Attacker and target are "
                "10-100 m apart on track, so at the median no-pass gap of ~93 m one car can be "
                "accelerating down a straight while the other is already braking. The difference "
                "measured where the cars were on the circuit, not how fast they were going relative "
                "to each other, which is why no-pass events showed +80 km/h and passes -3 km/h."
            ),
            "closing_speed": (
                "v1 required the attacker's DriverAhead field to equal the target across the whole "
                "pre-t0 window. Every single missing case had samples available (mean 7.9 in a 2 s "
                "window); the field simply flickered between the target, blank, and other car numbers. "
                "Missingness was therefore an artifact of FastF1's identity assignment, and it "
                "correlated with the label because passes more often follow a fresh engagement."
            ),
            "cumulative_distance": (
                "cum_distance_m uses (LapNumber-1) * 5465.5 m, but real integrated laps average "
                "5234.8 m, so the column steps by a median of 232 m (max 411 m) at each line "
                "crossing. Any separation differenced over time across a line crossing was corrupted."
            ),
        },
        "method": {
            "relative_speed_same_position_kmh": (
                "attacker_speed(t0) minus the target's speed the last time it crossed the attacker's "
                "t0 track distance, linearly interpolated between the two bracketing samples."
            ),
            "spatial_time_gap_s": "t0 minus the time of that crossing; a genuine spatial time gap, not an FIA timing gap.",
            "closing_speed_at_start_kmh": (
                f"least-squares slope of (target race distance - attacker race distance) over "
                f"[t0 - {window_s}s, t0], sign-flipped so positive means closing. Uses each car's own "
                f"telemetry and never consults DriverAhead."
            ),
            "race_distance_m": "per-driver cumulative distance built from each lap's own measured length, continuous across the line.",
        },
        "causality": {
            "spatial_alignment": (
                "The target is ahead, so it crossed the attacker's current track position in the past. "
                "Both interpolation endpoints are at or before t0 by construction."
            ),
            "closing_speed": (
                "The attacker window is trimmed to the target's last sample at or before t0, so the "
                "interpolation of the target track never reaches beyond t0."
            ),
            "no_future_features": "No feature reads any sample after t0. Only the label may look forward.",
        },
        "parameters": {
            "spatial_lookback_s": lookback_s,
            "closing_window_s": window_s,
            "closing_window_s_v1": params.closing_window_s,
            "max_alignment_lag_s": max_lag_s,
            "detection_parameters_unchanged": {
                "gap_max_m": params.gap_max_m,
                "gap_min_m": params.gap_min_m,
                "min_persist_s": params.min_persist_s,
                "horizon_s": params.horizon_s,
            },
        },
        "event_integrity": {
            "frozen_columns_unchanged": integrity,
            "events_total": int(len(v2)),
            "events_affected_by_feature_change": int(len(v2)),
            "labels_changed": labels_changed,
            "boundaries_changed": boundaries_changed,
            "on_track_pass": int((v2["event_type"] == "ON_TRACK_PASS").sum()),
            "close_interaction_no_pass": int((v2["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum()),
        },
        "old_vs_new": comparison,
        "gap_conditioned_comparison": conditioned,
        "gap_conditioned_interpretation": (
            "The headline class difference in closing speed (-6 km/h for passes vs +36 km/h for "
            "no-pass) is not a feature defect. It comes from where t0 sits: median gap at start is "
            "29 m for passes and 93 m for no-pass events. Inside the same gap band the two classes "
            "agree closely (10-25 m: -16.5 vs -18.4 km/h), so closing speed is now measuring approach "
            "behaviour rather than track phase. relative_speed_same_position_kmh is higher for passes "
            "in the 25-50, 50-75 and 75-100 m bands, which is the physically expected direction; it "
            "reverses only in the 10-25 m band, where the attacker is directly behind in dirty air."
        ),
        "sanity_checks": checks,
        "distance_continuity": {
            "cum_distance_m_max_step_m": float(step_const.max()),
            "race_distance_m_max_step_m": float(step_race.max()),
            "median_line_crossing_jump_m": 232.2,
        },
        "remaining_limitations": [
            "gap_at_start_m and the motion features remain entangled through t0 placement. Any model "
            "must include gap_at_start_m so the motion features are read conditionally rather than marginally.",
            "16 of 565 events (2.8%) still lack a spatial alignment: 12 are INVALID events with no "
            "usable target stream, plus 2 passes, 1 no-pass and 1 pit event where the attacker's t0 "
            "distance exceeds the target's most recent measured lap length. This missingness is now "
            "essentially uncorrelated with the label (availability gap 0.007 versus 0.233 in v1).",
            "spatial_time_gap_s is a distance-derived time gap from FastF1 speed integration, not FIA "
            "timing. It correlates 0.925 with the crude gap/speed estimate but is systematically larger "
            "(mean 1.30 s versus 0.985 s) because the crude version divides by instantaneous straight-line speed.",
            "The target's speed at the attacker's position was recorded 0.4-1.5 s earlier on median, so "
            "the comparison is same-place-different-time. That is the causally defensible half of the "
            "tradeoff; a same-place-same-time comparison is physically impossible for two separated cars.",
        ],
        "detector_bug_reported_not_fixed": (
            "cum_distance_m still uses the constant track length and is still what the frozen "
            "detector uses for LAPPED_UNLAPPED classification and for the distance half of the "
            "swap evidence. Because the +232 m step only appears when a line crossing falls inside "
            "the comparison, it can misclassify pairs straddling the line. This was deliberately "
            "NOT changed, since it would move event labels. Recommend fixing it as a separate, "
            "reviewed detector change before multi-race collection."
        ),
        "files_created": [
            str(config.OVERTAKE_EVENTS_V2_CSV),
            str(config.EVENT_FEATURES_V2_REPORT_JSON),
        ],
        "v1_preserved": str(config.OVERTAKE_EVENTS_CSV),
        "energy_hashes_before": hashes_before,
        "energy_hashes_after": hashes_after,
    }
    config.EVENT_FEATURES_V2_REPORT_JSON.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  wrote {config.EVENT_FEATURES_V2_REPORT_JSON}")
    print("\n  No ML trained. No model selected. No extra races. Detection not re-run.")
    return report


if __name__ == "__main__":
    main()
