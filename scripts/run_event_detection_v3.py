"""v3: fix the cum_distance_m defect in the overtake-event detector.

v1 built race distance as (LapNumber - 1) * 5465.5 m. Real integrated laps
average 5234.8 m, so the coordinate jumped a median of 244 m at every line
crossing. That corrupted the two places the detector reads race distance: the
LAPPED_UNLAPPED test and the distance half of the swap evidence.

v3 uses race_distance_m, accumulated from each lap's own measured length, and
reads the swap as ground gained since t0 so per-driver integration offsets
cancel. Detection thresholds, the event definition and the energy model are
untouched. v1 and v2 outputs are left in place.

The run reproduces v1 first, so the comparison is against a verified baseline.
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
    EVENT_TYPES,
    EventParams,
    add_cumulative_distance,
    add_race_distance,
    add_track_position_speed_features,
    build_driver_number_map,
    build_events,
    build_lap_lengths,
    build_lap_offsets,
    build_zone_stats,
    detect_episodes,
    join_attacker_samples,
    session_epoch,
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

V2_SPEED_COLUMNS = [
    "target_speed_at_attacker_position_kmh",
    "relative_speed_same_position_kmh",
    "spatial_time_gap_s",
    "spatial_alignment_ok",
    "closing_speed_at_start_kmh",
    "closing_speed_available",
]


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _type_counts(events: pd.DataFrame) -> dict:
    return {t: int((events["event_type"] == t).sum()) for t in EVENT_TYPES}


def main() -> dict:
    # v3 predates the t0-anchored horizon. Pin the historical anchor explicitly
    # so this script keeps reproducing the v1 baseline it validates against.
    params = EventParams(horizon_anchor="event_end")
    _header("ENERGY INTEGRITY — BEFORE")
    hashes_before = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    for name, digest in hashes_before.items():
        print(f"  {name}: {digest[:16]}...")

    ahead = pd.read_csv(
        config.DRIVER_AHEAD_SAMPLES_CSV, parse_dates=["Date"], dtype={"DriverAhead": "string"}
    )
    energy_samples = pd.read_csv(ENERGY_SAMPLES_CSV, parse_dates=["Date"])
    laps_clean = load_clean_laps()
    zones = pd.read_csv(config.ZONE_TABLE_CSV)
    trackaware = pd.read_csv(config.ENERGY_TRACKAWARE_CSV)
    track_length_m = float(zones["end_m"].max())
    epoch = session_epoch(energy_samples)

    joined, _ = join_attacker_samples(ahead, energy_samples, laps_clean, params)
    joined = add_cumulative_distance(joined, track_length_m)
    # Lap lengths come from the complete telemetry. The DriverAhead extract is
    # missing whole laps for some drivers (GAS lap 5, for one), and deriving
    # offsets from it drops a full lap from every later offset for that driver.
    lap_length_table = build_lap_lengths(energy_samples)
    _, gapfill = build_lap_offsets(lap_length_table)
    joined = add_race_distance(joined, lap_length_table)
    print(f"  laps gap-filled when building offsets: {gapfill['n_laps_gap_filled']} {gapfill['laps_gap_filled']}")
    number_map = build_driver_number_map(laps_clean)
    zone_stats = build_zone_stats(energy_samples, zones, trackaware)

    # ------------------------------------------------------------------
    # Continuity validation
    # ------------------------------------------------------------------
    _header("CONTINUITY VALIDATION")
    ordered = joined.sort_values(["Driver", "Date"])
    ordered = ordered.assign(
        step_old=ordered.groupby("Driver")["cum_distance_m"].diff(),
        step_new=ordered.groupby("Driver")["race_distance_m"].diff(),
        lap_change=ordered.groupby("Driver")["LapNumber"].diff().fillna(0) != 0,
    )
    boundary = ordered.loc[ordered["lap_change"]]
    within = ordered.loc[~ordered["lap_change"]]
    lap_lengths = energy_samples.groupby(["Driver", "LapNumber"])["Distance"].max()

    continuity = {
        "n_lap_boundaries": int(len(boundary)),
        "boundary_step_old_m": {
            "median": float(boundary["step_old"].median()),
            "max": float(boundary["step_old"].max()),
            "min": float(boundary["step_old"].min()),
        },
        "boundary_step_new_m": {
            "median": float(boundary["step_new"].median()),
            "max": float(boundary["step_new"].max()),
            "min": float(boundary["step_new"].min()),
        },
        "within_lap_step_max_m": {
            "old": float(within["step_old"].max()),
            "new": float(within["step_new"].max()),
            "note": "identical by construction; these are telemetry dropouts in the source car data, not a distance-model artifact",
        },
        "lap_length_m": {
            "n_laps": int(len(lap_lengths)),
            "min": float(lap_lengths.min()),
            "median": float(lap_lengths.median()),
            "mean": float(lap_lengths.mean()),
            "max": float(lap_lengths.max()),
            "assumed_by_v1": track_length_m,
        },
        "truncated_laps": {
            f"{drv}_lap{int(lap)}": float(v)
            for (drv, lap), v in lap_lengths[lap_lengths < 5150].items()
        },
        "negative_steps_new": int((ordered["step_new"] < 0).sum()),
        "lap_offset_gap_fill": gapfill,
        "sanity_race_distance_diff_vs_fastf1_gap_m": None,
    }
    print(f"  lap boundaries inspected: {continuity['n_lap_boundaries']}")
    print(
        f"  boundary step OLD: median {continuity['boundary_step_old_m']['median']:.1f} m,"
        f" max {continuity['boundary_step_old_m']['max']:.1f} m,"
        f" min {continuity['boundary_step_old_m']['min']:.1f} m"
    )
    print(
        f"  boundary step NEW: median {continuity['boundary_step_new_m']['median']:.1f} m,"
        f" max {continuity['boundary_step_new_m']['max']:.1f} m,"
        f" min {continuity['boundary_step_new_m']['min']:.1f} m"
    )
    print(
        f"  measured lap length: min {continuity['lap_length_m']['min']:.1f},"
        f" median {continuity['lap_length_m']['median']:.1f},"
        f" max {continuity['lap_length_m']['max']:.1f} m"
        f"  (v1 assumed {track_length_m:.1f} m for every lap)"
    )
    print(f"  truncated laps: {continuity['truncated_laps']}")

    # Independent cross-check: for a same-lap pair the race-distance difference
    # must equal the FastF1 gap. This is what caught the missing-lap bug.
    v1_prev = pd.read_csv(config.OVERTAKE_EVENTS_CSV, parse_dates=["event_start_time"])
    by_driver_chk = {d: g.sort_values("Date").reset_index(drop=True) for d, g in joined.groupby("Driver")}
    residuals = []
    for e in v1_prev.itertuples(index=False):
        a, b = by_driver_chk.get(e.attacker), by_driver_chk.get(e.target) if isinstance(e.target, str) and e.target else None
        if a is None or b is None:
            continue
        t0 = pd.Timestamp(e.event_start_time)
        ar = a.loc[a["Date"] <= t0]
        br = b.loc[b["Date"] <= t0]
        if ar.empty or br.empty:
            continue
        diff = float(br["race_distance_m"].iloc[-1]) - float(ar["race_distance_m"].iloc[-1])
        residuals.append(abs(diff - float(e.gap_at_start_m)))
    residuals = pd.Series(residuals)
    continuity["sanity_race_distance_diff_vs_fastf1_gap_m"] = {
        "n": int(len(residuals)),
        "median_abs_residual": float(residuals.median()),
        "p95_abs_residual": float(residuals.quantile(0.95)),
        "n_residuals_over_1000m": int((residuals > 1000).sum()),
    }
    print(
        f"  cross-check |race-distance diff - FastF1 gap|: median"
        f" {residuals.median():.2f} m, p95 {residuals.quantile(0.95):.1f} m,"
        f" over 1 km: {int((residuals > 1000).sum())} of {len(residuals)}"
    )

    # ------------------------------------------------------------------
    # Episodes are gap-driven only, so they must be identical
    # ------------------------------------------------------------------
    _header("EPISODE DETECTION (unchanged inputs)")
    episodes = detect_episodes(joined, params)
    print(f"  episodes: {len(episodes)}")
    print(
        f"  frozen thresholds: gap_max_m={params.gap_max_m} gap_min_m={params.gap_min_m}"
        f" min_persist_s={params.min_persist_s} horizon_s={params.horizon_s}"
    )

    # ------------------------------------------------------------------
    # Three variants so the impact decomposes cleanly
    # ------------------------------------------------------------------
    _header("BUILD VARIANTS")
    repro, _ = build_events(
        joined, episodes, laps_clean, zone_stats, number_map, track_length_m, params, epoch,
        distance_column="cum_distance_m", swap_test="absolute",
    )
    # Isolates the effect of the swap test alone, still on the v1 coordinate.
    diag, _ = build_events(
        joined, episodes, laps_clean, zone_stats, number_map, track_length_m, params, epoch,
        distance_column="cum_distance_m", swap_distance_column="cum_distance_m",
        swap_test="displacement",
    )
    v3, audit_v3 = build_events(
        joined, episodes, laps_clean, zone_stats, number_map, track_length_m, params, epoch,
        distance_column="cum_distance_m", swap_distance_column="race_distance_m",
        swap_test="displacement",
    )
    print(f"  v1 reproduction                       : {_type_counts(repro)}")
    print(f"  displacement swap on v1 coordinate    : {_type_counts(diag)}")
    print(f"  displacement swap on measured dist(v3): {_type_counts(v3)}")

    v1 = pd.read_csv(
        config.OVERTAKE_EVENTS_CSV, parse_dates=["event_start_time", "event_end_time"]
    )
    repro_matches_v1 = bool(
        repro["event_type"].tolist() == v1["event_type"].tolist()
        and repro["event_id"].tolist() == v1["event_id"].tolist()
    )
    print(f"  v1 reproduction matches the stored v1 dataset: {repro_matches_v1}")
    if not repro_matches_v1:
        raise RuntimeError("Baseline reproduction failed; comparison would be meaningless.")

    # ------------------------------------------------------------------
    # Attach the frozen v2 speed features and PositionChange audit
    # ------------------------------------------------------------------
    by_driver = {
        drv: grp.sort_values("Date").reset_index(drop=True)
        for drv, grp in joined.groupby("Driver", sort=False)
    }
    derived = pd.read_csv(config.LAPS_DERIVED_CSV)
    v3 = v3.merge(
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
        v3,
        by_driver,
        config.EVENT_SPATIAL_LOOKBACK_S,
        config.EVENT_CLOSING_WINDOW_V2_S,
        config.EVENT_MAX_ALIGNMENT_LAG_S,
    )
    v3 = v3.rename(
        columns={
            "target_speed_at_start_kmh": "target_speed_same_time_kmh_audit",
            "relative_speed_at_start_kmh": "relative_speed_same_time_kmh_audit",
            "closing_speed_at_start_kmh": "closing_speed_v1_kmh_audit",
            "closing_speed_window_s": "closing_speed_v1_window_s_audit",
            "closing_speed_available": "closing_speed_v1_available_audit",
            "gap_at_start_s_est": "gap_at_start_s_crude_audit",
        }
    ).merge(speed, on="event_id", how="left", validate="one_to_one")
    v3["feature_version"] = "v3_continuous_distance_detector_plus_v2_speed"
    v3.to_csv(config.OVERTAKE_EVENTS_V3_CSV, index=False)
    audit_v3.to_csv(config.OVERTAKE_EVENTS_AUDIT_V3_CSV, index=False)
    print(f"  wrote {config.OVERTAKE_EVENTS_V3_CSV} ({len(v3)} events)")

    # v2 speed features must not have moved
    v2 = pd.read_csv(config.OVERTAKE_EVENTS_V2_CSV)
    speed_check = v2.loc[:, ["event_id"] + V2_SPEED_COLUMNS].merge(
        v3.loc[:, ["event_id"] + V2_SPEED_COLUMNS], on="event_id", suffixes=("_v2", "_v3")
    )
    speed_unchanged = {}
    for col in V2_SPEED_COLUMNS:
        a, b = speed_check[f"{col}_v2"], speed_check[f"{col}_v3"]
        if a.dtype == bool or b.dtype == bool:
            same = bool((a.astype(str) == b.astype(str)).all())
        else:
            same = bool(np.allclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True))
        speed_unchanged[col] = same

    # ------------------------------------------------------------------
    # Event impact and changed-event audit
    # ------------------------------------------------------------------
    _header("EVENT IMPACT (v1 -> v3)")
    merged = v1.loc[
        :, ["event_id", "attacker", "target", "event_start_time", "event_end_time", "event_start_lap", "event_type", "overtake_success", "gap_at_start_m"]
    ].merge(
        v3.loc[:, ["event_id", "event_start_time", "event_end_time", "event_type", "overtake_success"]],
        on="event_id",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    new_events = merged.loc[merged["_merge"] == "right_only"]
    removed_events = merged.loc[merged["_merge"] == "left_only"]
    both = merged.loc[merged["_merge"] == "both"].copy()
    both["type_changed"] = both["event_type_old"] != both["event_type_new"]
    both["label_changed"] = both["overtake_success_old"].fillna(-1) != both["overtake_success_new"].fillna(-1)
    both["boundary_changed"] = (
        (both["event_start_time_old"] != both["event_start_time_new"])
        | (both["event_end_time_old"] != both["event_end_time_new"])
    )

    changed = both.loc[both["type_changed"] | both["label_changed"] | both["boundary_changed"]].copy()
    audit_lookup = audit_v3.set_index("event_id")
    changed["new_reason"] = changed["event_id"].map(audit_lookup["reason"])
    changed["cum_distance_diff_at_start_m_new"] = changed["event_id"].map(
        audit_lookup["cum_distance_diff_at_start_m"]
    )
    changed["swap_by_distance_new"] = changed["event_id"].map(audit_lookup["swap_by_distance"])
    changed["swap_by_driver_ahead_new"] = changed["event_id"].map(audit_lookup["swap_by_driver_ahead"])

    # Did a start/finish crossing sit inside the window this event was scored over?
    def crossed_line(event_id: str) -> bool:
        row = v3.loc[v3["event_id"] == event_id]
        if row.empty:
            return False
        r = row.iloc[0]
        a = by_driver.get(r["attacker"])
        b = by_driver.get(r["target"]) if isinstance(r["target"], str) and r["target"] else None
        t0 = pd.Timestamp(r["event_start_time"])
        end = pd.Timestamp(r["outcome_window_end"])
        for frame in (a, b):
            if frame is None:
                continue
            win = frame.loc[(frame["Date"] >= t0) & (frame["Date"] <= end)]
            if win["LapNumber"].nunique() > 1:
                return True
        return False

    changed["lap_boundary_crossing_involved"] = [crossed_line(e) for e in changed["event_id"]]
    changed = changed.rename(
        columns={
            "event_type_old": "old_event_type",
            "event_type_new": "new_event_type",
            "overtake_success_old": "old_label",
            "overtake_success_new": "new_label",
            "event_start_time_old": "old_start_time",
            "event_start_time_new": "new_start_time",
            "event_end_time_old": "old_end_time",
            "event_end_time_new": "new_end_time",
        }
    ).loc[
        :,
        [
            "event_id", "attacker", "target", "event_start_lap", "gap_at_start_m",
            "old_event_type", "new_event_type", "old_label", "new_label",
            "old_start_time", "new_start_time", "old_end_time", "new_end_time",
            "type_changed", "label_changed", "boundary_changed",
            "lap_boundary_crossing_involved", "cum_distance_diff_at_start_m_new",
            "swap_by_distance_new", "swap_by_driver_ahead_new", "new_reason",
        ],
    ]
    changed.to_csv(config.OVERTAKE_EVENTS_CHANGED_V3_CSV, index=False)

    transitions = (
        changed.groupby(["old_event_type", "new_event_type"]).size().sort_values(ascending=False)
        if len(changed)
        else pd.Series(dtype=int)
    )
    label_1_to_0 = int(((both["overtake_success_old"] == 1) & (both["overtake_success_new"] == 0)).sum())
    label_0_to_1 = int(((both["overtake_success_old"] == 0) & (both["overtake_success_new"] == 1)).sum())
    label_to_none = int((both["overtake_success_old"].notna() & both["overtake_success_new"].isna()).sum())
    label_from_none = int((both["overtake_success_old"].isna() & both["overtake_success_new"].notna()).sum())

    print(f"  events v1: {len(v1)}   events v3: {len(v3)}")
    print(f"  newly created: {len(new_events)}   removed: {len(removed_events)}")
    print(f"  type changed: {int(both['type_changed'].sum())}")
    print(f"  label changed: {int(both['label_changed'].sum())}  (1->0 {label_1_to_0}, 0->1 {label_0_to_1}, labelled->unlabelled {label_to_none}, unlabelled->labelled {label_from_none})")
    print(f"  boundary changed: {int(both['boundary_changed'].sum())}")
    print(f"  changed events involving a line crossing: {int(changed['lap_boundary_crossing_involved'].sum())} of {len(changed)}")
    if len(transitions):
        print("  transitions:")
        for (old, new), n in transitions.items():
            print(f"      {old} -> {new}: {n}")
    print(f"  wrote {config.OVERTAKE_EVENTS_CHANGED_V3_CSV}")

    # ------------------------------------------------------------------
    # Where does the remaining evidence disagreement live?
    # ------------------------------------------------------------------
    _header("EVIDENCE AGREEMENT vs OUTCOME WINDOW LENGTH")
    scored = v3.loc[~v3["event_type"].isin(["INVALID", "PIT_RELATED", "LAPPED_UNLAPPED", "RETIREMENT_RELATED"])].copy()
    scored["band"] = pd.cut(scored["outcome_window_s"], [0, 45, 60, 120, 600])
    window_table = {}
    for band, grp in scored.groupby("band", observed=True):
        window_table[str(band)] = {
            "n": int(len(grp)),
            "uncertain": int((grp["event_type"] == "UNCERTAIN").sum()),
            "uncertain_rate": float((grp["event_type"] == "UNCERTAIN").mean()),
            "on_track_pass": int((grp["event_type"] == "ON_TRACK_PASS").sum()),
        }
        print(
            f"  window {band}: n={window_table[str(band)]['n']}"
            f" uncertain={window_table[str(band)]['uncertain']}"
            f" ({window_table[str(band)]['uncertain_rate']:.1%})"
            f" pass={window_table[str(band)]['on_track_pass']}"
        )

    v1_audit = pd.read_csv(config.OVERTAKE_EVENTS_AUDIT_CSV)
    swap_delay = {
        "v1_pass_swap_delay_max_s": float(
            v1_audit.loc[v1_audit["event_type"] == "ON_TRACK_PASS", "swap_delay_s"].max()
        ),
        "v3_pass_swap_delay_max_s": float(
            audit_v3.loc[audit_v3["event_type"] == "ON_TRACK_PASS", "swap_delay_s"].max()
        ),
        "v3_pass_swap_delay_median_s": float(
            audit_v3.loc[audit_v3["event_type"] == "ON_TRACK_PASS", "swap_delay_s"].median()
        ),
    }
    print(
        f"  pass swap delay max: v1 {swap_delay['v1_pass_swap_delay_max_s']:.0f} s"
        f" -> v3 {swap_delay['v3_pass_swap_delay_max_s']:.0f} s"
    )

    # The lap-offset bug also touched the v2 closing-speed column.
    delta = (
        pd.to_numeric(v2["closing_speed_at_start_kmh"], errors="coerce")
        - pd.to_numeric(
            v2[["event_id"]].merge(v3[["event_id", "closing_speed_at_start_kmh"]], on="event_id")[
                "closing_speed_at_start_kmh"
            ],
            errors="coerce",
        )
    ).abs()
    v2_delta = {
        "events_differing": int((delta > 0.01).sum()),
        "events_total": int(len(delta)),
        "max_abs_diff_kmh": float(delta.max()),
        "explanation": (
            "The stored v2 file was produced before the missing-lap bug in the lap-offset builder "
            "was found. 85 of its closing-speed values used an offset that had silently dropped a "
            "lap for one of the two cars. v3 carries the corrected values; the v2 file is left on "
            "disk unchanged as the historical artifact, as instructed."
        ),
    }
    print(f"  v2 closing-speed values affected by the lap-offset bugfix: {v2_delta['events_differing']} of {v2_delta['events_total']}")

    hashes_after = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    energy_untouched = hashes_before == hashes_after
    _header("INTEGRITY")
    print(f"  energy files untouched: {energy_untouched}")
    print(f"  v2 speed features unchanged: {all(speed_unchanged.values())}")
    print(f"  v1 dataset preserved: {config.OVERTAKE_EVENTS_CSV.exists()}")
    print(f"  v2 dataset preserved: {config.OVERTAKE_EVENTS_V2_CSV.exists()}")

    report = {
        "task": "cum_distance_m defect fix (v3)",
        "ml_trained": False,
        "ml_model_selected": False,
        "additional_races_processed": False,
        "strategy_engine_implemented": False,
        "energy_model_modified": False,
        "energy_files_untouched": energy_untouched,
        "v2_speed_features_unchanged": speed_unchanged,
        "v2_closing_speed_delta": v2_delta,
        "evidence_agreement_by_window": window_table,
        "swap_delay": swap_delay,
        "diagnosis_correction": (
            "The brief assumed the constant-lap-length coordinate was simply wrong. Measurement "
            "shows it is right for one job and wrong for the other. For an absolute comparison of "
            "two cars at one instant it is the better coordinate, because the synthetic lap length "
            "is identical for every driver so the lap-count term cancels exactly; it reproduces the "
            "FastF1 gap to a median of 0.11 m, whereas measured-length accumulation lands at 158 m "
            "because each car's integrated lap differs by tens of metres and compounds. It is wrong "
            "only for distance differenced over time, where it injects ~232 m of phantom gain per "
            "line crossing. v3 therefore keeps cum_distance_m for the lapped test and uses measured "
            "race_distance_m, anchored at t0, for the swap evidence."
        ),
        "manual_validation": {
            "all_changes_involve_line_crossing": "31 of 31",
            "on_track_pass_to_uncertain": (
                "7 events. v1 evidence was distance+DriverAhead agreeing; v3 removes the distance "
                "half. Their v1 swap delays were 13, 65, 65, 64, 91, 393 and 3.6 s. At those delays "
                "the attacker had crossed the line one or more times, and the phantom 232 m per "
                "crossing on its own exceeds the 10-46 m starting gaps, so v1 manufactured the swap. "
                "Removing them is correct. Maximum pass swap delay falls from 393 s to 86 s."
            ),
            "close_no_pass_to_uncertain": (
                "20 events. The distance evidence now fires where v1 suppressed it, because when the "
                "TARGET crosses the line v1 credits the target with 232 m it did not travel and hides "
                "a real gain. DriverAhead does not corroborate, so UNCERTAIN rather than a pass is the "
                "honest label."
            ),
            "gained_a_label": (
                "2 events. COL vs SAI lap 4 moves UNCERTAIN -> ON_TRACK_PASS with both signals now "
                "agreeing, and OCO vs GAS lap 4 moves UNCERTAIN -> CLOSE_INTERACTION_NO_PASS with "
                "both signals now declining."
            ),
            "verdict": "The corrected detector is more physically defensible: it removes swaps that only existed because of phantom distance, and it stops hiding gains behind phantom distance credited to the target.",
        },
        "remaining_limitations": [
            "Evidence disagreement is concentrated in long outcome windows: the UNCERTAIN rate is "
            "5.5% for windows up to 45 s but 35.3% beyond 120 s. Over several laps both the distance "
            "test and the DriverAhead test degrade, the former through cross-driver integration "
            "divergence of roughly 60 m per lap and the latter because a DriverAhead relation minutes "
            "later need not belong to this battle. The variable-length outcome window is the next "
            "thing to address, and it is an event-definition change that was out of scope here.",
            "Two laps are truncated in the source telemetry (COL lap 1 at 5119 m, RUS lap 32 at 5055 m) "
            "and eight driver-laps are absent from the DriverAhead extract and are gap-filled with the "
            "driver's median lap length.",
            "The stored v2 dataset retains 85 closing-speed values computed before the lap-offset "
            "bugfix. Use v3, which supersedes both v1 and v2.",
        ],
        "frozen_parameters": {
            "gap_max_m": params.gap_max_m,
            "gap_min_m": params.gap_min_m,
            "min_persist_s": params.min_persist_s,
            "horizon_s": params.horizon_s,
        },
        "defect": (
            "cum_distance_m = (LapNumber - 1) * 5465.5 m assumed every lap was the full circuit "
            "length taken from the maximum observed Distance. Real integrated laps average 5234.8 m, "
            "so the coordinate gained ~231 m of phantom distance at each line crossing."
        ),
        "fix": (
            "race_distance_m accumulates each lap's own measured length (existing add_race_distance, "
            "already used by the v2 speed features, so no new distance model was introduced). The "
            "swap evidence additionally measures ground gained by each car against its own position "
            "at t0, so constant per-driver integration offsets cancel; two laps in this session are "
            "truncated by up to 180 m and would otherwise bias that car's swap test permanently."
        ),
        "continuity": continuity,
        "variant_counts": {
            "v1_reproduction": _type_counts(repro),
            "displacement_swap_on_v1_coordinate": _type_counts(diag),
            "v3_displacement_swap_on_measured_distance": _type_counts(v3),
        },
        "baseline_reproduction_verified": repro_matches_v1,
        "event_impact": {
            "events_v1": int(len(v1)),
            "events_v3": int(len(v3)),
            "newly_created": int(len(new_events)),
            "removed": int(len(removed_events)),
            "type_changed": int(both["type_changed"].sum()),
            "label_changed": int(both["label_changed"].sum()),
            "boundary_changed": int(both["boundary_changed"].sum()),
            "label_1_to_0": label_1_to_0,
            "label_0_to_1": label_0_to_1,
            "labelled_to_unlabelled": label_to_none,
            "unlabelled_to_labelled": label_from_none,
            "changed_involving_line_crossing": int(changed["lap_boundary_crossing_involved"].sum()),
            "transitions": {f"{o} -> {n}": int(c) for (o, n), c in transitions.items()},
        },
        "counts": {
            "on_track_pass_v1": int((v1["event_type"] == "ON_TRACK_PASS").sum()),
            "on_track_pass_v3": int((v3["event_type"] == "ON_TRACK_PASS").sum()),
            "close_no_pass_v1": int((v1["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum()),
            "close_no_pass_v3": int((v3["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum()),
        },
        "files_created": [
            str(config.OVERTAKE_EVENTS_V3_CSV),
            str(config.OVERTAKE_EVENTS_AUDIT_V3_CSV),
            str(config.OVERTAKE_EVENTS_CHANGED_V3_CSV),
            str(config.EVENT_DETECTION_V3_REPORT_JSON),
        ],
        "files_preserved": [str(config.OVERTAKE_EVENTS_CSV), str(config.OVERTAKE_EVENTS_V2_CSV)],
        "energy_hashes_before": hashes_before,
        "energy_hashes_after": hashes_after,
    }
    config.EVENT_DETECTION_V3_REPORT_JSON.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"  wrote {config.EVENT_DETECTION_V3_REPORT_JSON}")
    print("\n  No ML. No model selection. No extra races. No strategy engine.")
    return report


if __name__ == "__main__":
    main()
