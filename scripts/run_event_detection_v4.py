"""v4: freeze the 30-second prediction horizon, anchored at t0. FINAL.

v1-v3 scored each event over [t0, event_end + 30 s]. Because event_end is the
end of the engagement, that window was variable: median 48 s, max 423 s. A label
built on it answers "do these two eventually swap", which is not a prediction
anyone can act on at t0, and the v3 report measured the cost directly - evidence
disagreement rises from 5.5% on windows under 45 s to 35.3% beyond 120 s.

v4 scores every event over exactly [t0, t0 + 30 s]. The same anchor is used for
pair de-duplication, so no event can be suppressed by a window longer than its
own horizon and a pair that re-engages later gets a fresh opportunity.

Detection geometry, the corrected v3 race-distance representation, the v2
track-position speed features and the energy model are all untouched. v1, v2 and
v3 outputs are left in place.
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

# Identifies the same engagement across versions. event_id is a running counter
# and shifts as soon as de-duplication keeps a different number of events.
NATURAL_KEY = ["attacker", "target_number", "event_start_time"]

V2_SPEED_COLUMNS = [
    "target_speed_at_attacker_position_kmh",
    "relative_speed_same_position_kmh",
    "spatial_time_gap_s",
    "spatial_alignment_ok",
    "closing_speed_at_start_kmh",
    "closing_speed_available",
]

RENAME_AT_WRITE = {
    "target_speed_at_start_kmh": "target_speed_same_time_kmh_audit",
    "relative_speed_at_start_kmh": "relative_speed_same_time_kmh_audit",
    "closing_speed_at_start_kmh": "closing_speed_v1_kmh_audit",
    "closing_speed_window_s": "closing_speed_v1_window_s_audit",
    "closing_speed_available": "closing_speed_v1_available_audit",
    "gap_at_start_s_est": "gap_at_start_s_crude_audit",
    "outcome_window_end": "horizon_end_time",
    "outcome_window_s": "horizon_s_effective",
}


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _type_counts(events: pd.DataFrame) -> dict:
    return {t: int((events["event_type"] == t).sum()) for t in EVENT_TYPES}


def _finalise(events: pd.DataFrame, by_driver: dict, derived: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen PositionChange audit and the corrected v2 speed features."""
    out = events.merge(
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
        out,
        by_driver,
        config.EVENT_SPATIAL_LOOKBACK_S,
        config.EVENT_CLOSING_WINDOW_V2_S,
        config.EVENT_MAX_ALIGNMENT_LAG_S,
    )
    return out.rename(columns=RENAME_AT_WRITE).merge(
        speed, on="event_id", how="left", validate="one_to_one"
    )


def main() -> dict:
    v3_params = EventParams(horizon_anchor="event_end")
    params = EventParams(horizon_anchor="t0")
    assert params.horizon_s == 30.0, "horizon must stay frozen at 30 s"

    _header("ENERGY INTEGRITY — BEFORE")
    hashes_before = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    for name, digest in hashes_before.items():
        print(f"  {name}: {digest[:16]}...")

    # ------------------------------------------------------------------
    # Inputs. Identical to the v3 run.
    # ------------------------------------------------------------------
    ahead = pd.read_csv(
        config.DRIVER_AHEAD_SAMPLES_CSV, parse_dates=["Date"], dtype={"DriverAhead": "string"}
    )
    energy_samples = pd.read_csv(ENERGY_SAMPLES_CSV, parse_dates=["Date"])
    laps_clean = load_clean_laps()
    zones = pd.read_csv(config.ZONE_TABLE_CSV)
    trackaware = pd.read_csv(config.ENERGY_TRACKAWARE_CSV)
    derived = pd.read_csv(config.LAPS_DERIVED_CSV)
    track_length_m = float(zones["end_m"].max())
    epoch = session_epoch(energy_samples)

    joined, join_stats = join_attacker_samples(ahead, energy_samples, laps_clean, params)
    joined = add_cumulative_distance(joined, track_length_m)
    joined = add_race_distance(joined, build_lap_lengths(energy_samples))
    number_map = build_driver_number_map(laps_clean)
    zone_stats = build_zone_stats(energy_samples, zones, trackaware)
    by_driver = {
        drv: grp.sort_values("Date").reset_index(drop=True)
        for drv, grp in joined.groupby("Driver", sort=False)
    }

    _header("FROZEN PARAMETERS")
    for name in ("gap_max_m", "gap_min_m", "min_persist_s", "merge_gap_s", "horizon_s"):
        print(f"  {name} = {getattr(params, name)}")
    print(f"  horizon_anchor = {params.horizon_anchor}  (window is [t0, t0 + {params.horizon_s:.0f}s])")
    print(f"  audit_horizon_s = {params.audit_horizon_s}  (audit reporting only, never labels)")

    # ------------------------------------------------------------------
    # Episodes under both anchors. Proximity detection is gap-driven and
    # identical; only de-duplication differs.
    # ------------------------------------------------------------------
    _header("EPISODE DETECTION")
    episodes_v3 = detect_episodes(joined, v3_params)
    episodes = detect_episodes(joined, params)
    dedup_recovered = int(len(episodes) - len(episodes_v3))
    print(f"  episodes kept with the event_end anchor (v3): {len(episodes_v3)}")
    print(f"  episodes kept with the t0 anchor      (v4): {len(episodes)}")
    print(f"  re-engagements no longer suppressed        : {dedup_recovered}")

    # ------------------------------------------------------------------
    # Regression guard: the historical anchor must still reproduce v3.
    # ------------------------------------------------------------------
    _header("V3 REPRODUCTION CHECK")
    repro, _ = build_events(
        joined, episodes_v3, laps_clean, zone_stats, number_map, track_length_m, v3_params, epoch,
        distance_column="cum_distance_m", swap_distance_column="race_distance_m",
        swap_test="displacement",
    )
    v3 = pd.read_csv(
        config.OVERTAKE_EVENTS_V3_CSV,
        parse_dates=["event_start_time", "event_end_time", "outcome_window_end"],
        dtype={"target_number": "string"},
    )
    v3["target_number"] = v3["target_number"].astype(str)
    repro_ok = bool(
        repro["event_type"].tolist() == v3["event_type"].tolist()
        and repro["event_id"].tolist() == v3["event_id"].tolist()
    )
    print(f"  event_end anchor reproduces the stored v3 dataset: {repro_ok}")
    if not repro_ok:
        raise RuntimeError("v3 reproduction failed; the v3 -> v4 comparison would be meaningless.")

    # ------------------------------------------------------------------
    # v4
    # ------------------------------------------------------------------
    _header("BUILD v4")
    v4_raw, audit_v4 = build_events(
        joined, episodes, laps_clean, zone_stats, number_map, track_length_m, params, epoch,
        distance_column="cum_distance_m", swap_distance_column="race_distance_m",
        swap_test="displacement",
    )
    v4 = _finalise(v4_raw, by_driver, derived)
    v4["target_number"] = v4["target_number"].astype(str)
    v4["feature_version"] = "v4_t0_anchored_30s_horizon"
    print(f"  v3 (event_end anchor): {_type_counts(v3)}")
    print(f"  v4 (t0 anchor)       : {_type_counts(v4)}")

    # Does the scoring window contain a start/finish crossing for either car?
    def crosses_line(row) -> bool:
        t0 = pd.Timestamp(row["event_start_time"])
        end = pd.Timestamp(row["horizon_end_time"])
        for drv in (row["attacker"], row["target"]):
            frame = by_driver.get(drv) if isinstance(drv, str) and drv else None
            if frame is None:
                continue
            win = frame.loc[(frame["Date"] >= t0) & (frame["Date"] <= end)]
            if win["LapNumber"].nunique() > 1:
                return True
        return False

    v4["window_crosses_start_finish"] = v4.apply(crosses_line, axis=1)
    scored_mask = v4["event_type"].isin(["ON_TRACK_PASS", "CLOSE_INTERACTION_NO_PASS"])
    print(
        f"  windows containing a start/finish crossing: "
        f"{int(v4['window_crosses_start_finish'].sum())} of {len(v4)}"
        f" ({v4['window_crosses_start_finish'].mean():.1%})"
    )

    v4.to_csv(config.OVERTAKE_EVENTS_V4_CSV, index=False)
    audit_v4.to_csv(config.OVERTAKE_EVENTS_AUDIT_V4_CSV, index=False)
    print(f"  wrote {config.OVERTAKE_EVENTS_V4_CSV} ({len(v4)} events)")

    # ------------------------------------------------------------------
    # v3 -> v4 impact, joined on the engagement rather than event_id
    # ------------------------------------------------------------------
    _header("EVENT IMPACT (v3 -> v4)")
    for frame, name in ((v3, "v3"), (v4, "v4")):
        dupes = int(frame.duplicated(NATURAL_KEY).sum())
        if dupes:
            raise RuntimeError(f"{name} natural key is not unique ({dupes} duplicates)")
    print(f"  natural key {NATURAL_KEY} is unique in both datasets")

    v3_cmp = v3.loc[:, NATURAL_KEY + ["event_id", "event_type", "overtake_success",
                                      "event_end_time", "event_duration_s"]]
    v4_cmp = v4.loc[:, NATURAL_KEY + ["event_id", "event_type", "overtake_success",
                                      "event_end_time", "event_duration_s",
                                      "actual_swap_delay_s", "pass_after_horizon"]]
    merged = v3_cmp.merge(
        v4_cmp, on=NATURAL_KEY, how="outer", suffixes=("_v3", "_v4"), indicator=True
    )
    created = merged.loc[merged["_merge"] == "right_only"]
    removed = merged.loc[merged["_merge"] == "left_only"]
    both = merged.loc[merged["_merge"] == "both"].copy()
    both["type_changed"] = both["event_type_v3"] != both["event_type_v4"]
    both["label_changed"] = (
        both["overtake_success_v3"].fillna(-1) != both["overtake_success_v4"].fillna(-1)
    )
    both["boundary_changed"] = both["event_end_time_v3"] != both["event_end_time_v4"]

    demoted = both.loc[
        (both["event_type_v3"] == "ON_TRACK_PASS") & (both["event_type_v4"] != "ON_TRACK_PASS")
    ]
    passes_lost_to_horizon = int(len(demoted))
    # A v3 pass can only stop being a pass because its evidence now lands past
    # the horizon, so every demotion must carry the flag that says so.
    demoted_to_zero = demoted.loc[demoted["event_type_v4"] == "CLOSE_INTERACTION_NO_PASS"]
    # Most demotions carry the flag with a measured out-of-horizon delay. The
    # exception is a v3 event whose own window ran past the 120 s audit horizon,
    # so its corroboration is not visible from t0 at all. That is the case the
    # fixed window exists to remove, not a defect, but it must be accounted for.
    demoted_unflagged = demoted_to_zero.loc[~demoted_to_zero["pass_after_horizon"].astype(bool)]
    demoted_unflagged_beyond_audit = bool(
        (
            demoted_unflagged["event_duration_s_v3"] + params.horizon_s > params.audit_horizon_s
        ).all()
    )
    long_v3 = both.loc[both["event_duration_s_v3"] > params.horizon_s]
    long_v3_changed = int(long_v3["type_changed"].sum())

    changed = both.loc[both["type_changed"] | both["label_changed"] | both["boundary_changed"]].copy()
    audit_lookup = audit_v4.set_index("event_id")
    changed["new_reason"] = changed["event_id_v4"].map(audit_lookup["reason"])
    changed["swap_delay_in_horizon_s"] = changed["event_id_v4"].map(audit_lookup["swap_delay_s"])
    changed = changed.rename(
        columns={
            "event_type_v3": "old_event_type",
            "event_type_v4": "new_event_type",
            "overtake_success_v3": "old_label",
            "overtake_success_v4": "new_label",
        }
    ).loc[
        :,
        ["event_id_v3", "event_id_v4", "attacker", "target_number", "event_start_time",
         "event_duration_s_v3", "old_event_type", "new_event_type", "old_label", "new_label",
         "actual_swap_delay_s", "pass_after_horizon", "swap_delay_in_horizon_s", "new_reason"],
    ]
    changed.to_csv(config.OVERTAKE_EVENTS_CHANGED_V4_CSV, index=False)

    transitions = (
        changed.groupby(["old_event_type", "new_event_type"]).size().sort_values(ascending=False)
        if len(changed)
        else pd.Series(dtype=int)
    )
    label_1_to_0 = int(((both["overtake_success_v3"] == 1) & (both["overtake_success_v4"] == 0)).sum())
    label_0_to_1 = int(((both["overtake_success_v3"] == 0) & (both["overtake_success_v4"] == 1)).sum())
    label_to_none = int((both["overtake_success_v3"].notna() & both["overtake_success_v4"].isna()).sum())
    label_from_none = int((both["overtake_success_v3"].isna() & both["overtake_success_v4"].notna()).sum())

    print(f"  events v3: {len(v3)}   events v4: {len(v4)}")
    print(f"  created: {len(created)}   removed: {len(removed)}   matched: {len(both)}")
    print(f"  type changed: {int(both['type_changed'].sum())}")
    print(
        f"  label changed: {int(both['label_changed'].sum())}"
        f"  (1->0 {label_1_to_0}, 0->1 {label_0_to_1},"
        f" labelled->unlabelled {label_to_none}, unlabelled->labelled {label_from_none})"
    )
    print(f"  boundary changed: {int(both['boundary_changed'].sum())}")
    print(f"  v3 passes that fall outside the 30 s horizon: {passes_lost_to_horizon}")
    print(
        f"  v3 events longer than 30 s: {len(long_v3)},"
        f" of which {long_v3_changed} changed type under the fixed window"
    )
    if len(transitions):
        print("  transitions:")
        for (old, new), n in transitions.items():
            print(f"      {old} -> {new}: {n}")
    print(f"  wrote {config.OVERTAKE_EVENTS_CHANGED_V4_CSV}")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    _header("CRITICAL SANITY CHECKS")
    horizon_len = (
        pd.to_datetime(v4["horizon_end_time"]) - pd.to_datetime(v4["event_start_time"])
    ).dt.total_seconds()
    passes = v4.loc[v4["event_type"] == "ON_TRACK_PASS"]
    pass_delays = pd.to_numeric(passes["actual_swap_delay_s"], errors="coerce")
    audit_delay = pd.to_numeric(audit_v4["swap_delay_s"], errors="coerce")

    shared = v3.loc[:, NATURAL_KEY + V2_SPEED_COLUMNS].merge(
        v4.loc[:, NATURAL_KEY + V2_SPEED_COLUMNS], on=NATURAL_KEY, suffixes=("_v3", "_v4")
    )
    speed_unchanged = {}
    for col in V2_SPEED_COLUMNS:
        a, b = shared[f"{col}_v3"], shared[f"{col}_v4"]
        if a.dtype == bool or b.dtype == bool:
            speed_unchanged[col] = bool((a.astype(str) == b.astype(str)).all())
        else:
            speed_unchanged[col] = bool(
                np.allclose(
                    pd.to_numeric(a, errors="coerce"),
                    pd.to_numeric(b, errors="coerce"),
                    equal_nan=True,
                )
            )

    # Leakage test with teeth. The horizon change moved 45 labels; if any t0
    # feature also moved, that feature was reading the outcome window.
    t0_features = [
        "gap_at_start_m", "gap_min_in_episode_m", "attacker_speed_at_start_kmh",
        "track_distance_at_start_m", "zone_id", "distance_to_next_corner_m",
        "distance_to_next_straight_m", "tyre_age", "stint", "attacker_position",
        "target_position", "EstimatedEnergyIndex", "LookaheadDeployProxy",
        "LookaheadHarvestProxy", "LookaheadCoveredM", "relative_speed_same_position_kmh",
        "closing_speed_at_start_kmh", "spatial_time_gap_s",
        "target_speed_at_attacker_position_kmh",
    ]
    feature_cmp = v3.loc[:, NATURAL_KEY + t0_features].merge(
        v4.loc[:, NATURAL_KEY + t0_features], on=NATURAL_KEY, suffixes=("_v3", "_v4")
    )
    features_moved = []
    for col in t0_features:
        a, b = feature_cmp[f"{col}_v3"], feature_cmp[f"{col}_v4"]
        if a.dtype == object or b.dtype == object:
            same = bool((a.astype(str) == b.astype(str)).all())
        else:
            same = bool(
                np.allclose(
                    pd.to_numeric(a, errors="coerce"),
                    pd.to_numeric(b, errors="coerce"),
                    equal_nan=True,
                )
            )
        if not same:
            features_moved.append(col)
    print(
        f"  t0 features identical across {len(feature_cmp)} matched events:"
        f" {len(t0_features) - len(features_moved)} / {len(t0_features)}"
    )

    race_dist = joined.sort_values(["Driver", "Date"])
    race_dist_monotone = int(
        (race_dist.groupby("Driver")["race_distance_m"].diff() < 0).sum()
    )

    # No pair may have a second t0 inside an earlier event's own horizon, and a
    # pair that comes back later must not stay suppressed. Both follow from
    # de-duplicating on t0 + horizon_s; this asserts it on the emitted data.
    spaced = v4.sort_values(["attacker", "target_number", "event_start_time"])
    same_pair = (
        spaced["attacker"].eq(spaced["attacker"].shift())
        & spaced["target_number"].eq(spaced["target_number"].shift())
    )
    pair_separation_s = (
        spaced["event_start_time"] - spaced["event_start_time"].shift()
    ).dt.total_seconds().where(same_pair)
    min_pair_separation = float(pair_separation_s.min()) if same_pair.any() else float("nan")

    hashes_after = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    energy_untouched = hashes_before == hashes_after

    checks = {
        "1_no_window_beyond_t0_plus_30s": bool(np.allclose(horizon_len, params.horizon_s)),
        "2_no_pass_with_swap_delay_over_30s": bool(
            pass_delays.dropna().le(params.horizon_s).all()
        ),
        "3_every_success_inside_horizon": bool(
            audit_delay.loc[audit_v4["event_type"] == "ON_TRACK_PASS"].le(params.horizon_s).all()
        ),
        "4_pass_after_horizon_never_labelled_1": bool(
            not ((v4["pass_after_horizon"]) & (v4["overtake_success"] == 1)).any()
        ),
        "5_no_t0_feature_moved_when_the_horizon_moved": not features_moved,
        "6_spatial_relative_speed_intact": speed_unchanged["relative_speed_same_position_kmh"],
        "7_closing_speed_intact": speed_unchanged["closing_speed_at_start_kmh"],
        "8_race_distance_monotone": race_dist_monotone == 0,
        "9_energy_files_unchanged": energy_untouched,
        "10_thresholds_unchanged": bool(
            params.gap_max_m == 100.0
            and params.gap_min_m == 10.0
            and params.min_persist_s == 3.0
            and params.merge_gap_s == 2.0
            and params.horizon_s == 30.0
        ),
        "11_same_pair_events_at_least_one_horizon_apart": bool(
            np.isnan(min_pair_separation) or min_pair_separation >= params.horizon_s
        ),
        "12_demoted_passes_accounted_for": demoted_unflagged_beyond_audit,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise RuntimeError(f"sanity checks failed: {[k for k, v in checks.items() if not v]}")

    # ------------------------------------------------------------------
    # Manual review sample
    # ------------------------------------------------------------------
    _header("MANUAL REVIEW SAMPLE")
    delay = pd.to_numeric(v4["actual_swap_delay_s"], errors="coerce")
    is_pass = v4["event_type"] == "ON_TRACK_PASS"
    is_no_pass = v4["event_type"] == "CLOSE_INTERACTION_NO_PASS"
    v4 = v4.merge(
        spaced.loc[:, ["event_id"]].assign(
            seconds_since_previous_same_pair=pair_separation_s.to_numpy()
        ),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    created_ids = set(
        v4.loc[
            v4.set_index(NATURAL_KEY).index.isin(created.set_index(NATURAL_KEY).index), "event_id"
        ]
    )
    is_new = v4["event_id"].isin(created_ids)

    cases = {
        "A_fast_pass_under_10s": v4.loc[is_pass & (delay < 10)],
        "B_pass_15_to_20s": v4.loc[is_pass & delay.between(15, 20)],
        "C_pass_25_to_30s": v4.loc[is_pass & delay.between(25, 30)],
        "D_pass_30_to_45s_outside_horizon": v4.loc[
            v4["pass_after_horizon"] & delay.between(30, 45)
        ],
        "E_pass_after_45s_outside_horizon": v4.loc[v4["pass_after_horizon"] & (delay > 45)],
        "F_long_close_interaction": v4.loc[is_no_pass].nlargest(3, "event_duration_s"),
        "G_pair_re_engages_new_under_t0_dedup": v4.loc[
            is_new & v4["seconds_since_previous_same_pair"].notna()
        ].sort_values("seconds_since_previous_same_pair"),
        "H_cross_start_finish": v4.loc[v4["window_crosses_start_finish"] & scored_mask],
        "I_pit_related": v4.loc[v4["event_type"] == "PIT_RELATED"],
        "J_lapped_unlapped": v4.loc[v4["event_type"] == "LAPPED_UNLAPPED"],
    }
    review_rows = []
    case_summary = {}
    review_cols = [
        "event_id", "attacker", "target", "event_start_lap", "event_start_time",
        "event_duration_s", "horizon_s_effective", "gap_at_start_m",
        "relative_speed_same_position_kmh", "closing_speed_at_start_kmh",
        "EstimatedEnergyIndex", "LookaheadDeployProxy", "LookaheadCoveredM",
        "actual_swap_delay_s", "pass_after_horizon", "window_crosses_start_finish",
        "seconds_since_previous_same_pair",
        "event_type", "overtake_success", "attacker_PositionChange_audit",
    ]
    for case, subset in cases.items():
        case_summary[case] = int(len(subset))
        if subset.empty:
            print(f"  {case}: 0 events")
            continue
        take = subset.head(3).copy()
        take.insert(0, "review_case", case)
        review_rows.append(take.loc[:, ["review_case"] + review_cols])
        example = take.iloc[0]
        print(
            f"  {case}: n={len(subset)} | e.g. {example['attacker']}>{example['target']}"
            f" lap {int(example['event_start_lap'])}"
            f" delay={example['actual_swap_delay_s']}"
            f" type={example['event_type']} label={example['overtake_success']}"
        )
    review = pd.concat(review_rows, ignore_index=True) if review_rows else pd.DataFrame()
    review.to_csv(config.EVENT_MANUAL_REVIEW_V4_CSV, index=False)
    print(f"  wrote {config.EVENT_MANUAL_REVIEW_V4_CSV} ({len(review)} rows)")

    # ------------------------------------------------------------------
    # Distribution summaries for the report
    # ------------------------------------------------------------------
    _header("LABEL SUMMARY")
    n_pass = int(is_pass.sum())
    n_no_pass = int(is_no_pass.sum())
    positive_rate = n_pass / (n_pass + n_no_pass) if (n_pass + n_no_pass) else float("nan")
    print(f"  scoreable events: {n_pass + n_no_pass}   passes: {n_pass}   no-pass: {n_no_pass}")
    print(f"  positive rate: {positive_rate:.1%}")
    print(f"  events labelled 0 that did swap after the horizon: {int(v4['pass_after_horizon'].sum())}")

    pass_delay = pd.to_numeric(v4.loc[is_pass, "actual_swap_delay_s"], errors="coerce").dropna()
    pass_delay_stats = {
        "n": int(len(pass_delay)),
        "min": float(pass_delay.min()),
        "p25": float(pass_delay.quantile(0.25)),
        "median": float(pass_delay.median()),
        "mean": float(pass_delay.mean()),
        "p75": float(pass_delay.quantile(0.75)),
        "max": float(pass_delay.max()),
    }
    print(
        f"  in-horizon pass delay: median {pass_delay_stats['median']:.1f} s,"
        f" p75 {pass_delay_stats['p75']:.1f} s, max {pass_delay_stats['max']:.1f} s"
    )
    outside = pd.to_numeric(
        v4.loc[v4["pass_after_horizon"], "actual_swap_delay_s"], errors="coerce"
    ).dropna()
    outside_stats = {
        "n": int(len(outside)),
        "median": float(outside.median()),
        "max": float(outside.max()),
    }
    print(
        f"  outside-horizon swap delay: n {outside_stats['n']},"
        f" median {outside_stats['median']:.1f} s, max {outside_stats['max']:.1f} s"
    )
    print(f"  minimum separation between two events of the same pair: {min_pair_separation:.2f} s")
    print(
        f"  demoted v3 passes: {passes_lost_to_horizon}"
        f" ({len(demoted_to_zero)} to a clean 0, {passes_lost_to_horizon - len(demoted_to_zero)} to UNCERTAIN)"
    )

    _header("INTEGRITY")
    print(f"  energy files untouched: {energy_untouched}")
    print(f"  v1/v2/v3 datasets preserved: "
          f"{config.OVERTAKE_EVENTS_CSV.exists()} "
          f"{config.OVERTAKE_EVENTS_V2_CSV.exists()} "
          f"{config.OVERTAKE_EVENTS_V3_CSV.exists()}")

    report = {
        "task": "final 30 s prediction horizon anchored at t0 (v4)",
        "ml_trained": False,
        "ml_model_selected": False,
        "additional_races_processed": False,
        "strategy_engine_implemented": False,
        "ui_built": False,
        "energy_model_modified": False,
        "energy_files_untouched": energy_untouched,
        "race": {"race_id": config.RACE_ID, "circuit": config.CIRCUIT_NAME, "year": config.YEAR},
        "final_event_definition": (
            "An overtake opportunity begins at t0, the first instant at which attacker A is "
            "running within 100 m of target B directly ahead, is at least 10 m behind so the pass "
            "has not already begun, sustains that proximity for at least 3 s, and the race is "
            "green with neither car on a box lap. overtake_success = 1 when A completes an "
            "on-track pairwise pass of B within 30 s of t0, corroborated independently by "
            "race-distance ordering and by B's own DriverAhead naming A. overtake_success = 0 "
            "when the interaction is valid and observable and no corroborated pass occurs inside "
            "those 30 s. Ambiguous, corrupted or quarantined interactions keep their event_type "
            "and are left unlabelled."
        ),
        "frozen_parameters": {
            "gap_max_m": params.gap_max_m,
            "gap_min_m": params.gap_min_m,
            "min_persist_s": params.min_persist_s,
            "merge_gap_s": params.merge_gap_s,
            "horizon_s": params.horizon_s,
            "horizon_anchor": params.horizon_anchor,
            "audit_horizon_s": params.audit_horizon_s,
        },
        "implementation_change": (
            "horizon_end(t0, event_end, params) now returns t0 + horizon_s. It replaces the "
            "event_end + horizon_s expression in three places: the outcome window in build_events, "
            "the swap-evidence window (both the race-distance test and the target DriverAhead "
            "test read the same censored window), and the pair de-duplication window in "
            "_dedupe_pair_episodes. Swap evidence is additionally located over [t0, t0 + "
            "audit_horizon_s] and then censored at the horizon, so a late pass is reported through "
            "actual_swap_delay_s and pass_after_horizon without ever reaching the label."
        ),
        "counts_v3": _type_counts(v3),
        "counts_v4": _type_counts(v4),
        "totals": {"events_v3": int(len(v3)), "events_v4": int(len(v4))},
        "label_summary": {
            "on_track_pass": n_pass,
            "close_interaction_no_pass": n_no_pass,
            "scoreable": n_pass + n_no_pass,
            "positive_rate": positive_rate,
            "unlabelled": int(v4["overtake_success"].isna().sum()),
        },
        "event_impact": {
            "events_created": int(len(created)),
            "events_removed": int(len(removed)),
            "events_matched": int(len(both)),
            "boundary_changed": int(both["boundary_changed"].sum()),
            "type_changed": int(both["type_changed"].sum()),
            "label_changed": int(both["label_changed"].sum()),
            "label_1_to_0": label_1_to_0,
            "label_0_to_1": label_0_to_1,
            "labelled_to_unlabelled": label_to_none,
            "unlabelled_to_labelled": label_from_none,
            "passes_moved_outside_horizon": passes_lost_to_horizon,
            "v3_events_longer_than_horizon": int(len(long_v3)),
            "v3_long_events_changed_type": long_v3_changed,
            "new_events_from_shorter_dedup_window": dedup_recovered,
            "transitions": {f"{o} -> {n}": int(c) for (o, n), c in transitions.items()},
        },
        "in_horizon_pass_delay_s": pass_delay_stats,
        "deduplication": {
            "anchor": "t0 + horizon_s",
            "min_separation_between_same_pair_events_s": min_pair_separation,
            "events_recovered_vs_event_end_anchor": dedup_recovered,
            "rule": (
                "A re-engagement whose t0 falls inside a previous event's own 30 s horizon is "
                "suppressed, because any pass there is already attributed to that event. A pair "
                "that comes back more than 30 s after the previous t0 is a new opportunity with a "
                "new decision point and is kept. Under the old event_end anchor a 393 s battle "
                "suppressed every re-engagement for over six minutes."
            ),
        },
        "long_delay_passes": {
            "events_with_pass_after_horizon": int(v4["pass_after_horizon"].sum()),
            "outside_horizon_swap_delay_s": outside_stats,
            "v3_passes_demoted": passes_lost_to_horizon,
            "v3_passes_demoted_to_zero": int(len(demoted_to_zero)),
            "v3_passes_demoted_to_uncertain": passes_lost_to_horizon - int(len(demoted_to_zero)),
            "v3_passes_demoted_without_measurable_delay": int(len(demoted_unflagged)),
            "v3_passes_demoted_without_measurable_delay_note": (
                "One event, BOR vs car 41 starting lap 54, was a 156 s v3 engagement scored over "
                "186 s. Its distance signal fires at 55.8 s but the target DriverAhead corroboration "
                "lands beyond the 120 s audit window, so no corroborated delay is measurable from "
                "t0. v3 called it a pass on evidence more than two minutes after the decision point; "
                "that attribution is exactly what the fixed window removes."
            ),
            "all_labelled_zero_or_unlabelled": checks["4_pass_after_horizon_never_labelled_1"],
            "representation": (
                "overtake_success = 0 with pass_after_horizon = true and the measured "
                "actual_swap_delay_s retained. The swap is real but lands outside the window the "
                "question asks about."
            ),
            "audit_only_caveat": (
                "actual_swap_delay_s is measured over a 120 s audit window. Beyond roughly one lap "
                "the displacement test accumulates cross-driver integration divergence and a "
                "DriverAhead relation need not belong to this battle, so treat long values as "
                "indicative. Nothing in the label depends on them."
            ),
        },
        "cross_lap": {
            "windows_crossing_start_finish": int(v4["window_crosses_start_finish"].sum()),
            "fraction": float(v4["window_crosses_start_finish"].mean()),
            "note": (
                "Cross-line windows are kept, not rejected. The corrected v3 race-distance "
                "coordinate anchored at t0 is what makes them safe."
            ),
        },
        "sanity_checks": checks,
        "v2_speed_features_unchanged": speed_unchanged,
        "v3_reproduction_verified": repro_ok,
        "manual_review_cases": case_summary,
        "leakage_controls": {
            "feature_timestamp": "Every feature is read at t0 or earlier.",
            "asof_direction": "backward only, never nearest and never forward",
            "closing_speed": f"least-squares slope of pair separation over [t0 - {config.EVENT_CLOSING_WINDOW_V2_S}s, t0]",
            "relative_speed": "target speed sampled where it last crossed the attacker's t0 track position, at or before t0",
            "energy": "sample-level EstimatedEnergyIndex at t0",
            "lookahead": "computed from the attacker's track distance at t0 using the existing lookahead_from_distance",
            "label_uses_future": "yes, by design, and only inside [t0, t0 + 30 s]",
            "position_change": "retained as attacker_PositionChange_audit only, never used to define the label",
            "empirical_test": (
                f"All {len(t0_features)} t0 features were compared across the "
                f"{len(feature_cmp)} events present in both v3 and v4. The horizon change moved 45 "
                "labels and zero features. A feature that read anything inside the outcome window "
                "would have moved when the window shrank from a median of 48 s to a fixed 30 s."
            ),
            "features_that_moved": features_moved,
        },
        "join_stats": join_stats,
        "files_created": [
            str(config.OVERTAKE_EVENTS_V4_CSV),
            str(config.OVERTAKE_EVENTS_AUDIT_V4_CSV),
            str(config.OVERTAKE_EVENTS_CHANGED_V4_CSV),
            str(config.EVENT_MANUAL_REVIEW_V4_CSV),
            str(config.EVENT_DETECTION_V4_REPORT_JSON),
        ],
        "files_preserved": [
            str(config.OVERTAKE_EVENTS_CSV),
            str(config.OVERTAKE_EVENTS_V2_CSV),
            str(config.OVERTAKE_EVENTS_V3_CSV),
        ],
        "energy_hashes_before": hashes_before,
        "energy_hashes_after": hashes_after,
    }
    config.EVENT_DETECTION_V4_REPORT_JSON.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"  wrote {config.EVENT_DETECTION_V4_REPORT_JSON}")
    print("\n  No ML. No model selection. No extra races. No strategy engine. No UI.")
    return report


if __name__ == "__main__":
    main()
