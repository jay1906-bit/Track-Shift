"""Phases 1-7: Australia overtake-event dataset and kill-gate report.

Reads the frozen energy/ahead outputs, builds pairwise proximity episodes,
labels pass / no-pass, and stops. No ML, no model choice, no strategy engine,
no extra races. Energy files are hashed before and after to prove they are
untouched.
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
    GAP_UNITS_NOTE,
    EventParams,
    add_cumulative_distance,
    build_driver_number_map,
    build_events,
    build_zone_stats,
    detect_episodes,
    join_attacker_samples,
    session_epoch,
)
from trackshift.session_io import load_clean_laps  # noqa: E402
from trackshift.validate import file_sha256  # noqa: E402
from trackshift.zones import lap_start_lookahead  # noqa: E402

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

# Kill-gate criteria, fixed before looking at the labels.
MIN_PASSES_FOR_ML = 30
MIN_DISTINCT_ATTACKERS = 8
MIN_DISTINCT_ZONES = 3


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _describe(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def main() -> dict:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    params = EventParams()

    # ------------------------------------------------------------------
    # Phase 1 — freeze and inspect
    # ------------------------------------------------------------------
    _header("PHASE 1 — FREEZE EXISTING SYSTEM")
    hashes_before = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    for name, digest in hashes_before.items():
        print(f"  {name}: {digest[:16]}...")
    if not ENERGY_SAMPLES_CSV.exists():
        raise FileNotFoundError(
            f"{ENERGY_SAMPLES_CSV} is missing. Regenerate it with scripts/run_energy_system.py."
        )

    ahead = pd.read_csv(
        config.DRIVER_AHEAD_SAMPLES_CSV, parse_dates=["Date"], dtype={"DriverAhead": "string"}
    )
    energy_samples = pd.read_csv(ENERGY_SAMPLES_CSV, parse_dates=["Date"])
    laps_clean = load_clean_laps()
    zones = pd.read_csv(config.ZONE_TABLE_CSV)
    trackaware = pd.read_csv(config.ENERGY_TRACKAWARE_CSV)

    track_length_m = float(zones["end_m"].max())
    epoch = session_epoch(energy_samples)
    median_dt = float(pd.to_numeric(energy_samples["DeltaTimeSeconds"], errors="coerce").median())
    print(f"ahead samples={len(ahead)}  energy samples={len(energy_samples)}  laps={len(laps_clean)}")
    print(f"track length={track_length_m:.1f} m  median dt={median_dt:.3f} s (~{1/median_dt:.2f} Hz)")
    print(GAP_UNITS_NOTE)

    # ------------------------------------------------------------------
    # Phase 2 — joined sample table
    # ------------------------------------------------------------------
    _header("PHASE 2 — JOIN AHEAD + ENERGY + LAP SPINE")
    joined, join_stats = join_attacker_samples(ahead, energy_samples, laps_clean, params)
    joined = add_cumulative_distance(joined, track_length_m)
    for k, v in join_stats.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 3 — target identity
    # ------------------------------------------------------------------
    _header("PHASE 3 — TARGET DRIVER MAPPING")
    number_map = build_driver_number_map(laps_clean)
    seen = sorted({s for s in joined["DriverAhead"].dropna().unique() if s})
    unmapped = [s for s in seen if s not in number_map]
    print(f"  car numbers in DriverAhead: {len(seen)}  mapped: {len(seen) - len(unmapped)}")
    if unmapped:
        raise RuntimeError(f"Unmapped car numbers in DriverAhead: {unmapped}")
    print(f"  mapping: {number_map}")

    # ------------------------------------------------------------------
    # Phase 5 — zone baselines for event-time lookahead
    # ------------------------------------------------------------------
    _header("PHASE 5 — EVENT-TIME LOOKAHEAD BASELINES")
    zone_stats = build_zone_stats(energy_samples, zones, trackaware)
    check = trackaware.loc[
        :, ["Driver", "LapNumber", "LookaheadDeployProxy", "LookaheadHarvestProxy", "LookaheadCoveredM"]
    ].merge(lap_start_lookahead(energy_samples, zone_stats), on=["Driver", "LapNumber"], suffixes=("_stored", "_recomputed"))
    max_diffs = {}
    for col in ["LookaheadDeployProxy", "LookaheadHarvestProxy", "LookaheadCoveredM"]:
        diff = float((check[f"{col}_stored"] - check[f"{col}_recomputed"]).abs().max())
        max_diffs[col] = diff
        print(f"  {col}: max |stored - recomputed| = {diff:.3e}")
        if diff > 1e-6:
            raise RuntimeError(f"Zone baselines do not reproduce the frozen lookahead for {col}.")
    print("  Zone baselines reproduce the frozen Approach C lookahead. Reusing them at event time.")

    # ------------------------------------------------------------------
    # Phase 6 — proximity episodes
    # ------------------------------------------------------------------
    _header("PHASE 6 — PROXIMITY EPISODE DETECTION")
    print(f"  gap band=[{params.gap_min_m}, {params.gap_max_m}] m  min_persist_s={params.min_persist_s}  horizon_s={params.horizon_s}")
    print(f"  merge_gap_s={params.merge_gap_s}  closing_window_s={params.closing_window_s}")
    print(f"  excluded laps (standing start): {list(params.exclude_lap_numbers)}")
    n_close_samples = int(
        (
            joined["DriverAhead"].astype(str).str.strip().ne("")
            & (joined["gap_m"] <= params.gap_max_m)
            & (joined["gap_m"] >= params.gap_min_m)
            & ~joined["LapNumber"].isin(list(params.exclude_lap_numbers))
        ).sum()
    )
    episodes = detect_episodes(joined, params)
    print(f"  close samples (pre-collapse): {n_close_samples}")
    print(f"  episodes after collapsing and min-duration: {len(episodes)}")
    if len(episodes):
        print(f"  collapse ratio: {n_close_samples / len(episodes):.1f} samples per episode")

    # ------------------------------------------------------------------
    # Phase 7 — outcomes and labels
    # ------------------------------------------------------------------
    _header("PHASE 7 — PAIRWISE OUTCOME AND EVENT LABELS")
    events, audit = build_events(
        joined, episodes, laps_clean, zone_stats, number_map, track_length_m, params, epoch
    )

    derived = pd.read_csv(config.LAPS_DERIVED_CSV)
    events = events.merge(
        derived.loc[:, ["Driver", "LapNumber", "PositionChange"]].rename(
            columns={"Driver": "attacker", "LapNumber": "event_start_lap", "PositionChange": "attacker_PositionChange_audit"}
        ),
        on=["attacker", "event_start_lap"],
        how="left",
    )

    events.to_csv(config.OVERTAKE_EVENTS_CSV, index=False)
    audit.to_csv(config.OVERTAKE_EVENTS_AUDIT_CSV, index=False)
    print(f"  wrote {config.OVERTAKE_EVENTS_CSV} ({len(events)} events)")
    print(f"  wrote {config.OVERTAKE_EVENTS_AUDIT_CSV}")

    type_counts = {t: int((events["event_type"] == t).sum()) for t in EVENT_TYPES}
    for t in EVENT_TYPES:
        print(f"    {t}: {type_counts[t]}")

    labeled = events.loc[events["overtake_success"].notna()]
    n_pos = int((events["overtake_success"] == 1).sum())
    n_neg = int((events["overtake_success"] == 0).sum())

    # race phase thirds by lap number
    max_lap = int(laps_clean["LapNumber"].max())
    bins = [0, max_lap / 3, 2 * max_lap / 3, max_lap + 1]
    phase = pd.cut(events["event_start_lap"], bins=bins, labels=["early", "middle", "late"])

    passes = events.loc[events["event_type"] == "ON_TRACK_PASS"]
    n_attackers_pass = int(passes["attacker"].nunique())
    n_zones_pass = int(passes["zone_id"].nunique())

    case_a = (
        n_pos >= MIN_PASSES_FOR_ML
        and n_attackers_pass >= MIN_DISTINCT_ATTACKERS
        and n_zones_pass >= MIN_DISTINCT_ZONES
    )
    kill_gate = "CASE_A" if case_a else "CASE_B"

    # ------------------------------------------------------------------
    # Manual review examples
    # ------------------------------------------------------------------
    review_cols = [
        "event_id",
        "attacker",
        "target",
        "event_start_time",
        "event_start_lap",
        "event_duration_s",
        "gap_at_start_m",
        "gap_at_start_s_est",
        "closing_speed_at_start_kmh",
        "relative_speed_at_start_kmh",
        "zone_id",
        "EstimatedEnergyIndex",
        "event_type",
        "overtake_success",
    ]
    examples = []
    for t in EVENT_TYPES:
        subset = events.loc[events["event_type"] == t].head(4)
        if not subset.empty:
            examples.append(subset.loc[:, [c for c in review_cols if c in subset.columns]])
    review = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame(columns=review_cols)
    review = review.merge(audit.loc[:, ["event_id", "reason", "swap_delay_s"]], on="event_id", how="left")
    review.to_csv(config.EVENT_MANUAL_REVIEW_CSV, index=False)
    print(f"  wrote {config.EVENT_MANUAL_REVIEW_CSV} ({len(review)} examples)")

    # ------------------------------------------------------------------
    # Freeze verification
    # ------------------------------------------------------------------
    hashes_after = {name: file_sha256(path) for name, path in ENERGY_PATHS.items()}
    energy_untouched = hashes_before == hashes_after

    report = {
        "phase": "1-7 australia overtake event detection",
        "ml_trained": False,
        "ml_model_selected": False,
        "strategy_engine_implemented": False,
        "attack_save_defend_implemented": False,
        "additional_races_processed": False,
        "energy_model_modified": False,
        "energy_files_untouched": energy_untouched,
        "gap_units_note": GAP_UNITS_NOTE,
        "parameters": params.to_dict(),
        "parameter_justification": {
            "sampling_median_dt_s": median_dt,
            "sampling_hz": 1.0 / median_dt if median_dt else None,
            "gap_max_m": "100 m is roughly 1.2-1.8 s at Albert Park racing speed; chosen from speed and track scale, not from the outcome.",
            "gap_min_m": "10 m is just under two car lengths. Below it the speed-integrated ordering is inside measurement noise and the pass is already in progress, so t0 would leak the outcome.",
            "pair_deduplication": "A pair's re-engagement is dropped while the previous event's outcome window is still open, since any pass there is already attributed to that event.",
            "min_persist_s": f"3 s is about {3.0/median_dt:.0f} samples at the observed rate, so a battle must be sustained rather than a single blip.",
            "horizon_s": "30 s is about a third of an ~86 s green lap. The outcome window runs from t0 to the end of the engagement plus horizon_s, so a pass late in a long battle is still attributed to that battle. Features stay fixed at t0.",
            "engagement_unit": "One event per continuous engagement (pair within gap_max, bridging sub-gap_min phases), so an oscillating battle is not emitted several times.",
            "merge_gap_s": "2 s bridges short telemetry dropouts so one battle is not split into several events.",
            "closing_window_s": "2 s pre-t0 window; uses only samples at or before t0.",
            "tuned_against_labels": False,
        },
        "track_length_m": track_length_m,
        "join_stats": join_stats,
        "lookahead_reproduction_max_abs_diff": max_diffs,
        "driver_number_map": number_map,
        "totals": {
            "sample_rows_examined": int(len(ahead)),
            "close_samples_before_collapse": n_close_samples,
            "proximity_episodes": int(len(events)),
            "samples_per_episode": float(n_close_samples / len(events)) if len(events) else None,
        },
        "event_type_counts": type_counts,
        "class_balance": {
            "positive_on_track_pass": n_pos,
            "negative_close_no_pass": n_neg,
            "labeled_total": int(len(labeled)),
            "unlabeled_total": int(len(events) - len(labeled)),
            "positive_rate_among_labeled": float(n_pos / len(labeled)) if len(labeled) else None,
        },
        "events_by_attacker": _counts(events["attacker"]),
        "events_by_target": _counts(events["target"]),
        "events_by_zone": _counts(events["zone_id"]),
        "events_by_race_phase": _counts(phase.astype(str)),
        "passes_by_attacker": _counts(passes["attacker"]) if len(passes) else {},
        "passes_by_zone": _counts(passes["zone_id"]) if len(passes) else {},
        "event_duration_s": _describe(events["event_duration_s"]),
        "in_band_duration_s": _describe(events["in_band_duration_s"]),
        "outcome_window_s": _describe(events["outcome_window_s"]),
        "gap_at_start_m": _describe(events["gap_at_start_m"]),
        "closing_speed_at_start_kmh": _describe(events["closing_speed_at_start_kmh"]),
        "relative_speed_at_start_kmh": _describe(events["relative_speed_at_start_kmh"]),
        "estimated_energy_index_at_start": _describe(events["EstimatedEnergyIndex"]),
        "lookahead_deploy_at_start": _describe(events["LookaheadDeployProxy"]),
        "lookahead_covered_m": _describe(events["LookaheadCoveredM"]),
        "lookahead_truncated_events": int(events["lookahead_truncated"].sum()),
        "rejected_missing_data": int(
            audit["reason"].str.contains("no attacker sample|no target sample|no causal target|missing gap|not in driver map", regex=True).sum()
        ),
        "rejected_race_state_or_pit": int(
            (events["event_type"] == "PIT_RELATED").sum()
            + audit["reason"].str.contains("race state not green").sum()
        ),
        "missing_values_in_events": events.isna().sum().astype(int).to_dict(),
        "label_conditioned_summary": {
            col: {
                "no_pass_median": float(pd.to_numeric(labeled.loc[labeled["overtake_success"] == 0, col], errors="coerce").median()),
                "pass_median": float(pd.to_numeric(labeled.loc[labeled["overtake_success"] == 1, col], errors="coerce").median()),
            }
            for col in [
                "gap_at_start_m",
                "relative_speed_at_start_kmh",
                "closing_speed_at_start_kmh",
                "EstimatedEnergyIndex",
                "LookaheadDeployProxy",
                "in_band_duration_s",
            ]
        },
        "feature_validity_warnings": [
            "relative_speed_at_start_kmh and closing_speed_at_start_kmh are contaminated by "
            "track-phase offset. At the median no-pass gap of ~93 m the two cars are at "
            "different points on the circuit, so one can be accelerating on a straight while "
            "the other is braking. That makes the no-pass median relative speed +80 km/h versus "
            "-3 km/h for passes, which is a geometry artifact and not a pace signal. Before ML, "
            "sample the target at equal track distance rather than equal wall-clock time, or "
            "restrict these features to small gaps.",
            "closing_speed_at_start_kmh is missing for 52% of passes but only 25% of no-pass "
            "events, because a fresh engagement has no pre-t0 history against that target. The "
            "missingness itself carries label information; use closing_speed_available explicitly "
            "instead of imputing.",
            "EstimatedEnergyIndex at event time is compressed near 1.0 (median 0.994 for passes, "
            "0.992 for no-pass). It has very little dynamic range at these timestamps, so its "
            "marginal value to an overtake model on this race alone is likely small. This is an "
            "observation about the event sample, not a defect in the energy model.",
            "distance_to_next_corner_m / distance_to_next_straight_m are NaN for events in the "
            "final zone before the line, because the existing lookahead convention does not wrap "
            "past the start/finish line. This was preserved rather than changed.",
        ],
        "leakage_controls": {
            "feature_timestamp": "Every feature is read at t0 or earlier. Attacker state comes from the exact sample at t0; target state comes from a backward-only asof within the stated tolerance.",
            "asof_direction": "backward only, never nearest and never forward",
            "closing_speed": f"gap trend over [t0 - {params.closing_window_s}s, t0] against the same target; cannot span the pass",
            "energy": "sample-level EstimatedEnergyIndex at t0, not lap-start and not end-of-lap, and not read from the lap strategy CSV",
            "lookahead": "computed from the attacker's track distance at t0 using the existing lookahead_from_distance",
            "gap_min_m_rationale": "prevents t0 from sitting inside an already-completing pass",
            "label_uses_future": "yes, by design; only the label may look forward",
            "position_change": "retained as attacker_PositionChange_audit only, never used to define the label",
        },
        "reciprocal_same_lap_passes": int(
            sum(
                1
                for a, b, lap in set(zip(passes["attacker"], passes["target"], passes["event_start_lap"]))
                if (b, a, lap) in set(zip(passes["attacker"], passes["target"], passes["event_start_lap"]))
            )
        ),
        "known_limitations": [
            "Order-change evidence is not two independent measurements. cum_distance_diff tracks the FastF1 gap almost exactly, so the distance test and the DriverAhead test largely corroborate one estimator rather than cross-check two.",
            "The outcome window runs to the end of the engagement plus horizon_s, so it is variable (median 48 s, max 423 s). Long-tail events attribute a pass to a battle that lasted several laps.",
            "Passes concentrate on laps 2-6 (69 of 96). The opening-lap scramble dominates the positive class, which limits diversity even though 19 attackers appear.",
            "A pass and an immediate re-pass are stored as two events, one per direction. That is faithful to on-track order changes but inflates positives relative to net position gains.",
            "DistanceToDriverAhead is FastF1's speed-integrated estimate, not FIA timing. Absolute metres carry estimator error.",
            "One race only. Circuit, weather, and regulation effects are entirely unobserved.",
        ],
        "reported_energy_bugs": [],
        "kill_gate": {
            "criteria": {
                "min_on_track_passes": MIN_PASSES_FOR_ML,
                "min_distinct_attackers": MIN_DISTINCT_ATTACKERS,
                "min_distinct_zones": MIN_DISTINCT_ZONES,
                "note": "Criteria fixed before inspecting labels. Not loosened to raise the positive count.",
            },
            "observed_on_track_passes": n_pos,
            "observed_distinct_attackers": n_attackers_pass,
            "observed_distinct_zones": n_zones_pass,
            "result": kill_gate,
        },
        "files_created": [
            str(config.OVERTAKE_EVENTS_CSV),
            str(config.OVERTAKE_EVENTS_AUDIT_CSV),
            str(config.EVENT_DETECTION_REPORT_JSON),
            str(config.EVENT_MANUAL_REVIEW_CSV),
        ],
        "energy_hashes_before": hashes_before,
        "energy_hashes_after": hashes_after,
    }
    config.EVENT_DETECTION_REPORT_JSON.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    _header("VALIDATION / KILL GATE")
    print(f"  sample rows examined: {len(ahead)}")
    print(f"  proximity episodes: {len(events)}")
    print(f"  ON_TRACK_PASS: {n_pos}   CLOSE_INTERACTION_NO_PASS: {n_neg}")
    print(f"  labeled: {len(labeled)}   unlabeled: {len(events) - len(labeled)}")
    print(f"  distinct attackers with a pass: {n_attackers_pass}   distinct zones: {n_zones_pass}")
    print(f"  energy files untouched: {energy_untouched}")
    print(f"\n  KILL GATE: {kill_gate}")
    print(f"  wrote {config.EVENT_DETECTION_REPORT_JSON}")
    print("\n  ML was not trained. No model selected. No strategy engine. No extra races.")
    return report


if __name__ == "__main__":
    main()
