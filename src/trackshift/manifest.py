"""Column roles for the multi-race event dataset.

Every column carries exactly one role. The ML feature matrix is built only from
``feature_t0``; nothing else may reach a model.

    identifier     names a row or links it to another row
    group_key      defines the clusters used for grouped evaluation
    feature_t0     known at or before t0, legal model input
    episode_level  describes the episode, so spans time after t0
    audit          kept for inspection, never a model input
    label          the supervised target

``causal`` records whether a column can be evaluated with information available
at t0. A column with ``causal = False`` can never be ``feature_t0``; the
``reason`` says what future information it depends on.

``portability`` answers "can this feature exist on a completely unseen
circuit?": PORTABLE, CIRCUIT_SPECIFIC, or UNKNOWN.
"""

from __future__ import annotations

import pandas as pd

IDENTIFIER = "identifier"
GROUP_KEY = "group_key"
FEATURE_T0 = "feature_t0"
EPISODE_LEVEL = "episode_level"
AUDIT = "audit"
LABEL = "label"

PORTABLE = "PORTABLE"
CIRCUIT_SPECIFIC = "CIRCUIT_SPECIFIC"

# column, role, causal, portability, constant_within_race, reason
_SPEC: list[tuple] = [
    # ---------------- identifiers ----------------
    ("event_id", IDENTIFIER, True, PORTABLE, False, "row name"),
    ("attacker", IDENTIFIER, True, PORTABLE, False,
     "driver identity is kept for audit; making it a feature would let the model "
     "memorise driver-specific passing tendencies rather than learn the situation"),
    ("target", IDENTIFIER, True, PORTABLE, False, "as attacker"),
    ("target_number", IDENTIFIER, True, PORTABLE, False, "car number of the target"),
    ("reciprocal_event_id", IDENTIFIER, False, PORTABLE, False,
     "links the opposite-direction event describing the same racing situation; "
     "resolved using both events' horizons"),

    # ---------------- group keys ----------------
    ("race_id", GROUP_KEY, True, PORTABLE, True, "immutable grouping variable for race-based evaluation"),
    ("circuit", GROUP_KEY, True, PORTABLE, True, "circuit is a group, never a feature"),
    ("year", GROUP_KEY, True, PORTABLE, True, "season"),
    ("pair_id", GROUP_KEY, True, PORTABLE, False, "unordered attacker/target pair within a race"),
    ("cluster_id", GROUP_KEY, True, PORTABLE, False, "race + lap + pair, the tightest correlation cluster"),
    ("event_start_lap", GROUP_KEY, True, PORTABLE, False, "lap on which t0 falls"),

    # ---------------- t0 features ----------------
    ("gap_at_start_m", FEATURE_T0, True, PORTABLE, False, "separation at t0"),
    ("attacker_speed_at_start_kmh", FEATURE_T0, True, PORTABLE, False, "attacker telemetry at t0"),
    ("target_speed_at_attacker_position_kmh", FEATURE_T0, True, PORTABLE, False,
     "target speed sampled where it last crossed the attacker's t0 track position, at or before t0"),
    ("relative_speed_same_position_kmh", FEATURE_T0, True, PORTABLE, False,
     "spatially aligned speed difference; the wall-clock version compared two cars at different "
     "points on the track and is retained only as audit"),
    ("closing_speed_at_start_kmh", FEATURE_T0, True, PORTABLE, False,
     "least-squares slope of pair separation over a pre-t0 window; never uses the pass interval"),
    ("spatial_time_gap_s", FEATURE_T0, True, PORTABLE, False,
     "time for the attacker to reach where the target was, measured backwards from t0"),
    ("spatial_alignment_ok", FEATURE_T0, True, PORTABLE, False,
     "alignment quality flag, known at t0; an explicit indicator instead of a silent imputation"),
    ("closing_speed_available", FEATURE_T0, True, PORTABLE, False,
     "missingness indicator, known at t0"),
    ("track_distance_at_start_m", FEATURE_T0, True, CIRCUIT_SPECIFIC, False,
     "absolute metres into the lap; the same number means a different place on every circuit, "
     "use track_position_frac for cross-circuit work"),
    ("track_position_frac", FEATURE_T0, True, PORTABLE, False,
     "track_distance_at_start_m / circuit_length_m, comparable across circuits"),
    ("zone_type", FEATURE_T0, True, PORTABLE, False,
     "straight / brake / corner; the portable replacement for zone_id"),
    ("distance_to_next_corner_m", FEATURE_T0, True, PORTABLE, False,
     "structurally missing when no corner remains before the line; missingness is preserved"),
    ("distance_to_next_straight_m", FEATURE_T0, True, PORTABLE, False,
     "structurally missing when no straight remains before the line; missingness is preserved"),
    ("compound", FEATURE_T0, True, PORTABLE, False, "tyre compound at t0"),
    ("tyre_age", FEATURE_T0, True, PORTABLE, False, "laps on the tyre at t0"),
    ("stint", FEATURE_T0, True, PORTABLE, False, "stint index at t0"),
    ("EstimatedEnergyIndex", FEATURE_T0, True, PORTABLE, False,
     "sample-level energy state at t0, not the end-of-lap aggregate"),
    ("LookaheadDeployProxy", FEATURE_T0, True, PORTABLE, False,
     "computed from the attacker's track position at t0 over the next 400 m"),
    ("LookaheadHarvestProxy", FEATURE_T0, True, PORTABLE, False, "as LookaheadDeployProxy"),
    ("LookaheadCoveredM", FEATURE_T0, True, PORTABLE, False,
     "metres of the 400 m horizon that lie before the line"),
    ("lookahead_truncated", FEATURE_T0, True, PORTABLE, False,
     "the 400 m horizon ran past the line and was not wrapped"),
    ("circuit_length_m", FEATURE_T0, True, PORTABLE, True, "exogenous circuit geometry"),
    ("corner_count", FEATURE_T0, True, PORTABLE, True, "exogenous circuit geometry"),
    ("longest_straight_m", FEATURE_T0, True, PORTABLE, True, "exogenous circuit geometry"),
    ("n_long_straights_over_500m", FEATURE_T0, True, PORTABLE, True, "exogenous circuit geometry"),

    # ---------------- episode level ----------------
    ("event_end_time", EPISODE_LEVEL, False, PORTABLE, False,
     "the engagement's end, which is not known at t0"),
    ("event_duration_s", EPISODE_LEVEL, False, PORTABLE, False,
     "how long the pair stayed in the band, measurable only after the fact"),
    ("in_band_duration_s", EPISODE_LEVEL, False, PORTABLE, False, "as event_duration_s"),
    ("gap_min_in_episode_m", EPISODE_LEVEL, False, PORTABLE, False,
     "minimum over the whole episode, so it reads samples after t0"),
    ("n_samples_in_episode", EPISODE_LEVEL, False, PORTABLE, False, "counts samples after t0"),

    # ---------------- audit ----------------
    ("event_start_time", AUDIT, True, PORTABLE, False, "t0 itself; timestamps are not model inputs"),
    ("horizon_end_time", AUDIT, True, PORTABLE, False, "t0 + horizon_s, definitional"),
    ("horizon_s_effective", AUDIT, True, PORTABLE, True, "constant 30 s by construction"),
    ("event_type", AUDIT, False, PORTABLE, False, "the taxonomy decision that produces the label"),
    ("actual_swap_delay_s", AUDIT, False, PORTABLE, False, "measured after t0; derived from the outcome"),
    ("pass_after_horizon", AUDIT, False, PORTABLE, False, "derived from the outcome"),
    ("window_crosses_start_finish", AUDIT, False, PORTABLE, False,
     "evaluated over the whole horizon window"),
    ("attacker_PositionChange_audit", AUDIT, False, PORTABLE, False,
     "lap-over-lap classified position change; a future quantity and never used to build the label"),
    ("attacker_position", AUDIT, False, PORTABLE, False,
     "the lap spine's Position is the classification at the END of the lap, so on a lap where the "
     "pass happens it already reflects the pass; measured on Australia v4 the spine places the "
     "attacker ahead in 41% of passes against 3% of no-passes, which is the outcome leaking"),
    ("target_position", AUDIT, False, PORTABLE, False, "as attacker_position"),
    ("zone_id", AUDIT, True, CIRCUIT_SPECIFIC, False,
     "kept for visualisation and error analysis; brake_T11 exists on one circuit only"),
    ("gap_at_start_s_crude_audit", AUDIT, True, PORTABLE, False,
     "gap divided by attacker speed; superseded by spatial_time_gap_s"),
    ("target_speed_same_time_kmh_audit", AUDIT, True, PORTABLE, False,
     "wall-clock aligned target speed, superseded by the spatially aligned version"),
    ("relative_speed_same_time_kmh_audit", AUDIT, True, PORTABLE, False, "superseded"),
    ("closing_speed_v1_kmh_audit", AUDIT, True, PORTABLE, False, "superseded by the v3/v4 implementation"),
    ("closing_speed_v1_window_s_audit", AUDIT, True, PORTABLE, False, "superseded"),
    ("closing_speed_v1_available_audit", AUDIT, True, PORTABLE, False, "superseded"),
    ("spatial_alignment_lag_raw_s", AUDIT, True, PORTABLE, False, "uncensored alignment lag"),
    ("closing_speed_n_samples", AUDIT, True, PORTABLE, False, "diagnostic for the closing-speed fit"),
    ("closing_speed_window_s", AUDIT, True, PORTABLE, True, "constant by construction"),
    ("TrackStatus", AUDIT, True, PORTABLE, False,
     "green by construction on the scoreable set, so non-predictive there"),
    ("freeze_flag", AUDIT, True, PORTABLE, False, "false by construction on the scoreable set"),
    ("detector_version", AUDIT, True, PORTABLE, True,
     "provenance; the single version column for this dataset, replacing v4's separate "
     "feature_version"),

    # ---------------- label ----------------
    ("overtake_success", LABEL, False, PORTABLE, False,
     "1 when a corroborated on-track pairwise pass completes within 30 s of t0, 0 when the "
     "interaction is valid and observable and no such pass occurs, null otherwise"),
]

MANIFEST = pd.DataFrame(
    _SPEC, columns=["column", "role", "causal", "portability", "constant_within_race", "reason"]
)


def columns_for(role: str) -> list[str]:
    return MANIFEST.loc[MANIFEST["role"] == role, "column"].tolist()


def role_of(column: str) -> str | None:
    hit = MANIFEST.loc[MANIFEST["column"] == column, "role"]
    return None if hit.empty else str(hit.iloc[0])


def validate_against(frame: pd.DataFrame) -> dict:
    """Every dataset column must have exactly one role, and vice versa."""
    declared = set(MANIFEST["column"])
    actual = set(frame.columns)
    dupes = MANIFEST["column"][MANIFEST["column"].duplicated()].tolist()
    non_causal_features = MANIFEST.loc[
        (MANIFEST["role"] == FEATURE_T0) & (~MANIFEST["causal"]), "column"
    ].tolist()
    return {
        "n_declared": len(declared),
        "n_in_dataset": len(actual),
        "declared_but_missing_from_dataset": sorted(declared - actual),
        "in_dataset_but_undeclared": sorted(actual - declared),
        "duplicate_role_assignments": dupes,
        "feature_t0_columns_that_are_not_causal": non_causal_features,
        "passed": bool(
            not (declared - actual) and not (actual - declared) and not dupes
            and not non_causal_features
        ),
    }


def ml_safe_columns() -> list[str]:
    """Identifiers, group keys, t0 features and the label. Nothing else."""
    return (
        columns_for(IDENTIFIER)
        + columns_for(GROUP_KEY)
        + columns_for(FEATURE_T0)
        + columns_for(LABEL)
    )
