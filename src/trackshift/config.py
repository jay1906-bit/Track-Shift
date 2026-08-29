"""Modeling constants from the TrackShift Energy System Design Report.

These are simulation scale / numerical-guard choices.
They are NOT F1 battery capacity, MGU-K limits, ERS limits, or SOC parameters.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = REPO_ROOT / "cache"
RAW_LAPS_CSV = REPO_ROOT / "data" / "raw" / "australian_gp_2026_laps_raw.csv"
CLEAN_LAPS_CSV = REPO_ROOT / "data" / "processed" / "australian_gp_2026_laps_clean.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

YEAR = 2026
EVENT = "Australian Grand Prix"
SESSION_CODE = "R"

REFERENCE_DRIVER = "NOR"
REFERENCE_LAP = 5

# Delta-time guards
DT_GAP_SECONDS = 1.0

# Speed / throttle guards
SPEED_CLIP_KMH = (0.0, 400.0)
THROTTLE_MIN = 0.0
THROTTLE_MAX = 100.0  # 104 and other out-of-range values are INVALID, not clipped

# Acceleration
ROLLING_MEDIAN_WINDOW = 3
A_CLIP_MS2 = (-20.0, 15.0)  # numerical guard, not an F1 brake spec

# Harvest gate
D_MIN_MS2 = 1.0  # ~0.1 g; modeling choice, not an F1 regen map

# Race state
YELLOW_FACTOR = 0.3  # simulation choice, not an F1 rule
FREEZE_TRACK_CODES = frozenset({"4", "5", "6", "7"})
YELLOW_TRACK_CODE = "2"

# Thin-lap handling
THIN_LAP_MIN_VALID_SAMPLES = 50

# Winsorization
WINSOR_QUANTILE = 0.995

# Energy
E0 = 1.0
E_MIN = 0.0
E_MAX = 1.0
TARGET_MEDIAN_ABS_NET = 0.03  # desired median |net|; T is derived from calibration imbalance
CLIP_RATE_REDUCE_FACTOR = 0.5
CLIP_RATE_MAX_ITERS = 8
CLIP_LAP_FRACTION_LIMIT = 0.20

# Approach C geometry (Albert Park scale labels, not F1 brake markers)
BRAKE_ZONE_BEFORE_M = 150.0
CORNER_HALF_WIDTH_M = 30.0
LOOKAHEAD_M = 400.0

# Output artifacts
LAP_PROXIES_CSV = PROCESSED_DIR / "australian_gp_2026_lap_energy_proxies.csv"
ENERGY_V1_CSV = PROCESSED_DIR / "australian_gp_2026_laps_energy_v1.csv"
ENERGY_TRACKAWARE_CSV = PROCESSED_DIR / "australian_gp_2026_laps_energy_trackaware_v1.csv"
ZONE_TABLE_CSV = PROCESSED_DIR / "australian_gp_2026_circuit_zones.csv"
ENERGY_CONFIG_JSON = OUTPUTS_DIR / "energy_config.json"
CHECKPOINT_REPORT_JSON = OUTPUTS_DIR / "checkpoint_report.json"
SAMPLE_PROXIES_PARQUET = OUTPUTS_DIR / "sample_proxies.parquet"
REFERENCE_SAMPLES_CSV = OUTPUTS_DIR / "reference_lap_samples.csv"

# Strategy feature-integration artifacts (not energy). Do not write into energy CSVs.
LAPS_DERIVED_CSV = PROCESSED_DIR / "australian_gp_2026_laps_derived_v1.csv"
DRIVER_AHEAD_SAMPLES_CSV = PROCESSED_DIR / "australian_gp_2026_car_driver_ahead_samples_v1.csv"
STRATEGY_FEATURES_CSV = PROCESSED_DIR / "australian_gp_2026_laps_strategy_features_v1.csv"
STRATEGY_FEATURES_REPORT_JSON = OUTPUTS_DIR / "strategy_features_report.json"

# ----------------------------------------------------------------------
# Overtake-event detection (Phases 1-7). Detection geometry only.
# These are NOT energy parameters and do not enter the energy walk.
# ----------------------------------------------------------------------
RACE_ID = "australian_gp_2026_R"
CIRCUIT_NAME = "Albert Park"

# Proximity band that defines "in a battle". 100 m is ~1.2-1.8 s at Albert
# Park racing speeds. Chosen from track geometry and speed, never tuned
# against the pass/no-pass outcome.
EVENT_GAP_MAX_M = 100.0
# Lower edge of the band. An F1 car is ~5.6 m long, and DistanceToDriverAhead is
# speed-integrated, so below ~10 m "who is ahead" sits inside measurement noise
# and the pass is already in progress. Predicting from there would leak the
# outcome, so t0 must be a decision point with the attacker genuinely behind.
EVENT_GAP_MIN_M = 10.0
# Car data is ~4.17 Hz (median dt 0.24 s), so 3 s is >= ~12 samples.
EVENT_MIN_PERSIST_S = 3.0
# Outcome window after t0. Albert Park green laps run ~85-88 s, so 30 s is
# roughly a third of a lap: long enough to complete a pass after a follow.
# FINAL. Chosen from the observed t0->pass delay distribution: the median pass
# lands at 16.1 s and capture flattens to ~4 passes per extra 5 s beyond 30 s,
# while cross-line exposure climbs from ~35% at 30 s to ~71% at 60 s.
EVENT_HORIZON_S = 30.0
# The horizon is measured from t0, the decision point, so every event is scored
# over exactly EVENT_HORIZON_S. "event_end" reproduces the v1-v3 behaviour where
# the window ran to the end of the engagement and stretched to 423 s.
EVENT_HORIZON_ANCHOR = "t0"
# Audit-only lookahead for reporting a swap that lands past the horizon. It can
# never widen the label window; see events.build_events.
EVENT_AUDIT_HORIZON_S = 120.0
# Bridge short telemetry dropouts inside one battle instead of splitting it.
EVENT_MERGE_GAP_S = 2.0
# Strictly pre-t0 window used for the closing-speed feature.
EVENT_CLOSING_WINDOW_S = 2.0
# Causal asof tolerance when reading the target car's own sample stream.
EVENT_ASOF_TOLERANCE_MS = 300
# Lap 1 contains the standing start; grid packing makes gaps meaningless.
EVENT_EXCLUDE_LAP_NUMBERS = (1,)
# Pair separation beyond this fraction of a lap is a lapping situation.
EVENT_LAPPED_FRACTION = 0.5
# A jump larger than this fraction of a lap between consecutive samples is a
# discontinuity (leader/lapped-car reassignment), not a racing gap change.
EVENT_GAP_JUMP_FRACTION = 0.5

# v2 competitor-speed features: attacker and target compared at the SAME TRACK
# POSITION instead of the same wall-clock instant. Detection is unchanged.
EVENT_SPATIAL_LOOKBACK_S = 30.0
# Slope window for closing speed. 3 s is ~12 samples at 4.17 Hz, which makes the
# least-squares slope stable, and it still ends at t0.
EVENT_CLOSING_WINDOW_V2_S = 3.0
# A car 10-100 m ahead crossed the attacker's current position at most a few
# seconds earlier; a larger lag means the alignment matched a stale lap.
EVENT_MAX_ALIGNMENT_LAG_S = 10.0

OVERTAKE_EVENTS_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_v1.csv"
OVERTAKE_EVENTS_V2_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_v2.csv"
EVENT_FEATURES_V2_REPORT_JSON = OUTPUTS_DIR / "event_features_v2_report.json"

# v3: continuous race distance in the detector's lapped test and swap evidence.
OVERTAKE_EVENTS_V3_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_v3.csv"
OVERTAKE_EVENTS_AUDIT_V3_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_audit_v3.csv"
OVERTAKE_EVENTS_CHANGED_V3_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_changed_v1_to_v3.csv"
EVENT_DETECTION_V3_REPORT_JSON = OUTPUTS_DIR / "event_detection_v3_report.json"
OVERTAKE_EVENTS_AUDIT_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_audit_v1.csv"
EVENT_DETECTION_REPORT_JSON = OUTPUTS_DIR / "event_detection_report.json"
EVENT_MANUAL_REVIEW_CSV = OUTPUTS_DIR / "event_manual_review_examples.csv"

# v4: FINAL event definition. Horizon anchored at t0, fixed 30 s.
OVERTAKE_EVENTS_V4_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_v4.csv"
OVERTAKE_EVENTS_AUDIT_V4_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_audit_v4.csv"
OVERTAKE_EVENTS_CHANGED_V4_CSV = PROCESSED_DIR / "australian_gp_2026_overtake_events_changed_v3_to_v4.csv"
EVENT_DETECTION_V4_REPORT_JSON = OUTPUTS_DIR / "event_detection_v4_report.json"
EVENT_MANUAL_REVIEW_V4_CSV = OUTPUTS_DIR / "event_manual_review_v4.csv"

# ----------------------------------------------------------------------
# Strategy engine v1 (ATTACK / SAVE). Does not modify energy or events.
# ----------------------------------------------------------------------
ML_SELECTION_DIR = OUTPUTS_DIR / "ml_selection_v1"
ML_OOF_PREDICTIONS_CSV = ML_SELECTION_DIR / "oof_predictions.csv"
ML_LOCKED_TEST_PREDICTIONS_CSV = ML_SELECTION_DIR / "locked_test_predictions.csv"
ML_SAFE_EVENTS_CSV = PROCESSED_DIR / "multirace_v1" / "overtake_events_ml_safe_v1.csv"
POOLED_EVENTS_CSV = PROCESSED_DIR / "multirace_v1" / "overtake_events_pooled_v1.csv"
STRATEGY_V1_DIR = OUTPUTS_DIR / "strategy_v1"
STRATEGY_V1_CONFIG_JSON = STRATEGY_V1_DIR / "strategy_config.json"
STRATEGY_V1_REPORT_MD = STRATEGY_V1_DIR / "REPORT.md"

# Frontend replay adapter (read-only over frozen strategy/ML outputs).
FRONTEND_V1_DIR = OUTPUTS_DIR / "frontend_v1"
FRONTEND_REPLAY_JSON = FRONTEND_V1_DIR / "replay.json"
