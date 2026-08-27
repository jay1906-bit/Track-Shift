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
TARGET_MEDIAN_ABS_NET = 0.03  # prototype target; not an F1 constant
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
