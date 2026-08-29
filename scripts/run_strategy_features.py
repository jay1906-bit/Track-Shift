"""Feature-integration step: derived laps + car-data driver-ahead join.

Does not modify Approach B, alpha/beta, energy files, zones, or lookahead.
Does not implement ATTACK / SAVE / DEFEND.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.session_io import load_clean_laps, load_session, pick_session_lap  # noqa: E402
from trackshift.strategy_features import (  # noqa: E402
    AHEAD_NOTE,
    LAP_SUMMARY_NOTE,
    _normalize_driver_ahead,
    build_derived_laps,
    extract_all_driver_ahead,
    join_strategy_features,
    summarize_driver_ahead,
    validate_strategy_features,
)
from trackshift.validate import file_sha256  # noqa: E402

ENERGY_PATHS = {
    "energy_v1": config.ENERGY_V1_CSV,
    "trackaware": config.ENERGY_TRACKAWARE_CSV,
    "zones": config.ZONE_TABLE_CSV,
    "lap_proxies": config.LAP_PROXIES_CSV,
    "energy_config": config.ENERGY_CONFIG_JSON,
}


def _hash_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> dict:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_before = file_sha256(config.RAW_LAPS_CSV)
    clean_before = file_sha256(config.CLEAN_LAPS_CSV)
    energy_hashes_before = _hash_map(ENERGY_PATHS)
    energy_v1_before = pd.read_csv(config.ENERGY_V1_CSV)
    trackaware_before = pd.read_csv(config.ENERGY_TRACKAWARE_CSV)

    print("Loading FastF1 session from cache and cleaned lap spine...")
    session = load_session()
    laps_clean = load_clean_laps()
    print("Clean laps:", len(laps_clean), "drivers:", laps_clean["Driver"].nunique())

    _print_header("DERIVED LAP FEATURES")
    derived = build_derived_laps(laps_clean)
    derived.to_csv(config.LAPS_DERIVED_CSV, index=False)
    print("wrote", config.LAPS_DERIVED_CSV, "rows=", len(derived))
    derived = pd.read_csv(config.LAPS_DERIVED_CSV)
    print(derived[["LapTimeSeconds", "LapTimeDelta", "PositionChange"]].describe().to_string())
    print("NaN counts:\n", derived[["LapTimeSeconds", "LapTimeDelta", "PositionChange"]].isna().sum().to_string())

    _print_header("DRIVER AHEAD FROM ORIGINAL CAR DATA")
    print(AHEAD_NOTE)
    samples = extract_all_driver_ahead(session, laps_clean, progress_every=25)
    samples["DriverAhead"] = samples["DriverAhead"].map(_normalize_driver_ahead)
    samples.to_csv(config.DRIVER_AHEAD_SAMPLES_CSV, index=False)
    print("wrote", config.DRIVER_AHEAD_SAMPLES_CSV, "samples=", len(samples))
    samples = pd.read_csv(
        config.DRIVER_AHEAD_SAMPLES_CSV,
        parse_dates=["Date"],
        dtype={"DriverAhead": "string"},
    )
    samples["DriverAhead"] = samples["DriverAhead"].map(_normalize_driver_ahead)

    ahead_laps = summarize_driver_ahead(samples, laps_clean)
    print(LAP_SUMMARY_NOTE)

    _print_header("STRATEGY FEATURE JOIN")
    strategy = join_strategy_features(laps_clean, derived, trackaware_before, ahead_laps)
    if "DriverAhead_end" in strategy.columns:
        strategy["DriverAhead_end"] = strategy["DriverAhead_end"].map(_normalize_driver_ahead)
    strategy.to_csv(config.STRATEGY_FEATURES_CSV, index=False)
    print("wrote", config.STRATEGY_FEATURES_CSV, "rows=", len(strategy))
    strategy = pd.read_csv(
        config.STRATEGY_FEATURES_CSV,
        dtype={"DriverAhead_end": "string"},
    )
    if "DriverAhead_end" in strategy.columns:
        strategy["DriverAhead_end"] = strategy["DriverAhead_end"].map(_normalize_driver_ahead)

    ref_dates = None
    try:
        ref_lap = pick_session_lap(session, config.REFERENCE_DRIVER, config.REFERENCE_LAP)
        ref_car = ref_lap.get_car_data(pad=1, pad_side="both")
        ref_dates = ref_car.iloc[1:-1]["Date"]
        n_ref_ahead = (
            int(((samples["Driver"] == "NOR") & (samples["LapNumber"].astype(int) == 5)).sum())
            if len(samples)
            else 0
        )
        print("NOR lap 5 car_data padded n=", len(ref_car), "ahead samples n=", n_ref_ahead)
    except Exception as exc:
        print("reference date check skipped:", exc)

    energy_v1_after = pd.read_csv(config.ENERGY_V1_CSV)
    trackaware_after = pd.read_csv(config.ENERGY_TRACKAWARE_CSV)
    raw_after = file_sha256(config.RAW_LAPS_CSV)
    clean_after = file_sha256(config.CLEAN_LAPS_CSV)
    energy_hashes_after = _hash_map(ENERGY_PATHS)

    session_drivers = list(session.drivers) if getattr(session, "drivers", None) is not None else []
    validation = validate_strategy_features(
        laps_clean=laps_clean,
        derived=derived,
        samples=samples,
        strategy=strategy,
        trackaware_before=trackaware_before,
        energy_v1_before=energy_v1_before,
        trackaware_after=trackaware_after,
        energy_v1_after=energy_v1_after,
        session_drivers=session_drivers,
        raw_hash_before=raw_before,
        clean_hash_before=clean_before,
        raw_hash_after=raw_after,
        clean_hash_after=clean_after,
        energy_hashes_before=energy_hashes_before,
        energy_hashes_after=energy_hashes_after,
        reference_car_dates=ref_dates,
    )

    has_ahead_sample = (
        samples.assign(_has=samples["DriverAhead"].fillna("").astype(str).str.strip().ne(""))
        .groupby(["Driver", "LapNumber"], sort=False)["_has"]
        .any()
        if len(samples)
        else pd.Series(dtype=bool)
    )
    n_laps_with_ahead = int(has_ahead_sample.sum()) if len(has_ahead_sample) else 0
    n_laps_in_samples = int(has_ahead_sample.shape[0]) if len(has_ahead_sample) else 0
    n_laps_no_ahead = n_laps_in_samples - n_laps_with_ahead
    n_end_ahead = int((strategy["DriverAhead_end"].fillna("").astype(str).str.strip() != "").sum())
    dist = pd.to_numeric(samples["DistanceToDriverAhead"], errors="coerce") if len(samples) else pd.Series(dtype=float)

    report = {
        "step": "strategy_feature_integration",
        "attack_save_defend_implemented": False,
        "energy_model_modified": False,
        "get_telemetry_used": False,
        "ahead_source": "lap.get_car_data(pad=1, pad_side='both').iloc[1:-1].add_driver_ahead()",
        "ahead_note": AHEAD_NOTE,
        "lap_summary_note": LAP_SUMMARY_NOTE,
        "files_created": [
            str(config.LAPS_DERIVED_CSV),
            str(config.DRIVER_AHEAD_SAMPLES_CSV),
            str(config.STRATEGY_FEATURES_CSV),
            str(config.STRATEGY_FEATURES_REPORT_JSON),
        ],
        "n_derived_rows": int(len(derived)),
        "n_strategy_rows": int(len(strategy)),
        "n_ahead_samples": int(len(samples)),
        "n_laps_with_driver_ahead_any_sample": n_laps_with_ahead,
        "n_laps_with_no_driver_ahead": n_laps_no_ahead,
        "n_laps_with_driver_ahead_end": n_end_ahead,
        "n_laps_missing_ahead_extract": int((strategy["n_car_samples_ahead_extract"].fillna(0) == 0).sum()),
        "distance_to_driver_ahead": {
            "n_present": int(dist.notna().sum()) if len(dist) else 0,
            "n_missing": int(dist.isna().sum()) if len(dist) else 0,
            "min": float(dist.min()) if dist.notna().any() else None,
            "median": float(dist.median()) if dist.notna().any() else None,
            "mean": float(dist.mean()) if dist.notna().any() else None,
            "max": float(dist.max()) if dist.notna().any() else None,
        },
        "position_change": {
            "n_nan": int(derived["PositionChange"].isna().sum()),
            "min": float(derived["PositionChange"].min()) if derived["PositionChange"].notna().any() else None,
            "median": float(derived["PositionChange"].median()) if derived["PositionChange"].notna().any() else None,
            "mean": float(derived["PositionChange"].mean()) if derived["PositionChange"].notna().any() else None,
            "max": float(derived["PositionChange"].max()) if derived["PositionChange"].notna().any() else None,
            "n_gained": int((derived["PositionChange"] < 0).sum()),
            "n_lost": int((derived["PositionChange"] > 0).sum()),
            "n_unchanged": int((derived["PositionChange"] == 0).sum()),
        },
        "join_coverage": {
            "strategy_rows": int(len(strategy)),
            "energy_non_null": int(strategy["EstimatedEnergyIndex"].notna().sum()),
            "lookahead_deploy_non_null": int(strategy["LookaheadDeployProxy"].notna().sum()),
            "lookahead_harvest_non_null": int(strategy["LookaheadHarvestProxy"].notna().sum()),
            "position_change_non_null": int(strategy["PositionChange"].notna().sum()),
            "driver_ahead_end_non_empty": n_end_ahead,
        },
        "missing_values": strategy.isna().sum().astype(int).to_dict(),
        "energy_hashes_before": energy_hashes_before,
        "energy_hashes_after": energy_hashes_after,
        "raw_unchanged": raw_before == raw_after,
        "clean_unchanged": clean_before == clean_after,
        "validation": validation,
    }
    config.STRATEGY_FEATURES_REPORT_JSON.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("wrote", config.STRATEGY_FEATURES_REPORT_JSON)

    _print_header("VALIDATION")
    for check in validation["checks"]:
        flag = "PASS" if check["passed"] else "FAIL"
        print(f"  {flag}  {check['name']}: {check['detail']}")
    print(
        f"\n{validation['n_passed']}/{validation['n_checks']} checks passed. "
        f"overall={'PASS' if validation['passed'] else 'FAIL'}"
    )
    print("ATTACK / SAVE / DEFEND were NOT implemented.")
    print("Energy calculations were not rerun and energy files were not rewritten.")
    return report


if __name__ == "__main__":
    main()
