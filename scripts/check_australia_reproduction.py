"""Verify the multi-race orchestrator reproduces the frozen Australia v4 dataset.

The orchestrator rebuilds everything from the session: clean laps, telemetry,
proxies, per-race alpha/beta, zones, driver-ahead, events. If it lands on the
same events with the same labels as the stored v4 CSV, then running it on
another circuit is running the same detector, not a new one.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.race_pipeline import RaceSpec, process_race  # noqa: E402

NATURAL_KEY = ["attacker", "target_number", "event_start_time"]


def main() -> int:
    spec = RaceSpec(year=2026, event_name="Australian Grand Prix", slug="australia")
    t = time.time()
    art = process_race(spec, progress_every=250)
    elapsed = time.time() - t
    print(f"\nprocess_race took {elapsed:.0f}s")

    if art.error:
        print("FAILED at stage", art.stage, ":", art.error)
        print(art.quality.get("traceback", ""))
        return 1

    new = art.events.copy()
    new["target_number"] = new["target_number"].astype(str)
    new["event_start_time"] = pd.to_datetime(new["event_start_time"])

    ref = pd.read_csv(
        config.OVERTAKE_EVENTS_V4_CSV,
        parse_dates=["event_start_time", "event_end_time", "horizon_end_time"],
        dtype={"target_number": "string"},
    )
    ref["target_number"] = ref["target_number"].astype(str)

    print(f"\nreference v4 events : {len(ref)}")
    print(f"re-derived    events : {len(new)}")

    merged = ref.loc[:, NATURAL_KEY + ["event_id", "event_type", "overtake_success"]].merge(
        new.loc[:, NATURAL_KEY + ["event_id", "event_type", "overtake_success"]],
        on=NATURAL_KEY,
        how="outer",
        suffixes=("_ref", "_new"),
        indicator=True,
    )
    only_ref = int((merged["_merge"] == "left_only").sum())
    only_new = int((merged["_merge"] == "right_only").sum())
    both = merged.loc[merged["_merge"] == "both"]
    type_same = bool((both["event_type_ref"] == both["event_type_new"]).all())
    label_same = bool(
        (both["overtake_success_ref"].fillna(-1) == both["overtake_success_new"].fillna(-1)).all()
    )

    print(f"matched engagements  : {len(both)}")
    print(f"only in reference    : {only_ref}")
    print(f"only in re-derived   : {only_new}")
    print(f"event_type identical : {type_same}")
    print(f"label identical      : {label_same}")

    numeric = [
        "gap_at_start_m",
        "attacker_speed_at_start_kmh",
        "track_distance_at_start_m",
        "EstimatedEnergyIndex",
        "LookaheadDeployProxy",
        "LookaheadHarvestProxy",
        "LookaheadCoveredM",
        "relative_speed_same_position_kmh",
        "closing_speed_at_start_kmh",
        "spatial_time_gap_s",
    ]
    fcmp = ref.loc[:, NATURAL_KEY + numeric].merge(
        new.loc[:, NATURAL_KEY + numeric], on=NATURAL_KEY, suffixes=("_ref", "_new")
    )
    drift = {}
    for col in numeric:
        a = pd.to_numeric(fcmp[f"{col}_ref"], errors="coerce")
        b = pd.to_numeric(fcmp[f"{col}_new"], errors="coerce")
        d = (a - b).abs()
        drift[col] = {
            "max_abs_diff": float(np.nanmax(d)) if len(d) else 0.0,
            "allclose": bool(np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-6)),
        }
    print("\nfeature drift on matched events:")
    for col, d in drift.items():
        print(f"  {col:44s} max|diff|={d['max_abs_diff']:.6g}  allclose={d['allclose']}")

    ref_cfg = json.loads(config.ENERGY_CONFIG_JSON.read_text(encoding="utf-8"))
    print("\nalpha stored / re-derived:", ref_cfg["alpha"], art.energy_config["alpha"])
    print("beta  stored / re-derived:", ref_cfg["beta"], art.energy_config["beta"])
    print("alpha match:", np.isclose(ref_cfg["alpha"], art.energy_config["alpha"], rtol=1e-9))
    print("beta  match:", np.isclose(ref_cfg["beta"], art.energy_config["beta"], rtol=1e-9))

    ok = only_ref == 0 and only_new == 0 and type_same and label_same
    print("\nREPRODUCTION:", "PASS" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
