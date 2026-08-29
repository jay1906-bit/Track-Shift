"""Sample-level EstimatedEnergyIndex walk. Approach B owns this state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def walk_driver_energy(samples: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    out = samples.sort_values(["LapNumber", "Date"]).reset_index(drop=True)
    n = len(out)
    energy = np.empty(n, dtype=float)
    clipped_low = np.zeros(n, dtype=bool)
    clipped_high = np.zeros(n, dtype=bool)
    e = float(config.E0)
    energy[0] = e
    # First row of the driver: hold E0; first sample has undefined Δt anyway.
    for i in range(n):
        if i == 0:
            energy[i] = e
            continue
        usable = bool(out.at[i, "DeltaTimeValid"]) and bool(out.at[i, "SampleValid"])
        if not usable:
            energy[i] = e
            continue
        dt = float(out.at[i, "DeltaTimeSeconds"])
        deploy = float(out.at[i, "DeployProxy"])
        harvest = float(out.at[i, "HarvestProxy"])
        raw = e - alpha * deploy * dt + beta * harvest * dt
        # Ceiling clip while already full is expected at E0=1.0 when harvest>0.
        # Count as a bound hit only if the state had left the ceiling, or hit the floor.
        if raw < config.E_MIN:
            clipped_low[i] = True
        if raw > config.E_MAX and e < (config.E_MAX - 1e-12):
            clipped_high[i] = True
        e = min(config.E_MAX, max(config.E_MIN, raw))
        energy[i] = e
    out["EstimatedEnergyIndex"] = energy
    out["EnergyClippedLow"] = clipped_low
    out["EnergyClippedHigh"] = clipped_high
    return out


def walk_all_drivers(samples: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    parts = []
    for driver, grp in samples.groupby("Driver", sort=True):
        parts.append(walk_driver_energy(grp, alpha, beta))
    return pd.concat(parts, ignore_index=True)


def end_of_lap_energy(sample_energy: pd.DataFrame) -> pd.DataFrame:
    last = (
        sample_energy.sort_values(["Driver", "LapNumber", "Date"])
        .groupby(["Driver", "LapNumber"], as_index=False)
        .tail(1)
        .loc[:, ["Driver", "LapNumber", "EstimatedEnergyIndex"]]
        .rename(columns={"EstimatedEnergyIndex": "EstimatedEnergyIndex_end"})
    )
    stats = sample_energy.groupby(["Driver", "LapNumber"], as_index=False).agg(
        EstimatedEnergyIndex_min=("EstimatedEnergyIndex", "min"),
        EstimatedEnergyIndex_max=("EstimatedEnergyIndex", "max"),
        n_clip_low=("EnergyClippedLow", "sum"),
        n_clip_high=("EnergyClippedHigh", "sum"),
    )
    return last.merge(stats, on=["Driver", "LapNumber"], how="left")


def clip_lap_fraction(end_table: pd.DataFrame) -> float:
    """Fraction of laps that hit a bound during an update (not merely starting at E0)."""
    hit = (end_table["n_clip_low"] > 0) | (end_table["n_clip_high"] > 0)
    return float(hit.mean()) if len(end_table) else 0.0
