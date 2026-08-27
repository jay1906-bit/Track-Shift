"""Session loading and cleaned-lap spine. Never writes to data/raw/."""

from __future__ import annotations

import pandas as pd
import fastf1

from . import config


def enable_cache() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))


def load_session():
    enable_cache()
    session = fastf1.get_session(config.YEAR, config.EVENT, config.SESSION_CODE)
    session.load()
    return session


def load_clean_laps() -> pd.DataFrame:
    laps = pd.read_csv(config.CLEAN_LAPS_CSV)
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    return laps


def pick_session_lap(session, driver: str, lap_number: int):
    mask = (session.laps["Driver"] == driver) & (session.laps["LapNumber"] == lap_number)
    subset = session.laps.loc[mask]
    if subset.empty:
        raise KeyError(f"No session lap for {driver} lap {lap_number}")
    return subset.iloc[0]


def reference_lap_row(laps_clean: pd.DataFrame) -> pd.Series:
    preferred = laps_clean[
        (laps_clean["Driver"] == config.REFERENCE_DRIVER)
        & (laps_clean["LapNumber"] == config.REFERENCE_LAP)
        & (laps_clean["IsAccurate"] == True)  # noqa: E712
        & laps_clean["PitInTime"].isna()
        & laps_clean["PitOutTime"].isna()
        & (laps_clean["TrackStatus"].astype(str) == "1")
    ]
    if len(preferred) != 1:
        raise RuntimeError("Reference NOR lap 5 is not a unique clean flying green lap.")
    return preferred.iloc[0]
