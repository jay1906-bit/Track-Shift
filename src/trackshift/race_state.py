"""Race-state freeze / yellow reduction from lap-level TrackStatus and pit times."""

from __future__ import annotations

from . import config


def parse_track_status(track_status) -> str:
    return "".join(ch for ch in str(track_status) if ch.isdigit())


def is_freeze_status(track_status) -> bool:
    codes = set(parse_track_status(track_status))
    return bool(codes & config.FREEZE_TRACK_CODES)


def is_yellow_status(track_status) -> bool:
    if is_freeze_status(track_status):
        return False
    return config.YELLOW_TRACK_CODE in parse_track_status(track_status)


def is_green_only(track_status) -> bool:
    return parse_track_status(track_status) == "1"


def is_box_lap(pit_in, pit_out) -> bool:
    return pd_notna(pit_in) or pd_notna(pit_out)


def pd_notna(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:  # NaN
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    return text not in {"", "nan", "nat", "none"}


def lap_race_state(track_status, pit_in, pit_out) -> dict:
    """v1: lap-level TrackStatus from the cleaned CSV plus pit timestamps."""
    freeze = is_freeze_status(track_status) or is_box_lap(pit_in, pit_out)
    yellow = (not freeze) and is_yellow_status(track_status)
    if freeze:
        factor = 0.0
        label = "freeze"
    elif yellow:
        factor = config.YELLOW_FACTOR
        label = "yellow"
    else:
        factor = 1.0
        label = "normal"
    return {
        "freeze_flag": bool(freeze),
        "yellow_flag": bool(yellow),
        "race_state_factor": float(factor),
        "race_state_label": label,
    }
