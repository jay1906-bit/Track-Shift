"""Extract visualization-only circuit polylines from cached FastF1 position data.

Does not change event, energy, ML, or strategy artifacts. Along-track science
still uses the frozen 1D Distance features; this file only stores X/Y outlines
so the frontend can draw a real circuit shape.
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

import fastf1  # noqa: E402
from trackshift import config  # noqa: E402
from trackshift.race_pipeline import RaceSpec  # noqa: E402

RACES = [
    RaceSpec(2026, "Australian Grand Prix", "australia"),
    RaceSpec(2026, "Chinese Grand Prix", "china"),
    RaceSpec(2026, "Japanese Grand Prix", "japan"),
    RaceSpec(2026, "Miami Grand Prix", "miami"),
    RaceSpec(2026, "Canadian Grand Prix", "canada"),
    RaceSpec(2026, "Monaco Grand Prix", "monaco"),
    RaceSpec(2026, "Austrian Grand Prix", "austria"),
    RaceSpec(2026, "Belgian Grand Prix", "belgium"),
    RaceSpec(2026, "Dutch Grand Prix", "netherlands"),
]

OUT_JSON = config.FRONTEND_V1_DIR / "track_outlines.json"
MAX_POINTS = 400
MIN_POINTS = 80


def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _is_flying_lap(lap: pd.Series) -> bool:
    try:
        n = float(lap["LapNumber"])
    except (TypeError, ValueError, KeyError):
        return False
    if not np.isfinite(n) or n < 2:
        return False
    for col in ("PitInTime", "PitOutTime"):
        if col in lap.index and not _is_na(lap[col]):
            return False
    return True


def _pos_xy(lap) -> pd.DataFrame | None:
    try:
        pos = lap.get_pos_data()
    except Exception:
        return None
    if pos is None or pos.empty or "X" not in pos.columns or "Y" not in pos.columns:
        return None
    if "Status" in pos.columns:
        on_track = pos["Status"].astype(str).str.lower().eq("ontrack")
        if on_track.any():
            pos = pos.loc[on_track]
    xy = pos[["X", "Y"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(xy) < MIN_POINTS:
        return None
    return xy


def pick_geometry_lap(session):
    """Prefer a flying lap with usable position telemetry. Visualization only."""
    try:
        fastest = session.laps.pick_fastest()
        if _is_flying_lap(fastest):
            xy = _pos_xy(fastest)
            if xy is not None:
                return fastest, xy, "fastest"
    except Exception:
        pass

    best = None
    n = 0
    for _, lap in session.laps.iterlaps():
        n += 1
        if not _is_flying_lap(lap):
            continue
        xy = _pos_xy(lap)
        if xy is None:
            continue
        score = len(xy)
        if best is None or score > best[0]:
            best = (score, lap, xy, f"fallback_{n}")
        if n >= 250:
            break
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    n = len(x)
    if n <= max_points:
        return x, y
    stride = int(np.ceil(n / max_points))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return x[idx], y[idx]


def polyline_from_xy(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    x, y = downsample_xy(x, y)
    seg = np.hypot(np.diff(x), np.diff(y))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total <= 0:
        return []
    frac = dist / total
    points = []
    for i in range(len(x)):
        points.append(
            [
                round(float(x[i]), 1),
                round(float(y[i]), 1),
                round(float(frac[i]), 5),
            ]
        )
    points[-1][2] = 1.0
    return points


def extract_race(spec: RaceSpec) -> dict | None:
    session = fastf1.get_session(spec.year, spec.event_name, spec.session_code)
    session.load(telemetry=True, weather=False, messages=False)
    lap, xy, source = pick_geometry_lap(session)
    if xy is None:
        return None
    x = xy["X"].to_numpy(float)
    y = xy["Y"].to_numpy(float)
    points = polyline_from_xy(x, y)
    if len(points) < MIN_POINTS:
        return None
    return {
        "race_id": spec.race_id,
        "circuit": spec.event_name,
        "source": "fastf1_pos_data",
        "kind": "telemetry_polyline",
        "source_lap": {
            "driver": str(lap["Driver"]),
            "lap": int(lap["LapNumber"]),
            "pick": source,
        },
        "n_points": len(points),
        "points": points,
        "note": (
            "Visualization only. Polyline is FastF1 position telemetry from one "
            "representative flying lap. Event distance / 400 m / strategy values "
            "are unchanged frozen features."
        ),
    }


def load_or_extract(force: bool = False) -> dict:
    config.FRONTEND_V1_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists() and not force:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        have = set((payload.get("races") or {}).keys())
        want = {spec.race_id for spec in RACES}
        if want <= have:
            return payload
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    races = {}
    for spec in RACES:
        print(f"geometry {spec.slug}...", flush=True)
        row = extract_race(spec)
        if row is None:
            print(f"  FAILED {spec.slug}", flush=True)
            continue
        races[spec.race_id] = row
        print(
            f"  {spec.slug} {row['source_lap']['pick']} "
            f"{row['source_lap']['driver']} L{row['source_lap']['lap']} "
            f"n={row['n_points']}",
            flush=True,
        )
    payload = {
        "version": "frontend_track_outlines_v1",
        "mode": "visualization_only",
        "races": races,
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")
    return payload


def main() -> None:
    force = "--force" in sys.argv
    payload = load_or_extract(force=force)
    missing = [spec.race_id for spec in RACES if spec.race_id not in payload.get("races", {})]
    if missing:
        raise SystemExit(f"Missing geometry for: {missing}")


if __name__ == "__main__":
    main()
