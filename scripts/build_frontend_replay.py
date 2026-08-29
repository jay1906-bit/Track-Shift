"""Build a compact frontend replay JSON from frozen strategy + event tables.

Read-only over scientific artifacts. Does not retrain, rescore, or rewrite
event/energy/ML/strategy CSVs. Replay is sufficient; no GBDT pickle.

Usage:
    python scripts/build_frontend_replay.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.validate import file_sha256  # noqa: E402

VERSION = "frontend_v1"

EVENT_FIELDS = [
    "event_id",
    "race_id",
    "circuit",
    "split",
    "event_start_time",
    "attacker",
    "target",
    "event_start_lap",
    "gap_at_start_m",
    "attacker_speed_at_start_kmh",
    "target_speed_at_attacker_position_kmh",
    "relative_speed_same_position_kmh",
    "closing_speed_at_start_kmh",
    "spatial_time_gap_s",
    "spatial_alignment_ok",
    "closing_speed_available",
    "track_position_frac",
    "track_distance_at_start_m",
    "circuit_length_m",
    "distance_to_next_corner_m",
    "distance_to_next_straight_m",
    "compound",
    "tyre_age",
    "stint",
    "EstimatedEnergyIndex",
    "LookaheadDeployProxy",
    "LookaheadHarvestProxy",
    "LookaheadCoveredM",
    "lookahead_truncated",
    "zone_type",
    "p_hat",
    "strategy_action",
    "strategy_reason_code",
    "strategy_reason",
    "strategy_deploy_share",
    "strategy_lookahead_usable",
    "strategy_spend_compatible",
]

SAFE_JOIN_COLS = [
    "event_id",
    "attacker",
    "target",
    "event_start_lap",
    "gap_at_start_m",
    "attacker_speed_at_start_kmh",
    "target_speed_at_attacker_position_kmh",
    "relative_speed_same_position_kmh",
    "closing_speed_at_start_kmh",
    "spatial_time_gap_s",
    "spatial_alignment_ok",
    "closing_speed_available",
    "track_position_frac",
    "track_distance_at_start_m",
    "circuit_length_m",
    "distance_to_next_corner_m",
    "distance_to_next_straight_m",
    "compound",
    "tyre_age",
    "stint",
]

PROTECTED = [
    config.ML_SAFE_EVENTS_CSV,
    config.ML_OOF_PREDICTIONS_CSV,
    config.ML_LOCKED_TEST_PREDICTIONS_CSV,
    config.STRATEGY_V1_DIR / "development_replay.csv",
    config.STRATEGY_V1_DIR / "locked_test_replay.csv",
    config.ENERGY_V1_CSV,
    config.ENERGY_TRACKAWARE_CSV,
    config.OVERTAKE_EVENTS_V4_CSV,
]


def _json_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool,)):
        return bool(value)
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    if text.lower() in {"nan", "none", "nat", ""}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def _row_record(row: pd.Series) -> dict:
    record = {}
    for key in EVENT_FIELDS:
        record[key] = _json_value(row.get(key))
    success = _json_value(row.get("overtake_success_analysis_only"))
    if success is None:
        success = _json_value(row.get("overtake_success"))
    record["analysis_only_overtake_success"] = success
    return record


def _pick(frame: pd.DataFrame, **equals) -> pd.Series | None:
    subset = frame
    for key, value in equals.items():
        subset = subset[subset[key] == value]
    if subset.empty:
        return None
    ranked = subset.copy()
    if "lookahead_truncated" in ranked.columns:
        ranked["_trunc"] = ranked["lookahead_truncated"].astype(bool).astype(int)
    else:
        ranked["_trunc"] = 0
    if "p_hat" in ranked.columns:
        ranked["_p"] = pd.to_numeric(ranked["p_hat"], errors="coerce")
    else:
        ranked["_p"] = 0.0
    ranked = ranked.sort_values(["_trunc", "_p"], ascending=[True, False])
    return ranked.iloc[0]


def _pick_nearest_below_gate(frame: pd.DataFrame, race_id: str) -> pd.Series | None:
    subset = frame[
        (frame["race_id"] == race_id)
        & (frame["strategy_reason_code"] == "p_hat_below_gate")
    ]
    if subset.empty:
        return None
    p = pd.to_numeric(subset["p_hat"], errors="coerce")
    subset = subset.assign(_p=p)
    subset = subset[subset["_p"].notna() & (subset["_p"] < 0.45)]
    if subset.empty:
        return None
    subset = subset.sort_values("_p", ascending=False)
    return subset.iloc[0]


def _pick_energy_floor_high_p(frame: pd.DataFrame) -> pd.Series | None:
    subset = frame[frame["strategy_reason_code"] == "energy_floor"].copy()
    if subset.empty:
        return None
    subset["_p"] = pd.to_numeric(subset["p_hat"], errors="coerce")
    subset = subset.sort_values("_p", ascending=False)
    return subset.iloc[0]


def _pick_spend_context_high_p(frame: pd.DataFrame) -> pd.Series | None:
    subset = frame[frame["strategy_reason_code"] == "spend_context"].copy()
    if subset.empty:
        return None
    subset["_p"] = pd.to_numeric(subset["p_hat"], errors="coerce")
    subset = subset.sort_values("_p", ascending=False)
    return subset.iloc[0]


def _pick_truncated(frame: pd.DataFrame) -> pd.Series | None:
    truncated = frame[frame["lookahead_truncated"].astype(str).str.lower().isin(["true", "1"])]
    if truncated.empty and "lookahead_truncated" in frame.columns:
        truncated = frame[frame["lookahead_truncated"] == True]  # noqa: E712
    if truncated.empty:
        return None
    truncated = truncated.copy()
    truncated["_p"] = pd.to_numeric(truncated["p_hat"], errors="coerce")
    return truncated.sort_values("_p", ascending=False).iloc[0]


def curated_slots(frame: pd.DataFrame) -> list[dict]:
    picks = [
        ("australia_attack", "ATTACK · Australia", _pick(frame, race_id="australia_2026_R", strategy_action="ATTACK")),
        ("australia_save_p", "SAVE · p-gate", _pick_nearest_below_gate(frame, "australia_2026_R")),
        ("energy_floor", "SAVE · Energy", _pick_energy_floor_high_p(frame)),
        ("spend_context", "SAVE · 400m", _pick_spend_context_high_p(frame)),
        ("belgium_attack", "ATTACK · Belgium locked", _pick(frame, race_id="belgium_2026_R", strategy_action="ATTACK")),
        ("miami_save", "SAVE · Miami locked", _pick(frame, race_id="miami_2026_R", strategy_action="SAVE")),
        ("monaco_save", "SAVE · Monaco", _pick(frame, race_id="monaco_2026_R", strategy_action="SAVE")),
        ("japan_attack", "ATTACK · Japan", _pick(frame, race_id="japan_2026_R", strategy_action="ATTACK")),
        ("truncated", "ATTACK · truncated 400m", _pick_truncated(frame)),
    ]
    out = []
    seen = set()
    for slot_id, label, row in picks:
        if row is None:
            continue
        event_id = str(row["event_id"])
        if event_id in seen:
            continue
        seen.add(event_id)
        out.append(
            {
                "id": slot_id,
                "label": label,
                "event_id": event_id,
                "race_id": str(row["race_id"]),
                "strategy_action": str(row["strategy_action"]),
                "strategy_reason_code": str(row["strategy_reason_code"]),
            }
        )
    return out


def _metrics_payload() -> dict:
    ml_path = config.ML_SELECTION_DIR / "ml_selection_summary.json"
    st_path = config.STRATEGY_V1_DIR / "strategy_summary.json"
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    st = json.loads(st_path.read_text(encoding="utf-8"))
    loro = ml["loro_summaries"]
    gbdt = loro["gbdt"]
    baseline = loro["baseline"]
    logistic = loro["logistic"]
    locked = ml["locked_test"]
    hp = ml.get("hyperparameters", {}).get("gbdt", {})
    dataset = ml.get("dataset", {})
    return {
        "model": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_depth": hp.get("max_depth", 3),
            "max_iter": hp.get("max_iter", 150),
            "learning_rate": hp.get("learning_rate", 0.05),
            "seed": hp.get("random_state", 42),
        },
        "dataset": {
            "n_labelled": dataset.get("n_rows"),
            "n_pos": dataset.get("n_pos"),
            "n_neg": dataset.get("n_neg"),
            "n_races": dataset.get("n_races"),
            "dev_n": dataset.get("dev_n"),
            "test_n": dataset.get("test_n"),
        },
        "note": (
            "These are frozen validation results. This demo is historical replay, "
            "not live inference. PR-AUC / Brier / log loss are the relevant metrics. "
            "There is no accuracy score on this probability model."
        ),
        "model_comparison": [
            {
                "id": "baseline",
                "name": "Domain baseline",
                "pr_auc": baseline["macro_pr_auc"],
                "brier": baseline["macro_brier"],
                "selected": False,
            },
            {
                "id": "logistic",
                "name": "Full L2 logistic",
                "pr_auc": logistic["macro_pr_auc"],
                "brier": logistic["macro_brier"],
                "selected": False,
            },
            {
                "id": "gbdt",
                "name": "Shallow GBDT",
                "pr_auc": gbdt["macro_pr_auc"],
                "brier": gbdt["macro_brier"],
                "selected": True,
            },
        ],
        "development_loro": {
            "split_note": "Leave-one-race-out on the 7 development races. Belgium and Miami were locked.",
            "pr_auc": gbdt["macro_pr_auc"],
            "brier": gbdt["macro_brier"],
            "log_loss": gbdt["macro_log_loss"],
            "roc_auc": gbdt["macro_roc_auc"],
        },
        "locked_test": {
            "split_note": "Belgium and Miami were unused until final confirmation.",
            "belgium_pr_auc": locked["per_race"]["belgium_2026_R"]["pr_auc"],
            "miami_pr_auc": locked["per_race"]["miami_2026_R"]["pr_auc"],
            "pooled_brier": locked["pooled"]["brier"],
            "pooled_pr_auc": locked["pooled"]["pr_auc"],
            "pooled_log_loss": locked["pooled"]["log_loss"],
        },
        "strategy": {
            "defend_implemented": False,
            "development_attack_rate": st["development"]["attack_rate"],
            "development_save_rate": st["development"]["save_rate"],
            "locked_test_attack_rate": st["locked_test"]["attack_rate"],
            "locked_test_save_rate": st["locked_test"]["save_rate"],
            "ablation_attack_rate": {
                "p_only": st["ablation"]["A"]["attack_rate"],
                "p_energy": st["ablation"]["B"]["attack_rate"],
                "p_energy_400m": st["ablation"]["C"]["attack_rate"],
            },
            "ablation_note": (
                "Immediate 400 m track context changes the ATTACK rate. "
                "This does not prove strategy optimality."
            ),
        },
        "strategy_config": st["config"],
    }


def _load_track_outlines() -> dict:
    """Load FastF1 X/Y polylines. Extract from cache if the artifact is missing."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from extract_frontend_geometry import load_or_extract

    payload = load_or_extract(force=False)
    return payload.get("races") or {}


def _load_circuits(merged: pd.DataFrame) -> dict:
    """Copy existing per-race zone tables and attach visualization-only polylines."""
    races_root = config.PROCESSED_DIR / "multirace_v1" / "races"
    outlines = _load_track_outlines()
    circuits = {}
    for race_id in sorted(merged["race_id"].astype(str).unique()):
        slug = str(race_id).split("_")[0]
        path = races_root / slug / f"{slug}_circuit_zones.csv"
        length = None
        subset = merged.loc[merged["race_id"].astype(str) == race_id, "circuit_length_m"]
        numeric = pd.to_numeric(subset, errors="coerce").dropna()
        if len(numeric):
            length = float(numeric.iloc[0])
        zones = []
        if path.exists():
            table = pd.read_csv(path)
            for _, zrow in table.iterrows():
                start_m = _json_value(zrow.get("start_m"))
                end_m = _json_value(zrow.get("end_m"))
                if start_m is None or end_m is None:
                    continue
                zones.append(
                    {
                        "zone_id": _json_value(zrow.get("zone_id")),
                        "zone_type": _json_value(zrow.get("zone_type")),
                        "start_m": float(start_m),
                        "end_m": float(end_m),
                        "length_m": _json_value(zrow.get("length_m")),
                    }
                )
            if length is None and zones:
                length = float(max(z["end_m"] for z in zones))
        outline = outlines.get(race_id) or {}
        points = outline.get("points") or []
        if points:
            schematic = "telemetry_polyline"
        elif zones and length:
            schematic = "zone_fallback"
        else:
            schematic = "linear_fallback"
        circuits[race_id] = {
            "race_id": race_id,
            "circuit_length_m": length,
            "zones": zones,
            "schematic": schematic,
            "geometry_source": outline.get("source"),
            "geometry_kind": outline.get("kind"),
            "geometry_lap": outline.get("source_lap"),
            "geometry_note": outline.get("note"),
            "polyline": points,
        }
    return circuits


def main() -> None:
    hashes_before = {str(path): file_sha256(path) for path in PROTECTED if path.exists()}

    dev = pd.read_csv(config.STRATEGY_V1_DIR / "development_replay.csv")
    locked = pd.read_csv(config.STRATEGY_V1_DIR / "locked_test_replay.csv")
    replay = pd.concat([dev, locked], ignore_index=True)
    safe = pd.read_csv(config.ML_SAFE_EVENTS_CSV, usecols=lambda c: c in SAFE_JOIN_COLS)
    merged = replay.merge(safe, on="event_id", how="left", validate="one_to_one")
    if merged["attacker"].isna().any():
        missing = int(merged["attacker"].isna().sum())
        raise SystemExit(f"Join failed: {missing} strategy rows missing ml_safe attacker/target.")

    events = [_row_record(row) for _, row in merged.iterrows()]
    curated = curated_slots(merged)
    races = (
        merged[["race_id", "circuit", "split"]]
        .drop_duplicates()
        .sort_values("race_id")
        .to_dict(orient="records")
    )
    for race in races:
        race["n_events"] = int((merged["race_id"] == race["race_id"]).sum())

    smoke = next((item for item in curated if item["id"] == "australia_attack"), curated[0])
    smoke_row = merged.loc[merged["event_id"] == smoke["event_id"]].iloc[0]

    payload = {
        "version": VERSION,
        "mode": "historical_replay",
        "live_inference": False,
        "title": "ApexIQ",
        "project": "TrackShift",
        "subtitle": "Historical overtake opportunity replay",
        "honesty": {
            "p_hat": "P(overtake within 30s)",
            "energy": "Simulated Energy Index",
            "context": "Immediate 400m Track Context",
            "mode_label": "Historical Replay",
            "not": [
                "live inference",
                "battery percentage",
                "ERS / FIA energy",
                "full-race energy forecast",
                "predicted finishing position",
                "DEFEND",
                "accuracy score",
            ],
        },
        "races": [{k: _json_value(v) for k, v in race.items()} for race in races],
        "circuits": _load_circuits(merged),
        "curated": curated,
        "events": events,
        "metrics": _metrics_payload(),
        "smoke_test": {
            "event_id": str(smoke_row["event_id"]),
            "p_hat": _json_value(smoke_row["p_hat"]),
            "strategy_action": _json_value(smoke_row["strategy_action"]),
            "strategy_reason": _json_value(smoke_row["strategy_reason"]),
            "EstimatedEnergyIndex": _json_value(smoke_row["EstimatedEnergyIndex"]),
            "zone_type": _json_value(smoke_row["zone_type"]),
            "strategy_deploy_share": _json_value(smoke_row["strategy_deploy_share"]),
        },
        "n_events": len(events),
    }

    out_dir = config.FRONTEND_V1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = config.FRONTEND_REPLAY_JSON
    out_json.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    public_dir = ROOT / "frontend" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_json, public_dir / "replay.json")

    hashes_after = {str(path): file_sha256(path) for path in PROTECTED if path.exists()}
    if hashes_before != hashes_after:
        raise SystemExit("Protected scientific artifacts changed during frontend JSON build.")

    hash_path = out_dir / "protected_hashes.json"
    hash_path.write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")

    print(f"Wrote {out_json} ({out_json.stat().st_size} bytes, {len(events)} events)")
    print(f"Copied {public_dir / 'replay.json'}")
    print(f"Smoke event: {payload['smoke_test']['event_id']}")
    print(f"  p_hat={payload['smoke_test']['p_hat']}")
    print(f"  action={payload['smoke_test']['strategy_action']}")
    print("Curated:")
    for item in curated:
        print(f"  {item['id']}: {item['event_id']} ({item['strategy_action']})")


if __name__ == "__main__":
    main()
