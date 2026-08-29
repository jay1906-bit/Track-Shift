"""Build the first multi-race 2026 overtake-event dataset.

Data collection, event detection, feature construction, dataset validation.
Nothing is trained, no model is selected, no split is created.

The Australia v4 event definition is applied unchanged to every race. The
orchestrator is verified against the stored v4 dataset by
scripts/check_australia_reproduction.py, which re-derives Australia from the
session and matches all 673 events, all event types and all labels.
"""

from __future__ import annotations

import argparse
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

from trackshift import config, manifest  # noqa: E402
from trackshift.events import EVENT_TYPES, EventParams  # noqa: E402
from trackshift.race_pipeline import (  # noqa: E402
    DETECTOR_VERSION,
    RaceSpec,
    process_race,
    write_race_artifacts,
)
from trackshift.validate import file_sha256  # noqa: E402

VERSION = "multirace_v1"
OUT_DIR = config.PROCESSED_DIR / "multirace_v1"
RACE_DIR = OUT_DIR / "races"
REPORT_DIR = config.OUTPUTS_DIR / "multirace_v1"

POOLED_CSV = OUT_DIR / "overtake_events_pooled_v1.csv"
ML_SAFE_CSV = OUT_DIR / "overtake_events_ml_safe_v1.csv"
AUDIT_CSV = OUT_DIR / "overtake_events_audit_pooled_v1.csv"
MANIFEST_CSV = OUT_DIR / "feature_manifest_v1.csv"
MANIFEST_JSON = REPORT_DIR / "feature_manifest_v1.json"
RACE_SUMMARY_JSON = REPORT_DIR / "race_processing_summary_v1.json"
QUALITY_JSON = REPORT_DIR / "pooled_dataset_quality_v1.json"
PER_RACE_QUALITY_JSON = REPORT_DIR / "per_race_quality_v1.json"
LEAKAGE_JSON = REPORT_DIR / "leakage_audit_v1.json"

# Rounds 1-12 of the 2026 season have been run. These nine are chosen for
# circuit diversity rather than count: parkland street, high-speed permanent,
# flowing technical, low-grip street, stop-start semi-street, tight street,
# short high-overtaking, long high-overtaking, and narrow banked.
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

FROZEN = {
    "gap_max_m": 100.0,
    "gap_min_m": 10.0,
    "min_persist_s": 3.0,
    "merge_gap_s": 2.0,
    "horizon_s": 30.0,
}

# Untouched by this phase. Hashed before and after to prove it.
ENERGY_PATHS = {
    "energy_v1": config.ENERGY_V1_CSV,
    "trackaware": config.ENERGY_TRACKAWARE_CSV,
    "zones": config.ZONE_TABLE_CSV,
    "lap_proxies": config.LAP_PROXIES_CSV,
    "energy_config": config.ENERGY_CONFIG_JSON,
    "raw_laps": config.RAW_LAPS_CSV,
    "clean_laps": config.CLEAN_LAPS_CSV,
    "events_v4": config.OVERTAKE_EVENTS_V4_CSV,
}

SCOREABLE = ["ON_TRACK_PASS", "CLOSE_INTERACTION_NO_PASS"]


def header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74, flush=True)


def list_2026_races() -> list[dict]:
    """Inspect FastF1's 2026 schedule. Race sessions only."""
    import fastf1

    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    sched = fastf1.get_event_schedule(2026, include_testing=False)
    rows = []
    for _, ev in sched.iterrows():
        name = str(ev.get("EventName", ""))
        kind = str(ev.get("EventFormat", ""))
        loc = str(ev.get("Location", ""))
        if "testing" in kind.lower() or "test" in name.lower():
            continue
        rows.append({
            "round": int(ev["RoundNumber"]) if pd.notna(ev.get("RoundNumber")) else None,
            "event_name": name,
            "location": loc,
            "format": kind,
        })
    return rows


def _num(series: pd.Series) -> dict:
    # bool has no quantile in numpy, and a rate is the useful summary anyway
    if pd.api.types.is_bool_dtype(series):
        series = series.astype("float64")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


# ----------------------------------------------------------------------
# stage 1 - process races
# ----------------------------------------------------------------------
def collect(races: list[RaceSpec], reuse: bool) -> tuple[list, list, dict]:
    frames, audits, summary = [], [], {"processed": [], "failed": []}
    for spec in races:
        cached = RACE_DIR / spec.slug / f"{spec.slug}_events.csv"
        quality_path = RACE_DIR / spec.slug / f"{spec.slug}_quality.json"
        if reuse and cached.exists() and quality_path.exists():
            q = json.loads(quality_path.read_text(encoding="utf-8"))
            length_q = q.get("circuit_length_measurement") or {}
            circ = float(
                length_q.get("circuit_length_m")
                or (q.get("circuit") or {}).get("circuit_length_m")
                or 0
            )
            med = float(
                length_q.get("median_lap_length_m")
                or (q.get("lap_length_m") or {}).get("median")
                or 0
            )
            ratio = (circ / med) if med > 0 else 0.0
            look_trunc = q.get("lookahead_truncated_rate")
            inflated = ratio > 1.08 or (
                look_trunc == 0.0 and int(q.get("n_events") or 0) > 100
            )
            if inflated:
                print(
                    f"\n  {spec.slug}: cached circuit length looks inflated "
                    f"(length/median={ratio:.3f}, lookahead_truncated={look_trunc}); reprocessing",
                    flush=True,
                )
            else:
                print(f"\n  {spec.slug}: reusing {cached.name}", flush=True)
                ev = pd.read_csv(cached, parse_dates=["event_start_time", "event_end_time",
                                                      "horizon_end_time"])
                ev["target_number"] = ev["target_number"].astype(str)
                au_path = RACE_DIR / spec.slug / f"{spec.slug}_events_audit.csv"
                frames.append(ev)
                if au_path.exists():
                    audits.append(pd.read_csv(au_path))
                summary["processed"].append(q)
                continue

        print(f"\n  {spec.slug}: processing {spec.event_name}", flush=True)
        t = time.time()
        art = process_race(spec, progress_every=250)
        if art.error:
            print(f"  {spec.slug}: FAILED at {art.stage}: {art.error}", flush=True)
            summary["failed"].append({
                "race_id": spec.race_id,
                "event_name": spec.event_name,
                "stage": art.stage,
                "error": art.error,
                "partial_output": False,
                "elapsed_s": round(time.time() - t, 1),
            })
            continue

        write_race_artifacts(art, RACE_DIR)
        art.quality["elapsed_s"] = round(time.time() - t, 1)
        art.quality["energy_config"] = art.energy_config
        (RACE_DIR / spec.slug / f"{spec.slug}_quality.json").write_text(
            json.dumps(art.quality, indent=2, default=str), encoding="utf-8"
        )
        frames.append(art.events)
        audits.append(art.audit)
        summary["processed"].append(art.quality)
    return frames, audits, summary


# ----------------------------------------------------------------------
# stage 2 - reports
# ----------------------------------------------------------------------
def leakage_audit(pooled: pd.DataFrame) -> dict:
    """Structural audit: does each column depend on information after t0?"""
    checks = []
    for row in manifest.MANIFEST.itertuples(index=False):
        checks.append({
            "column": row.column,
            "role": row.role,
            "depends_on_information_after_t0": not row.causal,
            "eligible_as_feature_t0": bool(row.causal),
            "reason": row.reason,
        })
    must_not_be_features = [
        "event_duration_s", "in_band_duration_s", "gap_min_in_episode_m", "event_end_time",
        "actual_swap_delay_s", "pass_after_horizon", "attacker_PositionChange_audit",
        "attacker_position", "target_position", "event_type", "overtake_success",
        "n_samples_in_episode", "window_crosses_start_finish",
    ]
    features = set(manifest.columns_for(manifest.FEATURE_T0))
    violations = sorted(features & set(must_not_be_features))
    non_causal_features = manifest.MANIFEST.loc[
        (manifest.MANIFEST["role"] == manifest.FEATURE_T0) & (~manifest.MANIFEST["causal"]),
        "column",
    ].tolist()

    # Empirical corroboration for the two position columns.
    sc = pooled.loc[pooled["event_type"].isin(SCOREABLE)].dropna(
        subset=["attacker_position", "target_position"]
    )
    ahead = sc["attacker_position"] < sc["target_position"]
    pos_evidence = {
        "n_checked": int(len(sc)),
        "spine_places_attacker_ahead_rate_positives": float(
            ahead[sc["overtake_success"] == 1].mean()
        ) if (sc["overtake_success"] == 1).any() else None,
        "spine_places_attacker_ahead_rate_negatives": float(
            ahead[sc["overtake_success"] == 0].mean()
        ) if (sc["overtake_success"] == 0).any() else None,
        "interpretation": (
            "At t0 the attacker is behind the target by construction, so a genuine at-t0 "
            "position must never place the attacker ahead. It does so far more often on "
            "positives than on negatives, which confirms the lap spine's Position is the "
            "end-of-lap classification and already contains the outcome."
        ),
    }
    return {
        "result": "PASS" if not violations and not non_causal_features else "FAIL",
        "n_columns_audited": int(len(checks)),
        "feature_t0_columns_depending_on_the_future": violations,
        "non_causal_columns_marked_feature_t0": non_causal_features,
        "excluded_future_or_episode_columns": sorted(
            manifest.columns_for(manifest.EPISODE_LEVEL)
            + [c for c in manifest.columns_for(manifest.AUDIT)
               if not bool(manifest.MANIFEST.loc[manifest.MANIFEST["column"] == c, "causal"].iloc[0])]
        ),
        "position_field_evidence": pos_evidence,
        "columns": checks,
    }


def feature_quality(pooled: pd.DataFrame) -> dict:
    features = manifest.columns_for(manifest.FEATURE_T0)
    scoreable = pooled.loc[pooled["event_type"].isin(SCOREABLE)]
    pos = scoreable.loc[scoreable["overtake_success"] == 1]
    neg = scoreable.loc[scoreable["overtake_success"] == 0]

    per_feature, constants = {}, []
    for col in features:
        s = scoreable[col]
        entry = {
            "dtype": str(pooled[col].dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100.0),
            "unique_count": int(s.nunique(dropna=False)),
            "portability": str(
                manifest.MANIFEST.loc[manifest.MANIFEST["column"] == col, "portability"].iloc[0]
            ),
            "constant_within_race": bool(
                manifest.MANIFEST.loc[
                    manifest.MANIFEST["column"] == col, "constant_within_race"
                ].iloc[0]
            ),
        }
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            entry["overall"] = _num(s)
            entry["by_class"] = {"positive": _num(pos[col]), "negative": _num(neg[col])}
        else:
            entry["top_values"] = {
                str(k): int(v) for k, v in s.value_counts(dropna=False).head(8).items()
            }
        entry["missing_pct_positive"] = float(pos[col].isna().mean() * 100.0) if len(pos) else None
        entry["missing_pct_negative"] = float(neg[col].isna().mean() * 100.0) if len(neg) else None
        if entry["unique_count"] <= 1:
            constants.append(col)
        per_feature[col] = entry

    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(scoreable[c]) and scoreable[c].nunique(dropna=False) > 1
    ]
    corr = scoreable[numeric].astype(float).corr()
    redundant = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.85:
                redundant.append({"a": a, "b": b, "pearson_r": round(float(r), 4)})
    redundant.sort(key=lambda d: -abs(d["pearson_r"]))
    return {
        "n_feature_t0": len(features),
        "constant_features_on_scoreable_set": constants,
        "constant_within_race_features": [
            c for c in features if per_feature[c]["constant_within_race"]
        ],
        "redundant_pairs_abs_r_ge_0_85": redundant,
        "per_feature": per_feature,
    }


def comparability(pooled: pd.DataFrame) -> dict:
    scoreable = pooled.loc[pooled["event_type"].isin(SCOREABLE)]
    cols = [
        "gap_at_start_m", "closing_speed_at_start_kmh", "relative_speed_same_position_kmh",
        "EstimatedEnergyIndex", "LookaheadCoveredM", "tyre_age", "attacker_speed_at_start_kmh",
    ]
    out = {}
    for race, grp in scoreable.groupby("race_id"):
        out[race] = {
            "n_scoreable": int(len(grp)),
            "positive_rate": float((grp["overtake_success"] == 1).mean()),
            "compound_mix": {
                str(k): round(float(v), 3)
                for k, v in grp["compound"].value_counts(normalize=True).items()
            },
            "lookahead_truncated_rate": float(grp["lookahead_truncated"].astype(bool).mean()),
            "closing_speed_missing_rate": float(grp["closing_speed_at_start_kmh"].isna().mean()),
            **{c: _num(grp[c]) for c in cols},
        }
    shifts = []
    for c in cols:
        med = {r: v[c].get("median") for r, v in out.items() if v[c].get("n")}
        vals = [m for m in med.values() if m is not None]
        if len(vals) > 1:
            lo, hi = min(vals), max(vals)
            shifts.append({
                "feature": c,
                "min_race_median": lo,
                "max_race_median": hi,
                "spread": hi - lo,
                "relative_spread": abs(hi - lo) / (abs(np.median(vals)) + 1e-9),
            })
    shifts.sort(key=lambda d: -d["relative_spread"])
    return {"per_race": out, "largest_median_shifts": shifts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse per-race CSVs already written")
    ap.add_argument("--races", nargs="*", default=None, help="slugs to process")
    args = ap.parse_args()

    for d in (OUT_DIR, RACE_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    races = RACES if not args.races else [r for r in RACES if r.slug in set(args.races)]
    params = EventParams()
    for name, want in FROZEN.items():
        got = getattr(params, name)
        if got != want:
            raise RuntimeError(f"frozen parameter {name} is {got}, expected {want}")

    header("FROZEN EVENT DEFINITION")
    for k, v in FROZEN.items():
        print(f"  {k} = {v}")
    print(f"  horizon_anchor = {params.horizon_anchor}   window = [t0, t0 + 30 s]")
    print(f"  detector = {DETECTOR_VERSION}")

    hashes_before = {n: file_sha256(p) for n, p in ENERGY_PATHS.items() if p.exists()}

    header("2026 RACE SCHEDULE")
    try:
        schedule = list_2026_races()
    except Exception as exc:  # noqa: BLE001
        schedule = []
        print(f"  could not load schedule: {exc}")
    available_names = {r["event_name"] for r in schedule}
    for row in schedule:
        mark = "SELECT" if any(r.event_name == row["event_name"] for r in races) else ""
        print(f"  R{row['round']:>2}  {row['event_name']:<32s} {row['location']:<20s} {mark}")
    missing_names = [r.event_name for r in races if r.event_name not in available_names]
    if missing_names and schedule:
        print(f"  WARNING: requested events not in schedule: {missing_names}")

    header(f"PROCESSING {len(races)} RACES")
    frames, audits, summary = collect(races, args.reuse)
    if not frames:
        print("no races produced events")
        return 1

    pooled = pd.concat(frames, ignore_index=True)
    pooled["target_number"] = pooled["target_number"].astype(str)
    audit_pooled = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()

    # ------------------------------------------------------------------
    header("MANIFEST")
    check = manifest.validate_against(pooled)
    print(f"  declared columns : {check['n_declared']}")
    print(f"  dataset columns  : {check['n_in_dataset']}")
    if check["in_dataset_but_undeclared"]:
        print(f"  UNDECLARED       : {check['in_dataset_but_undeclared']}")
    if check["declared_but_missing_from_dataset"]:
        print(f"  MISSING          : {check['declared_but_missing_from_dataset']}")
    print(f"  manifest valid   : {check['passed']}")
    if not check["passed"]:
        raise RuntimeError(f"manifest does not describe the dataset: {check}")
    role_counts = manifest.MANIFEST["role"].value_counts().to_dict()
    for role, n in role_counts.items():
        print(f"    {role:15s} {n}")

    pooled = pooled.loc[:, manifest.MANIFEST["column"].tolist()]
    pooled.to_csv(POOLED_CSV, index=False)
    if not audit_pooled.empty:
        audit_pooled.to_csv(AUDIT_CSV, index=False)
    manifest.MANIFEST.to_csv(MANIFEST_CSV, index=False)
    MANIFEST_JSON.write_text(
        json.dumps(manifest.MANIFEST.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    ml_safe = pooled.loc[
        pooled["event_type"].isin(SCOREABLE), manifest.ml_safe_columns()
    ].reset_index(drop=True)
    ml_safe.to_csv(ML_SAFE_CSV, index=False)
    print(f"  wrote {POOLED_CSV.name} ({len(pooled)} rows, {pooled.shape[1]} cols)")
    print(f"  wrote {ML_SAFE_CSV.name} ({len(ml_safe)} rows, {ml_safe.shape[1]} cols)")

    # ------------------------------------------------------------------
    header("DATASET")
    scoreable = pooled.loc[pooled["event_type"].isin(SCOREABLE)]
    n_pos = int((pooled["event_type"] == "ON_TRACK_PASS").sum())
    n_neg = int((pooled["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum())
    n_unc = int((pooled["event_type"] == "UNCERTAIN").sum())
    n_quar = int(len(pooled) - len(scoreable) - n_unc)
    print(f"  races           : {pooled['race_id'].nunique()}")
    print(f"  events          : {len(pooled)}")
    print(f"  labelled        : {len(scoreable)}")
    print(f"  positives       : {n_pos}")
    print(f"  negatives       : {n_neg}")
    print(f"  positive rate   : {n_pos / max(n_pos + n_neg, 1):.1%}")
    print(f"  uncertain       : {n_unc}")
    print(f"  quarantined     : {n_quar}")
    print("\n  event types:")
    for t in EVENT_TYPES:
        print(f"    {t:28s} {int((pooled['event_type'] == t).sum())}")

    per_race_rows = []
    for race, grp in pooled.groupby("race_id"):
        sc = grp.loc[grp["event_type"].isin(SCOREABLE)]
        p = int((grp["event_type"] == "ON_TRACK_PASS").sum())
        n = int((grp["event_type"] == "CLOSE_INTERACTION_NO_PASS").sum())
        per_race_rows.append({
            "race_id": race,
            "circuit": grp["circuit"].iloc[0],
            "events": int(len(grp)),
            "labelled": int(len(sc)),
            "positives": p,
            "negatives": n,
            "positive_rate": p / max(p + n, 1),
        })
    per_race = pd.DataFrame(per_race_rows).sort_values("race_id")
    print("\n  per race:")
    print("    " + per_race.to_string(index=False).replace("\n", "\n    "))

    pos_per_race = per_race["positives"]
    eff = {
        "total_rows": int(len(pooled)),
        "scoreable_rows": int(len(scoreable)),
        "distinct_race_groups": int(pooled["race_id"].nunique()),
        "distinct_pairs": int(pooled["pair_id"].nunique()),
        "distinct_race_lap_pair_clusters": int(pooled["cluster_id"].nunique()),
        "distinct_attackers": int(pooled["attacker"].nunique()),
        "events_with_a_reciprocal": int((pooled["reciprocal_event_id"].fillna("") != "").sum()),
        "scoreable_distinct_pairs": int(scoreable["pair_id"].nunique()),
        "scoreable_distinct_clusters": int(scoreable["cluster_id"].nunique()),
    }
    header("EFFECTIVE SAMPLE SIZE")
    for k, v in eff.items():
        print(f"  {k:38s} {v}")

    # ------------------------------------------------------------------
    header("LEAKAGE AUDIT")
    leak = leakage_audit(pooled)
    print(f"  result: {leak['result']}")
    print(f"  columns audited: {leak['n_columns_audited']}")
    print(f"  excluded future/episode columns: {len(leak['excluded_future_or_episode_columns'])}")
    for c in leak["excluded_future_or_episode_columns"]:
        print(f"    {c}")
    pe = leak["position_field_evidence"]
    if pe["spine_places_attacker_ahead_rate_positives"] is not None:
        print(
            f"  lap-spine Position places attacker ahead on "
            f"{pe['spine_places_attacker_ahead_rate_positives']:.1%} of positives vs "
            f"{pe['spine_places_attacker_ahead_rate_negatives']:.1%} of negatives "
            f"-> demoted to audit"
        )
    LEAKAGE_JSON.write_text(json.dumps(leak, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    header("FEATURE QUALITY")
    fq = feature_quality(pooled)
    print(f"  feature_t0 columns: {fq['n_feature_t0']}")
    print(f"  constant on the scoreable set: {fq['constant_features_on_scoreable_set']}")
    print(f"  constant within race: {fq['constant_within_race_features']}")
    worst = sorted(
        ((c, d["missing_pct"]) for c, d in fq["per_feature"].items() if d["missing_pct"] > 0),
        key=lambda t: -t[1],
    )
    print("  missingness (scoreable set):")
    for c, pct in worst:
        d = fq["per_feature"][c]
        print(f"    {c:42s} {pct:5.1f}%   pos {d['missing_pct_positive']:5.1f}%  "
              f"neg {d['missing_pct_negative']:5.1f}%")
    if not worst:
        print("    no missing values")
    print("  redundant pairs |r| >= 0.85:")
    for p in fq["redundant_pairs_abs_r_ge_0_85"][:15]:
        print(f"    {p['a']:40s} {p['b']:40s} r={p['pearson_r']:+.3f}")

    # ------------------------------------------------------------------
    header("RACE COMPARABILITY")
    comp = comparability(pooled)
    for s in comp["largest_median_shifts"][:6]:
        print(f"  {s['feature']:38s} race medians {s['min_race_median']:.3g} .. "
              f"{s['max_race_median']:.3g}")

    # ------------------------------------------------------------------
    header("DETECTOR HEALTH")
    health = []
    for q in summary["processed"]:
        lm = q.get("circuit_length_measurement") or {}
        hz = q.get("horizon_s_effective") or {}
        row = {
            "race_id": q["race_id"],
            "join_exact_match_rate": (q.get("join") or {}).get("exact_match_rate"),
            "race_distance_negative_steps": q.get("race_distance_negative_steps"),
            "circuit_length_m": (q.get("circuit") or {}).get("circuit_length_m"),
            "median_lap_length_m": (q.get("lap_length_m") or {}).get("median"),
            "n_outlier_laps": lm.get("n_outlier_laps"),
            "raw_max_over_median": lm.get("raw_max_over_median"),
            "horizon_fixed": hz.get("all_equal_to_horizon"),
            "min_same_pair_separation_s": q.get("min_same_pair_separation_s"),
            "n_unmapped_targets": (q.get("target_mapping") or {}).get("n_events_with_unmapped_target"),
            "energy_missing": q.get("energy_missing"),
        }
        health.append(row)
        print(
            f"  {row['race_id']:28s} length={row['circuit_length_m'] or 0:.0f}m  "
            f"outliers={row['n_outlier_laps']}  join={row['join_exact_match_rate']}  "
            f"neg_steps={row['race_distance_negative_steps']}"
        )
    if summary["failed"]:
        print("  failures:")
        for f in summary["failed"]:
            print(f"    {f['event_name']} at {f['stage']}: {f['error']}")
    frozen_ok, detector_ok = True, True
    for q in summary["processed"]:
        fp = q.get("frozen_parameters", {})
        ok = all(abs(float(fp.get(k, -1)) - v) < 1e-9 for k, v in FROZEN.items())
        frozen_ok &= ok
        print(f"  {q['race_id']:28s} params_ok={ok}  "
              f"horizon_fixed={q.get('horizon_s_effective', {}).get('all_equal_to_horizon')}")
    detector_ok = bool(
        (pooled["detector_version"] == DETECTOR_VERSION).all()
        and np.allclose(pooled["horizon_s_effective"].astype(float), FROZEN["horizon_s"])
    )
    print(f"  every race used the frozen definition: {frozen_ok}")
    print(f"  every event scored over exactly 30 s : {detector_ok}")

    hashes_after = {n: file_sha256(p) for n, p in ENERGY_PATHS.items() if p.exists()}
    energy_untouched = hashes_before == hashes_after
    print(f"  energy files and Australia v4 untouched: {energy_untouched}")

    # ------------------------------------------------------------------
    header("ML READINESS")
    n_races_zero_pos = int((per_race["positives"] == 0).sum())
    blockers = []
    if n_pos < 150:
        blockers.append(f"only {n_pos} positives; below the ~150 planning gate")
    if eff["distinct_race_groups"] < 5:
        blockers.append(f"only {eff['distinct_race_groups']} race groups")
    if leak["result"] != "PASS":
        blockers.append("leakage audit failed")
    if not frozen_ok or not detector_ok:
        blockers.append("frozen definition not applied uniformly")
    ready = not blockers
    print(f"  positives {n_pos}, negatives {n_neg}, race groups {eff['distinct_race_groups']}")
    print(f"  races with zero positives: {n_races_zero_pos}")
    print(f"  READY FOR ML MODEL SELECTION: {'YES' if ready else 'NO'}")
    for b in blockers:
        print(f"    blocker: {b}")

    # ------------------------------------------------------------------
    report = {
        "version": VERSION,
        "detector_version": DETECTOR_VERSION,
        "phase": "data collection, event detection, feature construction, dataset validation",
        "ml_trained": False,
        "ml_model_selected": False,
        "splits_created": False,
        "strategy_engine_built": False,
        "ui_built": False,
        "ablation_performed": False,
        "energy_model_modified": False,
        "energy_files_untouched": energy_untouched,
        "australia_v4_overwritten": False,
        "frozen_parameters": FROZEN,
        "frozen_parameters_applied_to_every_race": frozen_ok,
        "every_event_scored_over_exactly_30s": detector_ok,
        "schedule_2026": schedule,
        "requested_races": [{"slug": r.slug, "event_name": r.event_name, "race_id": r.race_id} for r in races],
        "totals": {
            "races": int(pooled["race_id"].nunique()),
            "events": int(len(pooled)),
            "labelled": int(len(scoreable)),
            "positives": n_pos,
            "negatives": n_neg,
            "positive_rate": n_pos / max(n_pos + n_neg, 1),
            "uncertain": n_unc,
            "quarantined": n_quar,
            "event_type_counts": {t: int((pooled["event_type"] == t).sum()) for t in EVENT_TYPES},
        },
        "per_race": per_race.to_dict(orient="records"),
        "positives_per_race": {
            "min": int(pos_per_race.min()),
            "median": float(pos_per_race.median()),
            "max": int(pos_per_race.max()),
            "n_races_with_zero_positives": n_races_zero_pos,
        },
        "min_labelled_events_per_race": int(per_race["labelled"].min()),
        "effective_sample_size": eff,
        "manifest": {
            "role_counts": role_counts,
            "validation": check,
            "path": str(MANIFEST_CSV),
        },
        "leakage_audit": {
            "result": leak["result"],
            "excluded_future_or_episode_columns": leak["excluded_future_or_episode_columns"],
            "position_field_evidence": leak["position_field_evidence"],
        },
        "feature_quality": fq,
        "race_comparability": comp,
        "detector_health": health,
        "ml_readiness": {
            "ready": ready,
            "blockers": blockers,
            "sample_size_band": (
                "under_150_positives" if n_pos < 150
                else "150_to_400_positives" if n_pos <= 400
                else "over_400_positives"
            ),
        },
        "files": {
            "pooled": str(POOLED_CSV),
            "ml_safe": str(ML_SAFE_CSV),
            "audit": str(AUDIT_CSV),
            "manifest_csv": str(MANIFEST_CSV),
            "manifest_json": str(MANIFEST_JSON),
            "per_race_dir": str(RACE_DIR),
        },
        "energy_hashes_before": hashes_before,
        "energy_hashes_after": hashes_after,
    }
    QUALITY_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    PER_RACE_QUALITY_JSON.write_text(
        json.dumps(summary["processed"], indent=2, default=str), encoding="utf-8"
    )
    RACE_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "requested": [{"race_id": r.race_id, "event_name": r.event_name} for r in races],
                "processed": [
                    {"race_id": q["race_id"], "event_name": q["event_name"],
                     "events": q.get("n_events"), "elapsed_s": q.get("elapsed_s")}
                    for q in summary["processed"]
                ],
                "failed": summary["failed"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {QUALITY_JSON}")
    print("\n  No ML. No model selection. No splits. No strategy. No UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
