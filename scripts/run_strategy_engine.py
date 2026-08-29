"""Replay the frozen ATTACK/SAVE engine on saved GBDT probabilities.

Does not retrain the GBDT, modify events, or recompute energy/lookahead.

Usage:
    python scripts/run_strategy_engine.py
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
from trackshift.strategy import (  # noqa: E402
    ABLATION_FULL,
    ABLATION_P_ENERGY,
    ABLATION_P_ONLY,
    ACTION_ATTACK,
    ACTION_INVALID,
    ACTION_SAVE,
    FROZEN_STRATEGY_CONFIG,
    StrategyConfig,
    apply_strategy,
)
from trackshift.validate import file_sha256  # noqa: E402

VERSION = "strategy_v1"
OUT = config.STRATEGY_V1_DIR
DEV_RACES = [
    "australia_2026_R",
    "china_2026_R",
    "japan_2026_R",
    "canada_2026_R",
    "monaco_2026_R",
    "austria_2026_R",
    "netherlands_2026_R",
]
TEST_RACES = ["belgium_2026_R", "miami_2026_R"]
PROTECTED_PATHS = {
    "energy_v1": config.ENERGY_V1_CSV,
    "trackaware": config.ENERGY_TRACKAWARE_CSV,
    "zones": config.ZONE_TABLE_CSV,
    "energy_config": config.ENERGY_CONFIG_JSON,
    "events_v4": config.OVERTAKE_EVENTS_V4_CSV,
    "ml_safe": config.ML_SAFE_EVENTS_CSV,
    "pooled": config.POOLED_EVENTS_CSV,
    "oof": config.ML_OOF_PREDICTIONS_CSV,
    "locked_test": config.ML_LOCKED_TEST_PREDICTIONS_CSV,
}

# Precommitted candidate grid. Selection uses process metrics, not labels.
P_CANDIDATES = (0.35, 0.45, 0.55)
ENERGY_CANDIDATES = (0.05, 0.10, 0.20)
SHARE_CANDIDATES = (0.35, 0.40, 0.50)


def header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74, flush=True)


def hashes() -> dict[str, str]:
    out = {}
    for name, path in PROTECTED_PATHS.items():
        if path.exists():
            out[name] = file_sha256(path)
    return out


def load_gbdt_oof() -> pd.DataFrame:
    oof = pd.read_csv(config.ML_OOF_PREDICTIONS_CSV)
    gbdt = oof.loc[oof["model"].astype(str) == "gbdt"].copy()
    if gbdt.empty:
        raise RuntimeError("No GBDT rows in OOF predictions.")
    return gbdt


def load_locked_test() -> pd.DataFrame:
    return pd.read_csv(config.ML_LOCKED_TEST_PREDICTIONS_CSV)


def load_features() -> pd.DataFrame:
    safe = pd.read_csv(config.ML_SAFE_EVENTS_CSV)
    pooled = pd.read_csv(
        config.POOLED_EVENTS_CSV,
        usecols=["event_id", "event_start_time"],
    )
    return safe.merge(pooled, on="event_id", how="left")


def attach(preds: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    keep = features.loc[
        :,
        [
            "event_id",
            "race_id",
            "circuit",
            "event_start_time",
            "EstimatedEnergyIndex",
            "LookaheadDeployProxy",
            "LookaheadHarvestProxy",
            "LookaheadCoveredM",
            "lookahead_truncated",
            "zone_type",
            "overtake_success",
        ],
    ].copy()
    p = preds.loc[:, ["event_id", "p_hat"]].copy()
    out = p.merge(keep, on="event_id", how="left")
    if out["EstimatedEnergyIndex"].isna().any():
        raise RuntimeError("Feature join dropped energy for some events.")
    return out


def process_summary(replay: pd.DataFrame, split: str) -> dict:
    action = replay["strategy_action"].astype(str)
    n = len(replay)
    n_attack = int((action == ACTION_ATTACK).sum())
    n_save = int((action == ACTION_SAVE).sum())
    n_invalid = int((action == ACTION_INVALID).sum())
    energy_veto = int((replay["strategy_reason_code"] == "energy_floor").sum())
    energy_missing = int((replay["strategy_reason_code"] == "energy_missing").sum())
    look_veto = int((replay["strategy_reason_code"] == "lookahead_unusable").sum())
    p_veto = int((replay["strategy_reason_code"] == "p_hat_below_gate").sum())
    spend_veto = int((replay["strategy_reason_code"] == "spend_context").sum())

    def by_action(col: str, fn) -> dict:
        out = {}
        for act in (ACTION_ATTACK, ACTION_SAVE):
            part = pd.to_numeric(replay.loc[action == act, col], errors="coerce")
            out[act] = float(fn(part)) if part.notna().any() else float("nan")
        return out

    zone_by_action = {}
    for act in (ACTION_ATTACK, ACTION_SAVE):
        sub = replay.loc[action == act, "zone_type"].astype(str)
        zone_by_action[act] = sub.value_counts(normalize=True).round(4).to_dict()

    per_race = []
    for race_id, grp in replay.groupby("race_id", sort=True):
        a = grp["strategy_action"].astype(str)
        per_race.append(
            {
                "race_id": str(race_id),
                "circuit": str(grp["circuit"].iloc[0]) if "circuit" in grp else "",
                "n": int(len(grp)),
                "n_attack": int((a == ACTION_ATTACK).sum()),
                "n_save": int((a == ACTION_SAVE).sum()),
                "attack_rate": float((a == ACTION_ATTACK).mean()),
                "median_p_hat_attack": float(
                    pd.to_numeric(grp.loc[a == ACTION_ATTACK, "p_hat"], errors="coerce").median()
                )
                if (a == ACTION_ATTACK).any()
                else float("nan"),
                "median_p_hat_save": float(
                    pd.to_numeric(grp.loc[a == ACTION_SAVE, "p_hat"], errors="coerce").median()
                )
                if (a == ACTION_SAVE).any()
                else float("nan"),
                "median_energy_attack": float(
                    pd.to_numeric(
                        grp.loc[a == ACTION_ATTACK, "EstimatedEnergyIndex"], errors="coerce"
                    ).median()
                )
                if (a == ACTION_ATTACK).any()
                else float("nan"),
            }
        )

    energy_vetoed = replay.loc[replay["strategy_reason_code"] == "energy_floor"]
    attacked = replay.loc[action == ACTION_ATTACK]
    p_pass = pd.to_numeric(replay["p_hat"], errors="coerce") >= FROZEN_STRATEGY_CONFIG.p_hat_min
    energy_veto_among_p_passers = int(
        ((replay["strategy_reason_code"] == "energy_floor") & p_pass).sum()
    )
    spend_veto_among_p_passers = int(
        ((replay["strategy_reason_code"] == "spend_context") & p_pass).sum()
    )
    return {
        "split": split,
        "n": n,
        "n_attack": n_attack,
        "n_save": n_save,
        "n_invalid": n_invalid,
        "attack_rate": float(n_attack / n) if n else float("nan"),
        "save_rate": float(n_save / n) if n else float("nan"),
        "reason_counts": replay["strategy_reason_code"].value_counts().to_dict(),
        "energy_floor_vetoes": energy_veto,
        "energy_missing_vetoes": energy_missing,
        "lookahead_unusable_vetoes": look_veto,
        "p_hat_below_vetoes": p_veto,
        "spend_context_vetoes": spend_veto,
        "n_p_gate_passers": int(p_pass.sum()),
        "energy_floor_vetoes_among_p_passers": energy_veto_among_p_passers,
        "spend_context_vetoes_among_p_passers": spend_veto_among_p_passers,
        "mean_p_hat_by_action": by_action("p_hat", lambda s: s.mean()),
        "median_p_hat_by_action": by_action("p_hat", lambda s: s.median()),
        "mean_energy_by_action": by_action("EstimatedEnergyIndex", lambda s: s.mean()),
        "median_energy_by_action": by_action("EstimatedEnergyIndex", lambda s: s.median()),
        "median_deploy_share_by_action": by_action("strategy_deploy_share", lambda s: s.median()),
        "median_energy_of_energy_vetoes": float(
            pd.to_numeric(energy_vetoed["EstimatedEnergyIndex"], errors="coerce").median()
        )
        if len(energy_vetoed)
        else float("nan"),
        "median_energy_of_attacks": float(
            pd.to_numeric(attacked["EstimatedEnergyIndex"], errors="coerce").median()
        )
        if len(attacked)
        else float("nan"),
        "attack_p_hat_gt_save": bool(
            by_action("p_hat", lambda s: s.median())[ACTION_ATTACK]
            > by_action("p_hat", lambda s: s.median())[ACTION_SAVE]
        )
        if n_attack and n_save
        else False,
        "zone_share_by_action": zone_by_action,
        "per_race": per_race,
        "analysis_only_pass_rate_by_action": {
            act: float(
                pd.to_numeric(
                    replay.loc[action == act, "overtake_success"], errors="coerce"
                ).mean()
            )
            if (action == act).any()
            else float("nan")
            for act in (ACTION_ATTACK, ACTION_SAVE)
        },
    }


def grid_row(dev: pd.DataFrame, cfg: StrategyConfig) -> dict:
    replay = apply_strategy(dev, cfg, ablation=ABLATION_FULL)
    s = process_summary(replay, "grid")
    return {
        "p_hat_min": cfg.p_hat_min,
        "energy_floor": cfg.energy_floor,
        "deploy_share_min": cfg.deploy_share_min,
        "attack_rate": s["attack_rate"],
        "n_attack": s["n_attack"],
        "median_p_hat_attack": s["median_p_hat_by_action"][ACTION_ATTACK],
        "median_p_hat_save": s["median_p_hat_by_action"][ACTION_SAVE],
        "energy_floor_vetoes": s["energy_floor_vetoes"],
        "spend_context_vetoes": s["spend_context_vetoes"],
        "p_hat_below_vetoes": s["p_hat_below_vetoes"],
        "lookahead_unusable_vetoes": s["lookahead_unusable_vetoes"],
    }


def describe_distributions(dev: pd.DataFrame) -> dict:
    p = pd.to_numeric(dev["p_hat"], errors="coerce")
    e = pd.to_numeric(dev["EstimatedEnergyIndex"], errors="coerce")
    dep = pd.to_numeric(dev["LookaheadDeployProxy"], errors="coerce")
    har = pd.to_numeric(dev["LookaheadHarvestProxy"], errors="coerce")
    share = dep / (dep + har)
    trunc = dev["lookahead_truncated"]
    if trunc.dtype == object:
        trunc = trunc.astype(str).str.lower().isin(["true", "1", "yes"])
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90]
    reliability = []
    for t in P_CANDIDATES:
        mask = p >= t
        reliability.append(
            {
                "p_hat_min": t,
                "n": int(mask.sum()),
                "share_of_events": float(mask.mean()),
                "observed_pass_rate_analysis_only": float(
                    pd.to_numeric(dev.loc[mask, "overtake_success"], errors="coerce").mean()
                ),
            }
        )
    return {
        "n": int(len(dev)),
        "p_hat_quantiles": {str(k): float(v) for k, v in p.quantile(q).items()},
        "p_hat_mean": float(p.mean()),
        "p_hat_median": float(p.median()),
        "energy_quantiles": {str(k): float(v) for k, v in e.quantile(q).items()},
        "energy_frac_le_0_05": float((e <= 0.05).mean()),
        "energy_frac_le_0_10": float((e <= 0.10).mean()),
        "energy_frac_le_0_20": float((e <= 0.20).mean()),
        "deploy_share_quantiles": {str(k): float(v) for k, v in share.quantile(q).items()},
        "zone_counts": dev["zone_type"].astype(str).value_counts().to_dict(),
        "truncated_rate": float(trunc.mean()),
        "truncated_n": int(trunc.sum()),
        "p_gate_coverage": reliability,
    }


def replay_table(frame: pd.DataFrame, split: str, ablation: str, cfg: StrategyConfig) -> pd.DataFrame:
    out = apply_strategy(frame, cfg, ablation=ablation)
    cols = [
        "event_id",
        "race_id",
        "circuit",
        "event_start_time",
        "p_hat",
        "EstimatedEnergyIndex",
        "LookaheadDeployProxy",
        "LookaheadHarvestProxy",
        "LookaheadCoveredM",
        "lookahead_truncated",
        "zone_type",
        "strategy_action",
        "strategy_reason_code",
        "strategy_reason",
        "strategy_deploy_share",
        "strategy_lookahead_usable",
        "strategy_spend_compatible",
        "strategy_ablation",
        "overtake_success",
    ]
    keep = [c for c in cols if c in out.columns]
    table = out.loc[:, keep].copy()
    table["split"] = split
    table["overtake_success_analysis_only"] = table["overtake_success"]
    table = table.drop(columns=["overtake_success"])
    return table


def write_report(
    *,
    dist: dict,
    grid: pd.DataFrame,
    selected: dict,
    dev_summary: dict,
    test_summary: dict,
    ablation: dict,
    hashes_before: dict,
    hashes_after: dict,
    tests_ok: bool,
) -> str:
    cfg = FROZEN_STRATEGY_CONFIG
    per_race_lines = [
        "| Race | n | ATTACK | SAVE | ATTACK % | median p ATTACK | median p SAVE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev_summary["per_race"]:
        per_race_lines.append(
            f"| {row['race_id']} | {row['n']} | {row['n_attack']} | {row['n_save']} | "
            f"{100*row['attack_rate']:.1f}% | {row['median_p_hat_attack']:.3f} | "
            f"{row['median_p_hat_save']:.3f} |"
        )
    test_lines = [
        "| Race | n | ATTACK | SAVE | ATTACK % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in test_summary["per_race"]:
        test_lines.append(
            f"| {row['race_id']} | {row['n']} | {row['n_attack']} | {row['n_save']} | "
            f"{100*row['attack_rate']:.1f}% |"
        )
    ab_lines = [
        "| Ablation | ATTACK % | n ATTACK | energy vetoes among p-passers | spend vetoes among p-passers | median p ATTACK |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("A", "A: P-only"),
        ("B", "B: P + energy"),
        ("C", "C: P + energy + 400 m context"),
    ]:
        s = ablation[key]
        ab_lines.append(
            f"| {label} | {100*s['attack_rate']:.1f}% | {s['n_attack']} | "
            f"{s['energy_floor_vetoes_among_p_passers']} | "
            f"{s['spend_context_vetoes_among_p_passers']} | "
            f"{s['median_p_hat_by_action'][ACTION_ATTACK]:.3f} |"
        )
    hashes_ok = hashes_before == hashes_after
    md = f"""# TrackShift strategy engine v1

**Version:** `{VERSION}`  
**Frozen ML:** GBDT OOF + locked-test probabilities from `outputs/ml_selection_v1/`  
**DEFEND:** not implemented

Protected energy/event/ML artifacts hashed before and after: **{'UNCHANGED' if hashes_ok else 'CHANGED'}**.

---

### STRATEGY SPECIFICATION

Two actions only: **ATTACK** and **SAVE**.

1. Invalid `p_hat` (missing or outside [0, 1]) → `INVALID` (no action).
2. Hard infeasibility → SAVE
   - missing energy
   - `EstimatedEnergyIndex` < `{cfg.energy_floor}` (resource floor)
   - lookahead truncated/unusable **and** energy < `{cfg.energy_adequate_if_lookahead_unusable}`
3. Opportunity gate → SAVE if `p_hat` < `{cfg.p_hat_min}`
4. Immediate spend window → SAVE unless the next 400 m is a deploy-compatible context:
   - `zone_type == straight`, **or**
   - deploy share = LookaheadDeployProxy / (Deploy + Harvest) ≥ `{cfg.deploy_share_min}`
5. Otherwise ATTACK.

High raw deploy does **not** force SAVE. Gap and closing speed are not strategy gates.

---

### THRESHOLDS

Selected on the 7 development races only. Belgium and Miami unused.

| Parameter | Value | Development-only rationale |
|---|---:|---|
| `p_hat_min` | {cfg.p_hat_min} | Middle of the precommitted set {{0.35, 0.45, 0.55}}. OOF `p_hat` median is 0.12; 0.45 sits between the 75th (0.38) and 80th (0.48) percentiles — a selective upper-tail opportunity gate, not 0.70. |
| `energy_floor` | {cfg.energy_floor} | Low-tail of development energy (~5.8% of events ≤ 0.10). 0.05 is rarer; 0.20 starts cutting into the body. Energy is an affordability veto, not a second model. |
| `energy_adequate_if_lookahead_unusable` | {cfg.energy_adequate_if_lookahead_unusable} | Truncation is rare (13 development events). Require clearly adequate energy when the 400 m window is incomplete. |
| `deploy_share_min` | {cfg.deploy_share_min} | Row-local mix, not a raw cross-race LookaheadDeployProxy cut. Harvest proxies are systematically larger, so 0.50 would be biased; 0.40 means deploy is a substantial fraction of the 400 m mix. Straights remain spend-compatible even with a lower mix. |

**Not selected by overtake_success, F1, or pass count.**

---

### FILES CHANGED

- `src/trackshift/strategy.py` (new)
- `src/trackshift/config.py` (output paths only)
- `scripts/run_strategy_engine.py` (new)
- `tests/test_strategy.py` (new)
- `outputs/strategy_v1/` (new; does not overwrite event/energy/ML files)

---

### UNIT TESTS

{"Passed." if tests_ok else "FAILED."}

---

### DEVELOPMENT REPLAY

n = {dev_summary['n']}  
ATTACK = {dev_summary['n_attack']} ({100*dev_summary['attack_rate']:.1f}%)  
SAVE = {dev_summary['n_save']} ({100*dev_summary['save_rate']:.1f}%)

Median p_hat: ATTACK {dev_summary['median_p_hat_by_action'][ACTION_ATTACK]:.3f} vs SAVE {dev_summary['median_p_hat_by_action'][ACTION_SAVE]:.3f}  
Median energy: ATTACK {dev_summary['median_energy_by_action'][ACTION_ATTACK]:.3f} vs SAVE {dev_summary['median_energy_by_action'][ACTION_SAVE]:.3f}  
Energy-floor vetoes (all rows, hierarchy-first): {dev_summary['energy_floor_vetoes']}  
Energy-floor vetoes among p-gate passers: {dev_summary['energy_floor_vetoes_among_p_passers']}  
Spend-context vetoes among p-gate passers: {dev_summary['spend_context_vetoes_among_p_passers']}  
Lookahead-unusable vetoes: {dev_summary['lookahead_unusable_vetoes']}

ATTACK median p_hat is higher than SAVE: **{dev_summary['attack_p_hat_gt_save']}**.

{chr(10).join(per_race_lines)}

Zone share among ATTACK: {dev_summary['zone_share_by_action'].get(ACTION_ATTACK, {})}

`overtake_success` was not used to decide. Analysis-only pass rate: ATTACK {dev_summary['analysis_only_pass_rate_by_action'][ACTION_ATTACK]:.3f}, SAVE {dev_summary['analysis_only_pass_rate_by_action'][ACTION_SAVE]:.3f}.

---

### LOCKED TEST CONFIRMATION

One shot on frozen thresholds. Not used to change the engine.

n = {test_summary['n']}  
ATTACK = {test_summary['n_attack']} ({100*test_summary['attack_rate']:.1f}%)  
SAVE = {test_summary['n_save']} ({100*test_summary['save_rate']:.1f}%)

Median p_hat: ATTACK {test_summary['median_p_hat_by_action'][ACTION_ATTACK]:.3f} vs SAVE {test_summary['median_p_hat_by_action'][ACTION_SAVE]:.3f}

{chr(10).join(test_lines)}

Two races are confirmation only, not universal generalisation.

---

### ABLATION

{chr(10).join(ab_lines)}

Energy changes few ATTACK decisions because the simulated index is usually near the top of [0, 1]: among events that already clear the p-gate, the floor vetoes only a handful. Immediate 400 m context (straight or deploy-share) removes a larger slice of high-p fights that sit in harvest-heavy brake/corner windows. That is reported, not dressed up.

DEFEND ablation was not run.

---

### LIMITATIONS

- No causal behind-car signal; DEFEND is not implemented.
- `EstimatedEnergyIndex` is a simulated 0–1 resource index, not F1 SOC / ERS.
- Lookahead is the next ~400 m (~6–14 s), not later-lap or race demand.
- Energy is often saturated; affordability vetoes are rare and should stay rare.
- The engine is event-triggered at detector t0, not a full-race optimiser.
- Development replay uses frozen LORO OOF probabilities; locked test uses the saved one-shot GBDT scores. The GBDT was not retrained here.

---

### PPT-READY DESCRIPTION

TrackShift does not attack whenever an overtake looks possible. A frozen model estimates the probability that a pairwise pass completes within 30 seconds. A separate rule layer then asks whether the attacker can afford to spend in the next 400 metres of track — the immediate passing window, not a full-lap energy plan. It attacks only when that probability clears a development-chosen opportunity gate and the simulated energy budget plus immediate track mix support a deploy; otherwise it saves. We do not observe the car behind, so there is no DEFEND mode.

---

### CANDIDATE GRID (development process metrics)

See `threshold_grid.csv`. Selected row: p_hat_min={cfg.p_hat_min}, energy_floor={cfg.energy_floor}, deploy_share_min={cfg.deploy_share_min}.
"""
    return md


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    header("INTEGRITY")
    before = hashes()
    for name, digest in before.items():
        print(f"  {name:16s} {digest[:16]}...")

    header("LOAD FROZEN ML ARTIFACTS")
    features = load_features()
    oof = load_gbdt_oof()
    locked = load_locked_test()
    print(f"  OOF GBDT rows     : {len(oof)}")
    print(f"  locked-test rows  : {len(locked)}")
    print(f"  feature rows      : {len(features)}")

    dev = attach(oof, features)
    test = attach(locked, features)
    unknown_dev = set(dev.race_id.unique()) - set(DEV_RACES)
    unknown_test = set(test.race_id.unique()) - set(TEST_RACES)
    if unknown_dev or unknown_test:
        raise RuntimeError(f"Split leak: extra_dev={unknown_dev} extra_test={unknown_test}")
    if set(dev.race_id.unique()) != set(DEV_RACES):
        raise RuntimeError(f"Development races incomplete: {sorted(dev.race_id.unique())}")
    if set(test.race_id.unique()) != set(TEST_RACES):
        raise RuntimeError(f"Locked races incomplete: {sorted(test.race_id.unique())}")
    print(f"  development events: {len(dev)}")
    print(f"  locked-test events: {len(test)}")

    header("DEVELOPMENT DISTRIBUTIONS")
    dist = describe_distributions(dev)
    print(json.dumps({k: dist[k] for k in dist if k != "energy_quantiles"}, indent=2, default=str)[:2000])
    print("  energy_quantiles:", dist["energy_quantiles"])
    print("  energy <=0.10 frac:", dist["energy_frac_le_0_10"])

    header("THRESHOLD GRID (process metrics, not labels)")
    grid_rows = []
    frozen = FROZEN_STRATEGY_CONFIG
    for p in P_CANDIDATES:
        grid_rows.append(grid_row(dev, StrategyConfig(p_hat_min=p, energy_floor=frozen.energy_floor, deploy_share_min=frozen.deploy_share_min)))
    for e in ENERGY_CANDIDATES:
        grid_rows.append(grid_row(dev, StrategyConfig(p_hat_min=frozen.p_hat_min, energy_floor=e, deploy_share_min=frozen.deploy_share_min)))
    for s in SHARE_CANDIDATES:
        grid_rows.append(grid_row(dev, StrategyConfig(p_hat_min=frozen.p_hat_min, energy_floor=frozen.energy_floor, deploy_share_min=s)))
    grid = pd.DataFrame(grid_rows).drop_duplicates(
        subset=["p_hat_min", "energy_floor", "deploy_share_min"]
    )
    print(grid.to_string(index=False))
    grid.to_csv(OUT / "threshold_grid.csv", index=False)

    selected = frozen.as_dict()
    selected["selection_rule"] = (
        "Precommitted middle opportunity gate (0.45), low-tail energy floor (0.10), "
        "and row-local deploy share 0.40. Not maximised against overtake_success."
    )
    selected["belgium_miami_used_for_thresholds"] = False
    (OUT / "strategy_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("\n  FROZEN:", selected)

    header("DEVELOPMENT REPLAY")
    dev_replay = replay_table(dev, "development_oof", ABLATION_FULL, frozen)
    dev_applied = apply_strategy(dev, frozen, ABLATION_FULL)
    dev_summary = process_summary(dev_applied, "development_oof")
    print(json.dumps({k: v for k, v in dev_summary.items() if k != "per_race"}, indent=2, default=str))
    for row in dev_summary["per_race"]:
        print(
            f"  {row['race_id']:22s} n={row['n']:4d}  ATTACK={row['n_attack']:4d} "
            f"({100*row['attack_rate']:5.1f}%)  SAVE={row['n_save']:4d}"
        )
    dev_replay.to_csv(OUT / "development_replay.csv", index=False)

    header("LOCKED TEST CONFIRMATION (no retune)")
    test_replay = replay_table(test, "locked_test", ABLATION_FULL, frozen)
    test_applied = apply_strategy(test, frozen, ABLATION_FULL)
    test_summary = process_summary(test_applied, "locked_test")
    print(json.dumps({k: v for k, v in test_summary.items() if k != "per_race"}, indent=2, default=str))
    for row in test_summary["per_race"]:
        print(
            f"  {row['race_id']:22s} n={row['n']:4d}  ATTACK={row['n_attack']:4d} "
            f"({100*row['attack_rate']:5.1f}%)  SAVE={row['n_save']:4d}"
        )
    test_replay.to_csv(OUT / "locked_test_replay.csv", index=False)

    header("ABLATION (frozen thresholds)")
    ablation = {}
    for mode, name in [
        (ABLATION_P_ONLY, "A"),
        (ABLATION_P_ENERGY, "B"),
        (ABLATION_FULL, "C"),
    ]:
        applied = apply_strategy(dev, frozen, ablation=mode)
        ablation[name] = process_summary(applied, f"dev_ablation_{name}")
        replay_table(dev, f"development_ablation_{name}", mode, frozen).to_csv(
            OUT / f"development_ablation_{name}.csv", index=False
        )
        print(
            f"  {name}: attack_rate={ablation[name]['attack_rate']:.3f}  "
            f"energy_veto_among_p={ablation[name]['energy_floor_vetoes_among_p_passers']}  "
            f"spend_veto_among_p={ablation[name]['spend_context_vetoes_among_p_passers']}"
        )

    header("UNIT TESTS")
    import unittest

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_strategy.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    tests_ok = result.wasSuccessful()
    print(f"  tests_ok={tests_ok}  ran={result.testsRun}")

    header("WRITE REPORT")
    report = write_report(
        dist=dist,
        grid=grid,
        selected=selected,
        dev_summary=dev_summary,
        test_summary=test_summary,
        ablation=ablation,
        hashes_before=before,
        hashes_after=hashes(),
        tests_ok=tests_ok,
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        "version": VERSION,
        "frozen_ml": "outputs/ml_selection_v1 GBDT OOF + locked_test_predictions",
        "defend_implemented": False,
        "config": selected,
        "development_distributions": dist,
        "development": dev_summary,
        "locked_test": test_summary,
        "ablation": ablation,
        "integrity": {"before": before, "after": hashes(), "unchanged": before == hashes()},
        "unit_tests_passed": tests_ok,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "strategy_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    after = hashes()
    if before != after:
        print("STOP: protected files changed")
        return 2
    if not tests_ok:
        print("STOP: unit tests failed")
        return 1
    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
