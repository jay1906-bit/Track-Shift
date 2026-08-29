"""Deterministic ATTACK / SAVE engine on frozen GBDT probabilities.

This module does not train models, modify event labels, or recompute energy
or lookahead. DEFEND is not implemented: the dataset has no causal
behind-car signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


ACTION_ATTACK = "ATTACK"
ACTION_SAVE = "SAVE"
ACTION_INVALID = "INVALID"

ABLATION_P_ONLY = "A"
ABLATION_P_ENERGY = "B"
ABLATION_FULL = "C"

REASON_INVALID_P = "invalid_p_hat"
REASON_ENERGY_MISSING = "energy_missing"
REASON_ENERGY_FLOOR = "energy_floor"
REASON_LOOKAHEAD_UNUSABLE = "lookahead_unusable"
REASON_P_BELOW = "p_hat_below_gate"
REASON_SPEND_CONTEXT = "spend_context"
REASON_ATTACK = "attack"

REASON_TEXT = {
    REASON_INVALID_P: "Invalid overtake probability; strategy refused to decide.",
    REASON_ENERGY_MISSING: "Energy missing; affordability cannot be established.",
    REASON_ENERGY_FLOOR: "Energy too close to the resource floor.",
    REASON_LOOKAHEAD_UNUSABLE: (
        "Lookahead is truncated or unusable and affordability cannot be established."
    ),
    REASON_P_BELOW: "Overtake probability below the strategic opportunity threshold.",
    REASON_SPEND_CONTEXT: (
        "Immediate 400 m track context is not a sensible spending/deployment window."
    ),
    REASON_ATTACK: (
        "High overtake probability with sufficient energy for the immediate deployment window."
    ),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen operating point. Selected on development OOF only."""

    p_hat_min: float = 0.45
    energy_floor: float = 0.10
    energy_adequate_if_lookahead_unusable: float = 0.50
    deploy_share_min: float = 0.40
    straight_is_spend_window: bool = True
    covered_min_m: float = 50.0
    version: str = "strategy_v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_STRATEGY_CONFIG = StrategyConfig()


@dataclass
class StrategyDecision:
    action: str
    p_hat: float | None
    energy: float | None
    deploy_share: float | None
    zone_type: str | None
    lookahead_truncated: bool
    lookahead_usable: bool
    spend_compatible: bool
    reason_code: str
    reason: str
    config: dict[str, Any]
    ablation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _parse_truncated(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "t"}


def _parse_zone(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "unassigned", "missing"}:
        return None
    return text


def deploy_share(deploy: Any, harvest: Any) -> float | None:
    """Row-local mix: deploy / (deploy + harvest). Not a cross-race raw threshold."""
    d = _finite(deploy)
    h = _finite(harvest)
    if d is None and h is None:
        return None
    d = 0.0 if d is None else max(d, 0.0)
    h = 0.0 if h is None else max(h, 0.0)
    total = d + h
    if total <= 0.0:
        return None
    return d / total


def lookahead_is_usable(
    *,
    truncated: bool,
    covered_m: Any,
    deploy: Any,
    harvest: Any,
    covered_min_m: float,
) -> bool:
    if truncated:
        return False
    covered = _finite(covered_m)
    if covered is None or covered < covered_min_m:
        return False
    if _finite(deploy) is None and _finite(harvest) is None:
        return False
    return True


def spend_window_ok(
    *,
    zone_type: str | None,
    share: float | None,
    cfg: StrategyConfig,
) -> bool:
    if cfg.straight_is_spend_window and zone_type == "straight":
        return True
    if share is not None and share >= cfg.deploy_share_min:
        return True
    return False


def decide_strategy(
    p_hat: Any,
    energy: Any,
    lookahead_deploy: Any = None,
    lookahead_harvest: Any = None,
    lookahead_covered_m: Any = None,
    lookahead_truncated: Any = False,
    zone_type: Any = None,
    strategy_config: StrategyConfig | None = None,
    ablation: str = ABLATION_FULL,
) -> StrategyDecision:
    """Return ATTACK or SAVE from causal t0 inputs only.

    Does not accept outcome columns. gap / closing speed are not inputs.
    """
    cfg = strategy_config or FROZEN_STRATEGY_CONFIG
    if ablation not in {ABLATION_P_ONLY, ABLATION_P_ENERGY, ABLATION_FULL}:
        raise ValueError(f"Unknown ablation mode: {ablation}")

    p = _finite(p_hat)
    e = _finite(energy)
    truncated = _parse_truncated(lookahead_truncated)
    zone = _parse_zone(zone_type)
    share = deploy_share(lookahead_deploy, lookahead_harvest)
    usable = lookahead_is_usable(
        truncated=truncated,
        covered_m=lookahead_covered_m,
        deploy=lookahead_deploy,
        harvest=lookahead_harvest,
        covered_min_m=cfg.covered_min_m,
    )
    spend = spend_window_ok(zone_type=zone, share=share, cfg=cfg)
    use_energy = ablation in {ABLATION_P_ENERGY, ABLATION_FULL}
    use_demand = ablation == ABLATION_FULL

    def result(action: str, code: str) -> StrategyDecision:
        return StrategyDecision(
            action=action,
            p_hat=p,
            energy=e,
            deploy_share=share,
            zone_type=zone,
            lookahead_truncated=truncated,
            lookahead_usable=usable,
            spend_compatible=spend,
            reason_code=code,
            reason=REASON_TEXT[code],
            config=cfg.as_dict(),
            ablation=ablation,
        )

    if p is None or p < 0.0 or p > 1.0:
        return result(ACTION_INVALID, REASON_INVALID_P)

    if use_energy:
        if e is None:
            return result(ACTION_SAVE, REASON_ENERGY_MISSING)
        if e < cfg.energy_floor:
            return result(ACTION_SAVE, REASON_ENERGY_FLOOR)
        if (not usable) and e < cfg.energy_adequate_if_lookahead_unusable:
            return result(ACTION_SAVE, REASON_LOOKAHEAD_UNUSABLE)

    if p < cfg.p_hat_min:
        return result(ACTION_SAVE, REASON_P_BELOW)

    if use_demand and not spend:
        return result(ACTION_SAVE, REASON_SPEND_CONTEXT)

    return result(ACTION_ATTACK, REASON_ATTACK)


def apply_strategy(
    frame: pd.DataFrame,
    strategy_config: StrategyConfig | None = None,
    ablation: str = ABLATION_FULL,
    p_col: str = "p_hat",
) -> pd.DataFrame:
    """Vectorized apply. Requires p_hat and energy/lookahead columns; not labels."""
    cfg = strategy_config or FROZEN_STRATEGY_CONFIG
    rows = []
    for data in frame.to_dict(orient="records"):
        decision = decide_strategy(
            p_hat=data.get(p_col),
            energy=data.get("EstimatedEnergyIndex"),
            lookahead_deploy=data.get("LookaheadDeployProxy"),
            lookahead_harvest=data.get("LookaheadHarvestProxy"),
            lookahead_covered_m=data.get("LookaheadCoveredM"),
            lookahead_truncated=data.get("lookahead_truncated"),
            zone_type=data.get("zone_type"),
            strategy_config=cfg,
            ablation=ablation,
        )
        rows.append(decision.as_dict())
    out = pd.DataFrame(rows)
    keep = frame.reset_index(drop=True)
    return pd.concat([keep, out.add_prefix("strategy_")], axis=1)
