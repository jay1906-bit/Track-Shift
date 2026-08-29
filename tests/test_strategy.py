"""Unit tests for the ATTACK/SAVE strategy engine.

The engine must not require outcome or future columns.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift.strategy import (  # noqa: E402
    ACTION_ATTACK,
    ACTION_INVALID,
    ACTION_SAVE,
    FROZEN_STRATEGY_CONFIG,
    REASON_ATTACK,
    REASON_ENERGY_FLOOR,
    REASON_ENERGY_MISSING,
    REASON_INVALID_P,
    REASON_LOOKAHEAD_UNUSABLE,
    REASON_P_BELOW,
    REASON_SPEND_CONTEXT,
    decide_strategy,
)


CFG = FROZEN_STRATEGY_CONFIG


def decide(**kwargs):
    defaults = dict(
        p_hat=0.80,
        energy=0.90,
        lookahead_deploy=20000.0,
        lookahead_harvest=10000.0,
        lookahead_covered_m=400.0,
        lookahead_truncated=False,
        zone_type="straight",
        strategy_config=CFG,
    )
    defaults.update(kwargs)
    return decide_strategy(**defaults)


class StrategyEngineTests(unittest.TestCase):
    def test_high_p_healthy_energy_straight_attack(self):
        d = decide(p_hat=0.80, energy=0.90, zone_type="straight")
        self.assertEqual(d.action, ACTION_ATTACK)
        self.assertEqual(d.reason_code, REASON_ATTACK)

    def test_low_p_healthy_energy_save(self):
        d = decide(p_hat=0.10, energy=0.95)
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_P_BELOW)

    def test_high_p_energy_near_floor_save(self):
        d = decide(p_hat=0.90, energy=0.02)
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_ENERGY_FLOOR)

    def test_high_p_truncated_low_energy_save(self):
        d = decide(
            p_hat=0.90,
            energy=0.20,
            lookahead_truncated=True,
            lookahead_covered_m=80.0,
        )
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_LOOKAHEAD_UNUSABLE)

    def test_high_p_healthy_deploy_friendly_attack(self):
        d = decide(
            p_hat=0.70,
            energy=0.85,
            zone_type="brake",
            lookahead_deploy=0.60,
            lookahead_harvest=0.30,
        )
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_medium_p_just_below_threshold_save(self):
        d = decide(p_hat=CFG.p_hat_min - 1e-9, energy=0.90)
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_P_BELOW)

    def test_medium_p_just_above_threshold_attack(self):
        d = decide(p_hat=CFG.p_hat_min, energy=0.90, zone_type="straight")
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_probability_zero_save(self):
        d = decide(p_hat=0.0, energy=0.90)
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_P_BELOW)

    def test_probability_one_healthy_attack(self):
        d = decide(p_hat=1.0, energy=0.90, zone_type="straight")
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_invalid_probability_rejected(self):
        for bad in (-0.01, 1.01, float("nan"), None, "not-a-prob"):
            d = decide(p_hat=bad, energy=0.90)
            self.assertEqual(d.action, ACTION_INVALID, msg=str(bad))
            self.assertEqual(d.reason_code, REASON_INVALID_P)

    def test_missing_energy_save(self):
        d = decide(p_hat=0.90, energy=None)
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_ENERGY_MISSING)

    def test_missing_lookahead_conservative(self):
        d = decide(
            p_hat=0.90,
            energy=0.20,
            lookahead_deploy=None,
            lookahead_harvest=None,
            lookahead_covered_m=None,
            zone_type="brake",
        )
        self.assertEqual(d.action, ACTION_SAVE)
        self.assertEqual(d.reason_code, REASON_LOOKAHEAD_UNUSABLE)

    def test_truncated_true_with_adequate_energy_can_attack(self):
        d = decide(
            p_hat=0.80,
            energy=0.90,
            lookahead_truncated=True,
            lookahead_covered_m=90.0,
            zone_type="straight",
        )
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_energy_floor_boundary(self):
        below = decide(p_hat=0.80, energy=CFG.energy_floor - 1e-12)
        at = decide(p_hat=0.80, energy=CFG.energy_floor, zone_type="straight")
        self.assertEqual(below.action, ACTION_SAVE)
        self.assertEqual(below.reason_code, REASON_ENERGY_FLOOR)
        self.assertEqual(at.action, ACTION_ATTACK)

    def test_deploy_share_boundary_non_straight(self):
        # share = 0.40 exactly at the gate, brake zone.
        at = decide(
            p_hat=0.80,
            energy=0.90,
            zone_type="brake",
            lookahead_deploy=0.40,
            lookahead_harvest=0.60,
        )
        below = decide(
            p_hat=0.80,
            energy=0.90,
            zone_type="brake",
            lookahead_deploy=0.399,
            lookahead_harvest=0.601,
        )
        self.assertEqual(at.action, ACTION_ATTACK)
        self.assertEqual(below.action, ACTION_SAVE)
        self.assertEqual(below.reason_code, REASON_SPEND_CONTEXT)

    def test_high_deploy_does_not_force_save(self):
        d = decide(
            p_hat=0.80,
            energy=0.90,
            zone_type="straight",
            lookahead_deploy=1e6,
            lookahead_harvest=1.0,
        )
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_interface_rejects_future_columns(self):
        params = inspect.signature(decide_strategy).parameters
        forbidden = {
            "overtake_success",
            "actual_swap_delay_s",
            "pass_after_horizon",
            "Position",
            "PositionChange",
            "reciprocal_event_id",
            "event_duration_s",
            "gap_at_start_m",
            "closing_speed_at_start_kmh",
            "event_end_time",
        }
        self.assertTrue(forbidden.isdisjoint(params))

    def test_p_only_ablation_ignores_empty_energy(self):
        d = decide_strategy(
            p_hat=0.80,
            energy=0.01,
            lookahead_deploy=1.0,
            lookahead_harvest=100.0,
            lookahead_covered_m=400.0,
            lookahead_truncated=False,
            zone_type="corner",
            ablation="A",
        )
        self.assertEqual(d.action, ACTION_ATTACK)

    def test_energy_ablation_ignores_poor_spend_window(self):
        d = decide_strategy(
            p_hat=0.80,
            energy=0.90,
            lookahead_deploy=1.0,
            lookahead_harvest=100.0,
            lookahead_covered_m=400.0,
            lookahead_truncated=False,
            zone_type="corner",
            ablation="B",
        )
        self.assertEqual(d.action, ACTION_ATTACK)


if __name__ == "__main__":
    unittest.main()
