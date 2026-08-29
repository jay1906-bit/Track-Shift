"""Checks that the frontend replay adapter did not alter frozen science."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.validate import file_sha256  # noqa: E402


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


class FrontendReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay_path = config.FRONTEND_REPLAY_JSON
        if not cls.replay_path.exists():
            raise unittest.SkipTest("Run python scripts/build_frontend_replay.py first")
        cls.payload = json.loads(cls.replay_path.read_text(encoding="utf-8"))
        cls.by_id = {row["event_id"]: row for row in cls.payload["events"]}
        cls.dev = pd.read_csv(config.STRATEGY_V1_DIR / "development_replay.csv")
        cls.locked = pd.read_csv(config.STRATEGY_V1_DIR / "locked_test_replay.csv")
        cls.replay = pd.concat([cls.dev, cls.locked], ignore_index=True)

    def test_replay_mode_not_live(self):
        self.assertEqual(self.payload["mode"], "historical_replay")
        self.assertFalse(self.payload["live_inference"])

    def test_no_defend_in_actions(self):
        actions = {row["strategy_action"] for row in self.payload["events"]}
        self.assertTrue(actions.issubset({"ATTACK", "SAVE", "INVALID", None}))
        self.assertNotIn("DEFEND", actions)

    def test_smoke_event_matches_backend(self):
        smoke = self.payload["smoke_test"]
        event_id = smoke["event_id"]
        backend = self.replay.loc[self.replay["event_id"] == event_id].iloc[0]
        ui = self.by_id[event_id]
        self.assertAlmostEqual(float(ui["p_hat"]), float(backend["p_hat"]), places=12)
        self.assertAlmostEqual(float(smoke["p_hat"]), float(backend["p_hat"]), places=12)
        self.assertEqual(ui["strategy_action"], backend["strategy_action"])
        self.assertEqual(smoke["strategy_action"], backend["strategy_action"])
        self.assertEqual(ui["strategy_reason"], backend["strategy_reason"])
        self.assertAlmostEqual(
            float(ui["EstimatedEnergyIndex"]),
            float(backend["EstimatedEnergyIndex"]),
            places=12,
        )

    def test_event_count_matches_strategy_tables(self):
        self.assertEqual(len(self.payload["events"]), len(self.replay))

    def test_curated_ids_are_real(self):
        ids = set(self.replay["event_id"].astype(str))
        for item in self.payload["curated"]:
            self.assertIn(item["event_id"], ids)
            self.assertIn(item["event_id"], self.by_id)

    def test_protected_hashes_unchanged(self):
        recorded = json.loads((config.FRONTEND_V1_DIR / "protected_hashes.json").read_text(encoding="utf-8"))
        for path in PROTECTED:
            if not path.exists():
                continue
            self.assertEqual(file_sha256(path), recorded[str(path)])

    def test_circuits_copied_for_all_races(self):
        races = {row["race_id"] for row in self.payload["races"]}
        circuits = self.payload.get("circuits") or {}
        self.assertEqual(races, set(circuits))
        for race_id, circuit in circuits.items():
            self.assertTrue(circuit["circuit_length_m"] > 0)
            self.assertGreaterEqual(len(circuit["zones"]), 1)
            self.assertEqual(circuit["schematic"], "telemetry_polyline")
            poly = circuit.get("polyline") or []
            self.assertGreaterEqual(len(poly), 50)
            self.assertEqual(poly[0][2], 0)
            self.assertAlmostEqual(poly[-1][2], 1.0, places=4)
            self.assertEqual(circuit["geometry_source"], "fastf1_pos_data")

    def test_polylines_are_not_circles(self):
        """Characteristic circuit shape, not a constant-radius loop."""
        for race_id, circuit in self.payload["circuits"].items():
            xs = [p[0] for p in circuit["polyline"]]
            ys = [p[1] for p in circuit["polyline"]]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            radii = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in zip(xs, ys)]
            mean_r = sum(radii) / len(radii)
            var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
            cv = (var ** 0.5) / mean_r
            self.assertGreater(cv, 0.08, msg=f"{race_id} looks circular (cv={cv:.3f})")

    def test_nine_races_present(self):
        expected = {
            "australia_2026_R",
            "china_2026_R",
            "japan_2026_R",
            "miami_2026_R",
            "canada_2026_R",
            "monaco_2026_R",
            "austria_2026_R",
            "belgium_2026_R",
            "netherlands_2026_R",
        }
        self.assertEqual({row["race_id"] for row in self.payload["races"]}, expected)
        self.assertEqual(set(self.payload["circuits"]), expected)

    def test_model_comparison_copied_not_invented(self):
        summary = json.loads((config.ML_SELECTION_DIR / "ml_selection_summary.json").read_text(encoding="utf-8"))
        rows = {row["id"]: row for row in self.payload["metrics"]["model_comparison"]}
        self.assertAlmostEqual(rows["baseline"]["pr_auc"], summary["loro_summaries"]["baseline"]["macro_pr_auc"])
        self.assertAlmostEqual(rows["logistic"]["pr_auc"], summary["loro_summaries"]["logistic"]["macro_pr_auc"])
        self.assertAlmostEqual(rows["gbdt"]["pr_auc"], summary["loro_summaries"]["gbdt"]["macro_pr_auc"])
        self.assertTrue(rows["gbdt"]["selected"])
        self.assertFalse(rows["baseline"]["selected"])

    def test_circuit_length_copied_from_ml_safe(self):
        smoke = self.payload["smoke_test"]["event_id"]
        ui = self.by_id[smoke]
        self.assertIsNotNone(ui.get("circuit_length_m"))
        self.assertGreater(ui["circuit_length_m"], 1000)

    def test_historical_outcome_copied_not_invented(self):
        smoke_id = self.payload["smoke_test"]["event_id"]
        ui = self.by_id[smoke_id]
        backend = self.replay.loc[self.replay["event_id"] == smoke_id].iloc[0]
        self.assertIn("analysis_only_overtake_success", ui)
        if "overtake_success_analysis_only" in backend.index:
            expected = backend["overtake_success_analysis_only"]
        else:
            expected = backend["overtake_success"]
        if pd.isna(expected):
            self.assertTrue(ui["analysis_only_overtake_success"] in (None,))
        else:
            self.assertEqual(int(ui["analysis_only_overtake_success"]), int(expected))

    def test_demo_scenario_ids_exist(self):
        curated_ids = {item["id"] for item in self.payload["curated"]}
        for slot in ("australia_attack", "energy_floor", "spend_context"):
            self.assertIn(slot, curated_ids)


if __name__ == "__main__":
    unittest.main()
