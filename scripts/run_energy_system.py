"""Run Checkpoints 1–12 from the TrackShift Energy System Design Report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.calibrate import apply_scale, calibrate_alpha_beta, calibration_mask, save_config  # noqa: E402
from trackshift.energy import clip_lap_fraction, end_of_lap_energy, walk_all_drivers  # noqa: E402
from trackshift.pipeline import integrate_lap, prepare_lap_telemetry, process_all_laps  # noqa: E402
from trackshift.plots import (  # noqa: E402
    plot_energy_lookahead,
    plot_freeze_energy,
    plot_reference_proxies,
    plot_throttle_vs_dlap,
    plot_track_map,
    plot_two_driver_energy,
)
from trackshift.proxies import winsorize_session_proxies  # noqa: E402
from trackshift.race_state import is_box_lap, is_freeze_status  # noqa: E402
from trackshift.session_io import load_clean_laps, load_session, pick_session_lap, reference_lap_row  # noqa: E402
from trackshift.validate import file_sha256, run_validation  # noqa: E402
from trackshift.zones import (  # noqa: E402
    add_sample_lookahead,
    assign_zone,
    build_zone_table,
    lap_start_lookahead,
    zone_baselines,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def recompute_lap_integrals(lap_table: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    out = lap_table.copy()
    updated = []
    for (drv, ln), grp in samples.groupby(["Driver", "LapNumber"]):
        thin = bool(out.loc[(out["Driver"] == drv) & (out["LapNumber"] == ln), "thin_lap"].iloc[0])
        integ = integrate_lap(grp, {"thin_lap": thin})
        integ["Driver"] = drv
        integ["LapNumber"] = ln
        updated.append(integ)
    upd = pd.DataFrame(updated)
    drop_cols = [c for c in ["D_lap", "H_lap", "max_dt_used", "n_dt_gaps", "mean_throttle"] if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out.merge(upd, on=["Driver", "LapNumber"], how="left")


def main() -> dict:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_hash = file_sha256(config.RAW_LAPS_CSV)
    clean_hash = file_sha256(config.CLEAN_LAPS_CSV)
    report: dict = {"raw_sha256_before": raw_hash, "clean_sha256_before": clean_hash}

    print("Loading FastF1 session from cache and cleaned lap spine...")
    session = load_session()
    laps_clean = load_clean_laps()
    ref_row = reference_lap_row(laps_clean)
    print("Clean laps:", len(laps_clean), "drivers:", laps_clean["Driver"].nunique())

    # ------------------------------------------------------------------
    # CP1 — Telemetry extract
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 1 — TELEMETRY EXTRACT")
    ref_samples, ref_meta = prepare_lap_telemetry(session, ref_row)
    print("Driver:", ref_meta["Driver"], "LapNumber:", ref_meta["LapNumber"])
    print("columns:", list(ref_samples.columns))
    print("n padded raw samples:", ref_meta["n_raw_samples"])
    print("n in-lap window samples:", ref_meta["n_in_window"])
    print("n valid samples:", ref_meta["n_valid_samples"])
    dt = ref_samples["DeltaTimeSeconds"]
    print("DeltaTimeSeconds describe:\n", dt.describe().to_string())
    print("median dt:", float(dt.median()))
    thr = pd.to_numeric(ref_samples["Throttle"], errors="coerce")
    print("Throttle 104 count:", int((thr == 104).sum()))
    print("Throttle out of [0,100]:", int((~thr.between(0, 100)).sum()))
    print("Brake dtype:", ref_samples["Brake"].dtype)
    print("Source unique:" , ref_samples["Source"].unique().tolist() if "Source" in ref_samples.columns else "n/a")
    print("Used get_car_data(pad=1, pad_side='both') only. Did not use get_telemetry() for integrals.")
    report["cp1"] = {
        "driver": ref_meta["Driver"],
        "lap": ref_meta["LapNumber"],
        "n_in_window": ref_meta["n_in_window"],
        "median_dt": float(dt.median()),
        "throttle_104": int((thr == 104).sum()),
        "brake_dtype": str(ref_samples["Brake"].dtype),
    }

    # ------------------------------------------------------------------
    # CP2 — Distance overlay
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 2 — DISTANCE OVERLAY")
    print("Distance min/max:", float(ref_samples["Distance"].min()), float(ref_samples["Distance"].max()))
    print("Distance is_monotonic:", bool(ref_samples["Distance"].is_monotonic_increasing))
    print("Brake=True count:", int(ref_samples["Brake"].astype(bool).sum()))
    report["cp2"] = {
        "distance_min": float(ref_samples["Distance"].min()),
        "distance_max": float(ref_samples["Distance"].max()),
        "monotonic": bool(ref_samples["Distance"].is_monotonic_increasing),
    }

    # ------------------------------------------------------------------
    # CP3 — Sample-level proxies, one lap
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 3 — SAMPLE-LEVEL PROXIES (ONE LAP)")
    print("DeployProxy mean/max:", float(ref_samples["DeployProxy"].mean()), float(ref_samples["DeployProxy"].max()))
    print("HarvestProxy mean/max:", float(ref_samples["HarvestProxy"].mean()), float(ref_samples["HarvestProxy"].max()))
    print("HarvestProxy when Brake=False max:", float(ref_samples.loc[~ref_samples["Brake"].astype(bool), "HarvestProxy"].max()))
    print("Formulas: DeployProxy = v * tau * race_state; HarvestProxy = v * d * BrakeGate(d>=d_min).")
    print("d_min =", config.D_MIN_MS2, "m/s^2 (modeling choice, not F1).")
    report["cp3"] = {
        "deploy_mean": float(ref_samples["DeployProxy"].mean()),
        "harvest_mean": float(ref_samples["HarvestProxy"].mean()),
        "harvest_if_no_brake_max": float(
            ref_samples.loc[~ref_samples["Brake"].astype(bool), "HarvestProxy"].max()
        ),
    }

    # ------------------------------------------------------------------
    # CP4 — Race-state masking
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 4 — RACE-STATE MASKING")
    sc_rows = laps_clean[laps_clean["TrackStatus"].astype(str).map(lambda s: is_freeze_status(s))]
    pit_rows = laps_clean[laps_clean.apply(lambda r: is_box_lap(r["PitInTime"], r["PitOutTime"]), axis=1)]
    print("n freeze-status laps in clean file:", len(sc_rows))
    print("n box laps in clean file:", len(pit_rows))
    sc_demo = None
    pit_demo = None
    if len(sc_rows):
        sc_demo, sc_meta = prepare_lap_telemetry(session, sc_rows.iloc[0])
        print("SC/freeze example:", sc_meta["Driver"], "lap", sc_meta["LapNumber"], sc_meta)
        print("  mean DeployProxy:", float(sc_demo["DeployProxy"].mean()) if len(sc_demo) else None)
        print("  mean HarvestProxy:", float(sc_demo["HarvestProxy"].mean()) if len(sc_demo) else None)
    if len(pit_rows):
        pit_demo, pit_meta = prepare_lap_telemetry(session, pit_rows.iloc[0])
        print("Pit example:", pit_meta["Driver"], "lap", pit_meta["LapNumber"], pit_meta)
        print("  mean DeployProxy:", float(pit_demo["DeployProxy"].mean()) if len(pit_demo) else None)
        print("  mean HarvestProxy:", float(pit_demo["HarvestProxy"].mean()) if len(pit_demo) else None)
    report["cp4"] = {
        "n_freeze_status_laps": int(len(sc_rows)),
        "n_box_laps": int(len(pit_rows)),
    }

    # ------------------------------------------------------------------
    # CP5 — All-laps unscaled integrals
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 5 — ALL-LAPS UNSCALED INTEGRALS")
    print("Processing", len(laps_clean), "clean laps...")
    lap_table, samples = process_all_laps(session, laps_clean, progress_every=50)
    print("raw sample rows:", len(samples))
    samples, win_report = winsorize_session_proxies(samples)
    print("winsorize:", win_report)
    lap_table = recompute_lap_integrals(lap_table, samples)
    lap_table["calibration_set"] = calibration_mask(lap_table)
    proxy_cols = [
        "Driver",
        "LapNumber",
        "D_lap",
        "H_lap",
        "n_valid_samples",
        "freeze_flag",
        "yellow_flag",
        "thin_lap",
        "IsAccurate",
        "TrackStatus",
        "max_dt_used",
        "mean_throttle",
        "n_dt_gaps",
        "calibration_set",
        "error",
    ]
    lap_table.loc[:, [c for c in proxy_cols if c in lap_table.columns]].to_csv(
        config.LAP_PROXIES_CSV, index=False
    )
    print("wrote", config.LAP_PROXIES_CSV)
    print("rows:", len(lap_table), "duplicates:", int(lap_table.duplicated(["Driver", "LapNumber"]).sum()))
    print("freeze laps mean D_lap:", float(lap_table.loc[lap_table["freeze_flag"], "D_lap"].fillna(0).mean()))
    print("freeze laps mean H_lap:", float(lap_table.loc[lap_table["freeze_flag"], "H_lap"].fillna(0).mean()))
    print("thin laps:", int(lap_table["thin_lap"].sum()))
    print("errors:", int(lap_table["error"].notna().sum()))
    report["cp5"] = {
        "n_rows": int(len(lap_table)),
        "n_duplicates": int(lap_table.duplicated(["Driver", "LapNumber"]).sum()),
        "n_thin": int(lap_table["thin_lap"].sum()),
        "winsorize": win_report,
        "n_samples": int(len(samples)),
    }

    # ------------------------------------------------------------------
    # CP6 — alpha / beta
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 6 — ALPHA/BETA CALIBRATION")
    cfg = calibrate_alpha_beta(lap_table)
    print(json.dumps({k: cfg[k] for k in ["alpha", "beta", "target_median_abs_net", "calibration_set", "comment"]}, indent=2))
    report["cp6_initial"] = {"alpha": cfg["alpha"], "beta": cfg["beta"], "calibration_set": cfg["calibration_set"]}

    # ------------------------------------------------------------------
    # CP7 — EstimatedEnergyIndex walk
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 7 — ESTIMATED ENERGY INDEX WALK")
    samples_e = walk_all_drivers(samples, cfg["alpha"], cfg["beta"])
    end_e = end_of_lap_energy(samples_e)
    frac_clip = clip_lap_fraction(end_e)
    print("clip-lap fraction (floor or recover-to-ceiling):", frac_clip)
    scale_iters = 0
    # Reduce scale only if many laps hit the FLOOR (true over-spend). Ceiling clips
    # at E0=1.0 from surplus harvest are expected and must not zero the budget.
    floor_frac = float((end_e["n_clip_low"] > 0).mean())
    print("floor-clip lap fraction:", floor_frac)
    while floor_frac > config.CLIP_LAP_FRACTION_LIMIT and scale_iters < config.CLIP_RATE_MAX_ITERS:
        cfg = apply_scale(cfg, config.CLIP_RATE_REDUCE_FACTOR)
        scale_iters += 1
        samples_e = walk_all_drivers(samples, cfg["alpha"], cfg["beta"])
        end_e = end_of_lap_energy(samples_e)
        floor_frac = float((end_e["n_clip_low"] > 0).mean())
        frac_clip = clip_lap_fraction(end_e)
        print(f"  reduced alpha/beta by {config.CLIP_RATE_REDUCE_FACTOR}, floor-clip fraction now {floor_frac:.4f}")
    cfg["clip_lap_fraction"] = frac_clip
    cfg["n_scale_reductions"] = scale_iters
    save_config(cfg)
    print("saved", config.ENERGY_CONFIG_JSON)
    print("final alpha", cfg["alpha"], "beta", cfg["beta"])

    energy_laps = lap_table.merge(end_e, on=["Driver", "LapNumber"], how="left")
    energy_laps["EstimatedEnergyIndex"] = energy_laps["EstimatedEnergyIndex_end"]
    energy_out_cols = [
        "Driver",
        "LapNumber",
        "D_lap",
        "H_lap",
        "n_valid_samples",
        "freeze_flag",
        "yellow_flag",
        "thin_lap",
        "IsAccurate",
        "TrackStatus",
        "mean_throttle",
        "max_dt_used",
        "calibration_set",
        "EstimatedEnergyIndex",
        "EstimatedEnergyIndex_end",
        "EstimatedEnergyIndex_min",
        "EstimatedEnergyIndex_max",
        "n_clip_low",
        "n_clip_high",
    ]
    energy_laps.loc[:, [c for c in energy_out_cols if c in energy_laps.columns]].to_csv(
        config.ENERGY_V1_CSV, index=False
    )
    print("wrote", config.ENERGY_V1_CSV)
    print("E min/max:", float(energy_laps["EstimatedEnergyIndex"].min()), float(energy_laps["EstimatedEnergyIndex"].max()))
    print("fraction of end-of-lap E in (0.05, 0.95):", float(((energy_laps["EstimatedEnergyIndex"] > 0.05) & (energy_laps["EstimatedEnergyIndex"] < 0.95)).mean()))
    report["cp7"] = {
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "clip_lap_fraction": frac_clip,
        "n_scale_reductions": scale_iters,
        "e_min": float(energy_laps["EstimatedEnergyIndex"].min()),
        "e_max": float(energy_laps["EstimatedEnergyIndex"].max()),
    }

    # attach energy onto samples for validation
    samples = samples.merge(
        samples_e.loc[:, ["Driver", "LapNumber", "Date", "EstimatedEnergyIndex"]],
        on=["Driver", "LapNumber", "Date"],
        how="left",
    )
    ref_e = samples.loc[
        (samples["Driver"] == config.REFERENCE_DRIVER) & (samples["LapNumber"] == config.REFERENCE_LAP)
    ].copy()
    ref_e.to_csv(config.REFERENCE_SAMPLES_CSV, index=False)

    # ------------------------------------------------------------------
    # CP8 — Validation pack
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 8 — VALIDATION PACK")
    # merge pit times onto energy_laps if missing
    if "PitInTime" not in energy_laps.columns:
        energy_laps = energy_laps.merge(
            laps_clean.loc[:, ["Driver", "LapNumber", "PitInTime", "PitOutTime"]],
            on=["Driver", "LapNumber"],
            how="left",
        )
    val = run_validation(lap_table, samples, ref_e, energy_laps, raw_hash, clean_hash)
    for chk in val["checks"]:
        flag = "PASS" if chk["passed"] else "FAIL"
        print(f"  [{flag}] {chk['name']}: {chk['detail']}")
    print("validation passed:" , val["passed"], f"{val['n_passed']}/{val['n_checks']}")
    fig1 = plot_reference_proxies(ref_e)
    fig2 = plot_two_driver_energy(energy_laps)
    fig3 = plot_freeze_energy(energy_laps)
    fig4 = plot_throttle_vs_dlap(energy_laps)
    print("figures:", fig1, fig2, fig3, fig4)
    report["cp8"] = val
    report["cp8_figures"] = [str(fig1), str(fig2), str(fig3), str(fig4)]

    # ------------------------------------------------------------------
    # CP9 — CircuitInfo zones
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 9 — CIRCUITINFO ZONES")
    track_length = float(np.nanmax(samples["Distance"].to_numpy(dtype=float)))
    zones = build_zone_table(session, track_length)
    print(zones.to_string(index=False))
    coverage = float(zones["length_m"].sum())
    print("track_length", track_length, "covered", coverage, "gap", track_length - coverage)
    zones.to_csv(config.ZONE_TABLE_CSV, index=False)
    print("wrote", config.ZONE_TABLE_CSV)
    samples["zone_id"] = assign_zone(samples["Distance"].to_numpy(dtype=float), zones)
    ref_e["zone_id"] = assign_zone(ref_e["Distance"].to_numpy(dtype=float), zones)
    print("unassigned samples:", int((samples["zone_id"] == "unassigned").sum()))
    report["cp9"] = {
        "n_zones": int(len(zones)),
        "track_length_m": track_length,
        "covered_m": coverage,
        "zone_types": zones["zone_type"].value_counts().to_dict(),
    }

    # ------------------------------------------------------------------
    # CP10 — Zone baselines + 400 m lookahead
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 10 — ZONE BASELINES + 400m LOOKAHEAD")
    cal_keys = energy_laps.loc[energy_laps["calibration_set"] == True, ["Driver", "LapNumber"]]  # noqa: E712
    zones_stats = zone_baselines(samples, zones, cal_keys)
    print(zones_stats.loc[:, ["zone_id", "zone_type", "length_m", "n_samples", "mean_speed", "mean_throttle", "brake_frac", "mean_DeployProxy", "mean_HarvestProxy"]].to_string(index=False))
    look_laps = lap_start_lookahead(samples, zones_stats)
    energy_c = energy_laps.merge(look_laps, on=["Driver", "LapNumber"], how="left")
    energy_c.to_csv(config.ENERGY_TRACKAWARE_CSV, index=False)
    print("wrote", config.ENERGY_TRACKAWARE_CSV)
    # sanity: lookahead before longest straight
    straights = zones_stats.loc[zones_stats["zone_type"] == "straight"].copy()
    if len(straights):
        longest = straights.sort_values("length_m", ascending=False).iloc[0]
        before = max(0.0, float(longest["start_m"]) - 50.0)
        from trackshift.zones import lookahead_from_distance

        la_before = lookahead_from_distance(before, zones_stats)
        la_in = lookahead_from_distance(float(longest["start_m"]), zones_stats)
        print("longest straight", longest["zone_id"], "length", float(longest["length_m"]))
        print("lookahead 50m before:", la_before)
        print("lookahead at straight start:", la_in)
        report["cp10_lookahead_check"] = {
            "longest_straight": longest["zone_id"],
            "length_m": float(longest["length_m"]),
            "before": la_before,
            "at_start": la_in,
        }
    report["cp10"] = {"n_zones_with_stats": int(len(zones_stats))}

    # ------------------------------------------------------------------
    # CP11 — Visual demo slice
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 11 — VISUAL DEMO SLICE")
    session_lap = pick_session_lap(session, config.REFERENCE_DRIVER, config.REFERENCE_LAP)
    pos = session_lap.get_pos_data().copy()
    pos = pos.sort_values("Date")
    car_for_map = ref_e.sort_values("Date").loc[:, ["Date", "DeployProxy", "HarvestProxy", "zone_id", "Distance"]]
    if "Date" in pos.columns:
        pos_m = pd.merge_asof(
            pos[["Date", "X", "Y"]].sort_values("Date"),
            car_for_map,
            on="Date",
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=300),
        )
    else:
        pos_m = pos
    zone_codes = {z: i for i, z in enumerate(zones["zone_id"].tolist())}
    pos_m["zone_code"] = pos_m["zone_id"].map(zone_codes)
    fig_map = plot_track_map(
        pos_m.dropna(subset=["X", "Y"]),
        "DeployProxy",
        "CP11 — Albert Park X/Y colored by DeployProxy (estimated demand, not F1 ERS/SOC)",
        "cp11_track_map_deploy.png",
    )
    fig_map2 = plot_track_map(
        pos_m.dropna(subset=["X", "Y", "zone_code"]),
        "zone_code",
        "CP11 — Albert Park X/Y colored by CircuitInfo zone (geometry labels, not F1 ERS zones)",
        "cp11_track_map_zones.png",
    )
    fig_el = plot_energy_lookahead(samples, energy_c, driver=config.REFERENCE_DRIVER)
    print("demo figures:", fig_map, fig_map2, fig_el)
    report["cp11_figures"] = [str(fig_map), str(fig_map2), str(fig_el)]

    # ------------------------------------------------------------------
    # CP12 — Stop
    # ------------------------------------------------------------------
    _print_header("CHECKPOINT 12 — STOP / STRATEGY BOUNDARY")
    print("Approach B owns EstimatedEnergyIndex.")
    print("Approach C adds CircuitInfo zones + 400m lookahead.")
    print("C does not create a second battery/energy state.")
    print("ATTACK / SAVE / DEFEND are NOT implemented.")
    print("Gap-to-ahead / driver-ahead are NOT implemented.")
    report["cp12"] = {
        "strategy_implemented": False,
        "second_energy_state": False,
        "boundary": "stop after track-aware lookahead foundation",
    }

    raw_after = file_sha256(config.RAW_LAPS_CSV)
    clean_after = file_sha256(config.CLEAN_LAPS_CSV)
    report["raw_unchanged"] = raw_after == raw_hash
    report["clean_unchanged"] = clean_after == clean_hash
    print("raw unchanged:", report["raw_unchanged"])
    print("clean unchanged:", report["clean_unchanged"])

    config.CHECKPOINT_REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", config.CHECKPOINT_REPORT_JSON)
    try:
        samples.to_parquet(config.SAMPLE_PROXIES_PARQUET, index=False)
        print("wrote", config.SAMPLE_PROXIES_PARQUET)
    except Exception as exc:
        print("parquet save skipped:", exc)
        print("sample-level dump skipped (optional cache)")
    return report


if __name__ == "__main__":
    main()
