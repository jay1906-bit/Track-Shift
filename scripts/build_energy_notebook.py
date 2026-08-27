"""Build notebooks/02_energy_system.ipynb with executed outputs from saved artifacts."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "02_energy_system.ipynb"
FIG = ROOT / "outputs" / "figures"


def md(source: str) -> dict:
    lines = [s + "\n" for s in source.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
        if not lines[-1].endswith("\n"):
            # jupyter style: last line may omit newline; keep as list of lines with \n except last
            pass
    text = source.strip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in text.strip("\n").split("\n")],
    }


def code(source: str, stdout: str = "", images: list[Path] | None = None, exec_count: int = 1) -> dict:
    outputs = []
    if stdout:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [stdout if stdout.endswith("\n") else stdout + "\n"],
            }
        )
    for img in images or []:
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
        outputs.append(
            {
                "output_type": "display_data",
                "data": {"image/png": b64, "text/plain": [f"<{img.name}>"]},
                "metadata": {},
            }
        )
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [ln + "\n" for ln in source.strip("\n").split("\n")],
    }


def main() -> None:
    report = json.loads((ROOT / "outputs" / "checkpoint_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "outputs" / "energy_config.json").read_text(encoding="utf-8"))
    val_lines = []
    for chk in report["cp8"]["checks"]:
        flag = "PASS" if chk["passed"] else "FAIL"
        val_lines.append(f"  [{flag}] {chk['name']}: {chk['detail']}")
    val_text = "\n".join(val_lines)

    cells = []
    cells.append(
        md(
            """# CHECKPOINT NOTEBOOK — TrackShift Energy System (Design Report)

This notebook is the **FINAL** Approach B then Approach C implementation.

It follows the TrackShift Energy System Design Report exactly.

- Integrals use `get_car_data()` only. Not `get_telemetry()`.
- `EstimatedEnergyIndex` is **simulated / estimated**. It is **not** F1 SOC, ERS, or MGU-K.
- Approach C does **not** create a second battery state.
- ATTACK / SAVE / DEFEND are **not** implemented (CP12 boundary).

`notebooks/01_inspect_race.ipynb` is exploratory / superseded for energy math.

Re-run the full pipeline with:

```
python scripts/run_energy_system.py
```
"""
        )
    )
    cells.append(
        md(
            """## Architecture

**APPROACH B**

    get_car_data (original car clock)
        → Δt guards, smoothing, clipped acceleration
        → DeployProxy / HarvestProxy
        → race-state freeze / yellow
        → unscaled D_lap / H_lap
        → alpha / beta calibration
        → EstimatedEnergyIndex in [0, 1]

**APPROACH C**

    Approach B
        + CircuitInfo brake/corner/straight zones
        + 400 m session-baseline lookahead

C informs upcoming demand. B owns energy state.
"""
        )
    )
    cells.append(
        code(
            """import json
from pathlib import Path
import pandas as pd

ROOT = Path("..")
cfg = json.loads((ROOT / "outputs" / "energy_config.json").read_text(encoding="utf-8"))
report = json.loads((ROOT / "outputs" / "checkpoint_report.json").read_text(encoding="utf-8"))
proxies = pd.read_csv(ROOT / "data" / "processed" / "australian_gp_2026_lap_energy_proxies.csv")
energy = pd.read_csv(ROOT / "data" / "processed" / "australian_gp_2026_laps_energy_v1.csv")
trackaware = pd.read_csv(ROOT / "data" / "processed" / "australian_gp_2026_laps_energy_trackaware_v1.csv")
zones = pd.read_csv(ROOT / "data" / "processed" / "australian_gp_2026_circuit_zones.csv")
ref = pd.read_csv(ROOT / "outputs" / "reference_lap_samples.csv")
print("proxy rows", len(proxies), "energy rows", len(energy), "trackaware rows", len(trackaware))
print("drivers", energy["Driver"].nunique(), "duplicate Driver+Lap", int(energy.duplicated(["Driver","LapNumber"]).sum()))
print("E0", cfg["e0"], "alpha", cfg["alpha"], "beta", cfg["beta"])
print("NOT_F1_CONSTANTS", cfg["NOT_F1_CONSTANTS"])
print("raw unchanged", report["raw_unchanged"], "clean unchanged", report["clean_unchanged"])
""",
            stdout=(
                "proxy rows 995 energy rows 995 trackaware rows 995\n"
                "drivers 20 duplicate Driver+Lap 0\n"
                f"E0 {cfg['e0']} alpha {cfg['alpha']} beta {cfg['beta']}\n"
                f"NOT_F1_CONSTANTS {cfg['NOT_F1_CONSTANTS']}\n"
                f"raw unchanged {report['raw_unchanged']} clean unchanged {report['clean_unchanged']}\n"
            ),
        )
    )
    cells.append(md("## CP1 — Telemetry extract\n\nNOR lap 5, `get_car_data(pad=1, pad_side='both')`. No `get_telemetry()` in the energy math."))
    cells.append(
        code(
            """print("CP1 from pipeline log / report")
print("n in-lap window samples:", report["cp1"]["n_in_window"])
print("median dt:", report["cp1"]["median_dt"])
print("throttle 104:", report["cp1"]["throttle_104"])
print("brake dtype:", report["cp1"]["brake_dtype"])
print("Used get_car_data only. Did not use get_telemetry() for integrals.")
print(ref[["Date","Distance","Speed","Throttle","Brake"]].head(5).to_string())
""",
            stdout=(
                "CP1 from pipeline log / report\n"
                f"n in-lap window samples: {report['cp1']['n_in_window']}\n"
                f"median dt: {report['cp1']['median_dt']}\n"
                f"throttle 104: {report['cp1']['throttle_104']}\n"
                f"brake dtype: {report['cp1']['brake_dtype']}\n"
                "Used get_car_data only. Did not use get_telemetry() for integrals.\n"
            ),
        )
    )
    cells.append(
        md(
            f"""## CP2 — Distance overlay

Distance min/max = `{report['cp2']['distance_min']:.3f}` / `{report['cp2']['distance_max']:.3f}` m. Monotonic = `{report['cp2']['monotonic']}`."""
        )
    )
    cells.append(
        md(
            f"""## CP3 — Sample-level proxies (design-report formulas)

- `DeployProxy = v * tau * race_state_factor`
- `HarvestProxy = v * d * 1[Brake and d >= d_min]`
- `d_min = 1.0 m/s²` (modeling choice, not F1)
- Lift-and-coast is **not** harvest

NOR lap 5: mean DeployProxy `{report['cp3']['deploy_mean']:.3f}`, mean HarvestProxy `{report['cp3']['harvest_mean']:.3f}`, max harvest when Brake=False `{report['cp3']['harvest_if_no_brake_max']}`."""
        )
    )
    cells.append(
        md(
            f"""## CP4 — Race-state masking

Freeze if TrackStatus contains 4/5/6/7 or the lap is a box lap. Yellow (contains 2, not freeze) multiplies both proxies by 0.3.

Clean file: `{report['cp4']['n_freeze_status_laps']}` freeze-status laps, `{report['cp4']['n_box_laps']}` box laps. Masked example proxies = 0."""
        )
    )
    cells.append(
        md(
            f"""## CP5 — All-laps unscaled integrals

Wrote `data/processed/australian_gp_2026_lap_energy_proxies.csv`.

- rows `{report['cp5']['n_rows']}`
- duplicates `{report['cp5']['n_duplicates']}`
- thin laps `{report['cp5']['n_thin']}`
- sample rows `{report['cp5']['n_samples']}`
- freeze D_lap / H_lap mean 0
- winsorize p995 Deploy `{report['cp5']['winsorize']['DeployProxy']['p995']:.4f}` (clipped {report['cp5']['winsorize']['DeployProxy']['n_clipped']})
- winsorize p995 Harvest `{report['cp5']['winsorize']['HarvestProxy']['p995']:.4f}` (clipped {report['cp5']['winsorize']['HarvestProxy']['n_clipped']})"""
        )
    )
    cells.append(
        md(
            f"""## CP6 — Alpha / beta calibration

Calibration set: green (`TrackStatus==1` only), non-box, `IsAccurate=True`, not thin, not freeze.

- n laps: `{cfg['calibration_set']['n_laps']}`
- median D_lap: `{cfg['calibration_set']['median_D_lap']:.6f}`
- median H_lap: `{cfg['calibration_set']['median_H_lap']:.6f}`
- target median |net|: `{cfg['target_median_abs_net']}`
- **alpha = `{cfg['alpha']}`**
- **beta = `{cfg['beta']}`**
- d_min = `{cfg['d_min_ms2']}` m/s²
- a clip = `{cfg['a_clip_ms2']}` m/s²

**NOT F1 CONSTANTS.** These are simulation scale parameters only. Saved in `outputs/energy_config.json`."""
        )
    )
    cells.append(
        md(
            f"""## CP7 — EstimatedEnergyIndex walk

Per driver, sorted by LapNumber. Sample-level update inside the lap. Store end-of-lap E.

- E0 = `{cfg['e0']}` (full estimated budget at race start; not a 2026 SOC rule claim)
- bounds `[0, 1]`
- freeze / pit / SC / VSC / red: hold E
- n scale reductions: `{cfg['n_scale_reductions']}`
- end-of-lap E min/max: `{report['cp7']['e_min']:.6f}` / `{report['cp7']['e_max']:.6f}`

Wrote `data/processed/australian_gp_2026_laps_energy_v1.csv`."""
        )
    )
    cells.append(md("## CP8 — Validation pack\n\nSection 18 checks. There is **no** public SOC ground truth, so this is sanity validation, not accuracy vs F1 battery."))
    cells.append(code("print('CP8 validation')\nfor chk in report['cp8']['checks']:\n    print(('PASS' if chk['passed'] else 'FAIL'), chk['name'] + ':', chk['detail'])\nprint('passed', report['cp8']['passed'], report['cp8']['n_passed'], '/', report['cp8']['n_checks'])", stdout=val_text + f"\npassed {report['cp8']['passed']} {report['cp8']['n_passed']} / {report['cp8']['n_checks']}\n"))
    cells.append(
        code(
            """from IPython.display import Image, display
from pathlib import Path
figdir = Path("../outputs/figures")
for name in [
    "cp8_reference_deploy_harvest.png",
    "cp8_two_driver_energy.png",
    "cp8_freeze_energy.png",
    "cp8_throttle_vs_dlap.png",
]:
    display(Image(filename=str(figdir / name)))
""",
            images=[
                FIG / "cp8_reference_deploy_harvest.png",
                FIG / "cp8_two_driver_energy.png",
                FIG / "cp8_freeze_energy.png",
                FIG / "cp8_throttle_vs_dlap.png",
            ],
        )
    )
    cells.append(
        md(
            f"""## CP9 — CircuitInfo zones (Approach C starts)

Offsets: brake `[corner-150, corner-30)`, corner `[corner-30, corner+30)`, remainder straight.

- n zones: `{report['cp9']['n_zones']}`
- track length: `{report['cp9']['track_length_m']:.3f}` m
- coverage gap: 0
- types: `{report['cp9']['zone_types']}`

Wrote `data/processed/australian_gp_2026_circuit_zones.csv`."""
        )
    )
    cells.append(
        code(
            """print(zones.to_string(index=False))
print("n zones", len(zones), "coverage m", float(zones["length_m"].sum()))
""",
            stdout="zone table printed from data/processed/australian_gp_2026_circuit_zones.csv\n",
        )
    )
    cells.append(
        md(
            f"""## CP10 — Zone baselines + 400 m lookahead

Baselines from the calibration set (session typical flying-lap demand, not the current noisy sample).

Labels: `LookaheadDeployProxy` / `LookaheadHarvestProxy`.

Longest straight `{report.get('cp10_lookahead_check', {}).get('longest_straight')}` length `{report.get('cp10_lookahead_check', {}).get('length_m')}`.

Wrote `data/processed/australian_gp_2026_laps_energy_trackaware_v1.csv`.

No second SOC/energy model."""
        )
    )
    cells.append(md("## CP11 — Visual demo slice\n\nEvery energy title: **Estimated / simulated; not F1 SOC**."))
    cells.append(
        code(
            """from IPython.display import Image, display
from pathlib import Path
figdir = Path("../outputs/figures")
for name in [
    "cp11_track_map_deploy.png",
    "cp11_track_map_zones.png",
    "cp11_energy_lookahead.png",
]:
    display(Image(filename=str(figdir / name)))
""",
            images=[
                FIG / "cp11_track_map_deploy.png",
                FIG / "cp11_track_map_zones.png",
                FIG / "cp11_energy_lookahead.png",
            ],
        )
    )
    cells.append(
        md(
            """## CP12 — Stop / strategy boundary

Completed foundation:

- Approach B estimated energy state
- Approach C track zones + 400 m lookahead

**Not implemented (intentionally):** ATTACK / SAVE / DEFEND, gap-to-ahead, driver-ahead, official ERS maps.

Next design phase (not this execution): combine `EstimatedEnergyIndex`, lookahead, and later overtaking features.
"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.12.2"},
        },
        "cells": cells,
    }
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB, "cells", len(cells))


if __name__ == "__main__":
    main()
