"""Validation and demo figures. Every energy label says estimated / not F1 SOC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reference_proxies(ref: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(ref["Distance"], ref["Speed"])
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title("CP8 — NOR lap 5 Distance traces (proxies are estimated, not ERS/SOC)")
    axes[1].plot(ref["Distance"], ref["Throttle"])
    axes[1].set_ylabel("Throttle")
    axes[2].plot(ref["Distance"], ref["DeployProxy"])
    axes[2].set_ylabel("DeployProxy")
    axes[3].plot(ref["Distance"], ref["HarvestProxy"])
    axes[3].set_ylabel("HarvestProxy")
    axes[3].set_xlabel("Distance (m)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cp8_reference_deploy_harvest.png")


def plot_two_driver_energy(energy_laps: pd.DataFrame, drivers=("NOR", "VER")) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for drv in drivers:
        sub = energy_laps.loc[energy_laps["Driver"] == drv].sort_values("LapNumber")
        if sub.empty:
            continue
        ax.plot(sub["LapNumber"], sub["EstimatedEnergyIndex_end"], label=drv)
    ax.set_xlabel("LapNumber")
    ax.set_ylabel("EstimatedEnergyIndex (end of lap)")
    ax.set_title("CP8 — EstimatedEnergyIndex vs lap (simulated, not F1 SOC)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cp8_two_driver_energy.png")


def plot_freeze_energy(energy_laps: pd.DataFrame, driver="NOR") -> Path:
    sub = energy_laps.loc[energy_laps["Driver"] == driver].sort_values("LapNumber")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["LapNumber"], sub["EstimatedEnergyIndex_end"], label="EstimatedEnergyIndex")
    freeze = sub.loc[sub["freeze_flag"] == True]  # noqa: E712
    if len(freeze):
        ax.scatter(freeze["LapNumber"], freeze["EstimatedEnergyIndex_end"], s=40, label="freeze lap (pit/SC/VSC/red)")
    ax.set_xlabel("LapNumber")
    ax.set_ylabel("EstimatedEnergyIndex")
    ax.set_title(f"CP8 — {driver} freeze laps should be flat (simulated, not F1 SOC)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cp8_freeze_energy.png")


def plot_throttle_vs_dlap(energy_laps: pd.DataFrame) -> Path:
    cal = energy_laps.loc[energy_laps.get("calibration_set", False) == True]  # noqa: E712
    if cal.empty:
        cal = energy_laps
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cal["mean_throttle"], cal["D_lap"], s=12, alpha=0.5)
    ax.set_xlabel("mean Throttle")
    ax.set_ylabel("D_lap")
    ax.set_title("CP8 — mean throttle vs D_lap (estimated demand, not ERS kW)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cp8_throttle_vs_dlap.png")


def plot_track_map(pos: pd.DataFrame, color_col: str, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pos["X"], pos["Y"], c=pos[color_col] if color_col in pos.columns else None, s=6, cmap="viridis")
    if color_col in pos.columns:
        fig.colorbar(sc, ax=ax, label=color_col)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, name)


def plot_energy_lookahead(samples: pd.DataFrame, energy_laps: pd.DataFrame, driver="NOR") -> Path:
    sub = energy_laps.loc[energy_laps["Driver"] == driver].sort_values("LapNumber")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(sub["LapNumber"], sub["EstimatedEnergyIndex_end"])
    axes[0].set_ylabel("EstimatedEnergyIndex")
    axes[0].set_title(f"CP11 — {driver} estimated energy + 400m lookahead (not F1 SOC)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(sub["LapNumber"], sub["LookaheadDeployProxy"], label="LookaheadDeployProxy")
    axes[1].plot(sub["LapNumber"], sub["LookaheadHarvestProxy"], label="LookaheadHarvestProxy")
    axes[1].set_xlabel("LapNumber")
    axes[1].set_ylabel("Lookahead proxy · m")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cp11_energy_lookahead.png")
