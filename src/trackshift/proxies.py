"""Sample-level DeployProxy and HarvestProxy (design-report formulas).

Not watts, not ERS, not MGU-K, not recovered kJ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_proxies(tel: pd.DataFrame, race_state_factor: float) -> pd.DataFrame:
    out = tel.copy()
    valid = out["SampleValid"].fillna(False).astype(bool)
    v = out["SpeedMSSmoothed"].to_numpy(dtype=float)
    tau = out["tau"].to_numpy(dtype=float)
    d = out["DecelerationMS2"].to_numpy(dtype=float)
    brake = out["Brake"].fillna(False).astype(bool).to_numpy()

    b = brake & np.isfinite(d) & (d >= config.D_MIN_MS2)
    allows = float(race_state_factor)

    deploy = np.where(valid & np.isfinite(v) & np.isfinite(tau), v * tau * allows, 0.0)
    harvest = np.where(valid & np.isfinite(v) & np.isfinite(d) & b, v * d * allows, 0.0)

    # Lift-and-coast diagnostic only — never added into E
    lift = np.where(
        valid & (~brake) & np.isfinite(v) & np.isfinite(d) & (d > 0),
        v * d,
        0.0,
    )

    out["RaceStateFactor"] = allows
    out["HarvestGate"] = b
    out["DeployProxy"] = deploy
    out["HarvestProxy"] = harvest
    out["LiftDecelProxy"] = lift
    return out


def winsorize_session_proxies(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clip DeployProxy / HarvestProxy at the session 99.5th percentile."""
    out = samples.copy()
    report = {}
    for col in ("DeployProxy", "HarvestProxy"):
        finite = pd.to_numeric(out[col], errors="coerce")
        cap = float(finite.quantile(config.WINSOR_QUANTILE))
        if not np.isfinite(cap) or cap < 0:
            cap = float(finite.max()) if np.isfinite(finite.max()) else 0.0
        n_clip = int((finite > cap).sum())
        out[col] = finite.clip(upper=cap)
        report[col] = {"p995": cap, "n_clipped": n_clip}
    return out, report
