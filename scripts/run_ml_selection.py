"""TrackShift ML model-selection experiment.

Predicts P(on-track pairwise pass within 30 s of t0) from portable t0 features.
Leave-one-race-out on 7 development races; Belgium and Miami are locked until
the winner is frozen. Does not touch the event detector, energy system, or
source event tables.

Usage:
    python scripts/run_ml_selection.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trackshift import config  # noqa: E402
from trackshift.validate import file_sha256  # noqa: E402

SEED = 42
VERSION = "ml_selection_v1"
OUT_DIR = config.OUTPUTS_DIR / VERSION
FIG_DIR = OUT_DIR / "figures"

ML_SAFE_CSV = config.PROCESSED_DIR / "multirace_v1" / "overtake_events_ml_safe_v1.csv"
POOLED_CSV = config.PROCESSED_DIR / "multirace_v1" / "overtake_events_pooled_v1.csv"

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
NOISY_POS_THRESHOLD = 40

# Portable t0 allowlist. Circuit-identity proxies and absolute distance excluded.
NUMERIC_FEATURES = [
    "gap_at_start_m",
    "attacker_speed_at_start_kmh",
    "target_speed_at_attacker_position_kmh",
    "relative_speed_same_position_kmh",
    "closing_speed_at_start_kmh",
    "spatial_time_gap_s",
    "spatial_alignment_ok",
    "closing_speed_available",
    "track_position_frac",
    "distance_to_next_corner_m",
    "distance_to_next_straight_m",
    "tyre_age",
    "stint",
    "EstimatedEnergyIndex",
    "LookaheadDeployProxy",
    "LookaheadHarvestProxy",
    "lookahead_truncated",
]
CATEGORICAL_FEATURES = ["zone_type", "compound"]
FEATURE_ALLOWLIST = NUMERIC_FEATURES + CATEGORICAL_FEATURES
WINSOR_FEATURES = ["relative_speed_same_position_kmh", "closing_speed_at_start_kmh"]
BASELINE_FEATURES = [
    "gap_at_start_m",
    "closing_speed_at_start_kmh",
    "EstimatedEnergyIndex",
    "LookaheadDeployProxy",
]
EXCLUDED_FROM_X = [
    "event_id", "race_id", "circuit", "year", "attacker", "target", "target_number",
    "pair_id", "cluster_id", "reciprocal_event_id", "event_start_lap",
    "circuit_length_m", "corner_count", "longest_straight_m", "n_long_straights_over_500m",
    "track_distance_at_start_m", "LookaheadCoveredM",
]
PROTECTED_PATHS = {
    "energy_v1": config.ENERGY_V1_CSV,
    "trackaware": config.ENERGY_TRACKAWARE_CSV,
    "zones": config.ZONE_TABLE_CSV,
    "energy_config": config.ENERGY_CONFIG_JSON,
    "events_v4": config.OVERTAKE_EVENTS_V4_CSV,
    "ml_safe": ML_SAFE_CSV,
    "pooled": POOLED_CSV,
}

# Precommitted configs. No search.
LOGISTIC_C = 1.0
GBDT_PARAMS = {
    "max_depth": 3,
    "max_iter": 150,
    "learning_rate": 0.05,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": SEED,
}
# Close-call thresholds for "prefer logistic unless GBDT is clearly better".
PR_AUC_MARGIN = 0.02
BRIER_MARGIN = 0.005


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


def one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class FoldWinsorizer(BaseEstimator, TransformerMixin):
    """Clip named numeric columns at training-fold quantiles."""

    def __init__(self, columns: list[str], lower: float = 0.01, upper: float = 0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.bounds_ = {}
        for col in self.columns:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            lo = float(series.quantile(self.lower))
            hi = float(series.quantile(self.upper))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                continue
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for col, (lo, hi) in getattr(self, "bounds_", {}).items():
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").clip(lo, hi)
        return frame


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out = {
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "noisy_fold": n_pos < NOISY_POS_THRESHOLD,
    }
    if n_pos == 0 or n_neg == 0:
        out.update({
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
            "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        })
        return out
    out["pr_auc"] = float(average_precision_score(y, p))
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def nanmean(vals: list[float]) -> float:
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = frame.loc[:, FEATURE_ALLOWLIST].copy()
    for col in ("spatial_alignment_ok", "closing_speed_available", "lookahead_truncated"):
        if col in x.columns:
            x[col] = x[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype("object").fillna("missing")
    y = frame["overtake_success"].astype(int).to_numpy()
    return x, y


def make_pipeline(kind: str, feature_cols: list[str]) -> Pipeline:
    numeric = [c for c in feature_cols if c in NUMERIC_FEATURES]
    categorical = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    winsor = [c for c in WINSOR_FEATURES if c in numeric]
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))
        clf = LogisticRegression(
            penalty="l2",
            C=LOGISTIC_C,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=SEED,
        )
    elif kind == "gbdt":
        clf = HistGradientBoostingClassifier(
            **GBDT_PARAMS,
            class_weight=None,
        )
    else:
        raise ValueError(kind)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", one_hot())]), categorical)
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    steps: list = []
    if winsor:
        steps.append(("winsor", FoldWinsorizer(winsor)))
    steps.append(("pre", pre))
    steps.append(("clf", clf))
    return Pipeline(steps)


def loro(name: str, pipeline: Pipeline, dev: pd.DataFrame, feature_cols: list[str]) -> dict:
    oof_rows = []
    fold_rows = []
    for held in DEV_RACES:
        train = dev.loc[dev["race_id"] != held]
        valid = dev.loc[dev["race_id"] == held]
        x_tr, y_tr = prepare_xy(train)
        x_va, y_va = prepare_xy(valid)
        x_tr = x_tr.loc[:, feature_cols]
        x_va = x_va.loc[:, feature_cols]
        model = clone(pipeline)
        model.fit(x_tr, y_tr)
        p = model.predict_proba(x_va)[:, 1]
        m = metrics(y_va, p)
        m["model"] = name
        m["race_id"] = held
        m["circuit"] = str(valid["circuit"].iloc[0])
        fold_rows.append(m)
        part = valid.loc[:, ["event_id", "race_id", "circuit", "overtake_success"]].copy()
        part["model"] = name
        part["p_hat"] = p
        part["fold"] = held
        oof_rows.append(part)
        flag = " NOISY" if m["noisy_fold"] else ""
        print(
            f"  [{name:<12s}] {held:22s} n={m['n']:4d} pos={m['n_pos']:3d}  "
            f"PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.4f}  "
            f"LogLoss={m['log_loss']:.3f}  ROC={m['roc_auc']:.3f}{flag}",
            flush=True,
        )
    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    pooled = metrics(oof["overtake_success"].to_numpy(), oof["p_hat"].to_numpy())
    summary = {
        "model": name,
        "macro_pr_auc": nanmean(folds["pr_auc"].tolist()),
        "macro_brier": nanmean(folds["brier"].tolist()),
        "macro_log_loss": nanmean(folds["log_loss"].tolist()),
        "macro_roc_auc": nanmean(folds["roc_auc"].tolist()),
        "macro_pr_auc_stable_races": nanmean(
            folds.loc[~folds["noisy_fold"], "pr_auc"].tolist()
        ),
        "macro_brier_stable_races": nanmean(
            folds.loc[~folds["noisy_fold"], "brier"].tolist()
        ),
        "micro_pr_auc": pooled["pr_auc"],
        "micro_brier": pooled["brier"],
        "micro_log_loss": pooled["log_loss"],
        "micro_roc_auc": pooled["roc_auc"],
        "best_race_pr_auc": str(folds.loc[folds["pr_auc"].idxmax(), "race_id"]),
        "worst_race_pr_auc": str(folds.loc[folds["pr_auc"].idxmin(), "race_id"]),
        "median_race_pr_auc": float(folds["pr_auc"].median()),
        "pr_auc_spread": float(folds["pr_auc"].max() - folds["pr_auc"].min()),
        "brier_spread": float(folds["brier"].max() - folds["brier"].min()),
    }
    return {"folds": folds, "oof": oof, "summary": summary, "pooled": pooled}


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        # Bin counts via the same edges.
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        nonempty = counts > 0
        rows = []
        j = 0
        for i in range(n_bins):
            if not nonempty[i]:
                continue
            rows.append({
                "bin": i,
                "p_mean": float(mean_pred[j]),
                "y_rate": float(frac_pos[j]),
                "n": int(counts[i]),
                "abs_gap": float(abs(mean_pred[j] - frac_pos[j])),
            })
            j += 1
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def expected_calibration_error(rel: pd.DataFrame) -> float:
    if rel.empty or rel["n"].sum() == 0:
        return float("nan")
    w = rel["n"].to_numpy(dtype=float)
    return float(np.average(rel["abs_gap"].to_numpy(dtype=float), weights=w))


def save_reliability_plot(rel: pd.DataFrame, title: str, path: Path) -> None:
    if rel.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="perfect")
    ax.plot(rel["p_mean"], rel["y_rate"], marker="o", color="#1f4e79", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed overtake rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_winner(summaries: dict[str, dict], folds: dict[str, pd.DataFrame]) -> dict:
    lr = summaries["logistic"]
    gb = summaries["gbdt"]
    d_pr = gb["macro_pr_auc"] - lr["macro_pr_auc"]
    d_brier = lr["macro_brier"] - gb["macro_brier"]  # >0 GBDT better
    d_ll = lr["macro_log_loss"] - gb["macro_log_loss"]

    lr_folds = folds["logistic"]
    gb_folds = folds["gbdt"]
    merged = lr_folds.merge(
        gb_folds, on="race_id", suffixes=("_lr", "_gb")
    )
    stable = merged.loc[~merged["noisy_fold_lr"]]
    catastrophic = False
    if not stable.empty:
        catastrophic = bool(((stable["pr_auc_gb"] - stable["pr_auc_lr"]) < -0.10).any())

    close = abs(d_pr) < PR_AUC_MARGIN and abs(d_brier) < BRIER_MARGIN
    gbdt_clear = (
        d_pr >= PR_AUC_MARGIN
        and d_brier >= 0.002
        and not catastrophic
    )
    if gbdt_clear and not close:
        winner = "gbdt"
        why = (
            f"GBDT improved macro PR-AUC by {d_pr:+.3f} and macro Brier by {d_brier:+.4f} "
            "with no catastrophic drop on a stable-size LORO race."
        )
    else:
        winner = "logistic"
        if close:
            why = (
                f"Logistic and GBDT were close (Δ PR-AUC {d_pr:+.3f}, Δ Brier {d_brier:+.4f}). "
                "The precommitted rule prefers logistic: simpler, interpretable, naturally "
                "probabilistic."
            )
        elif catastrophic:
            why = (
                "GBDT did not show a consistent unseen-race gain without a large drop on a "
                "stable-size race; logistic selected."
            )
        else:
            why = (
                f"Logistic matched or beat GBDT on the co-primary metrics "
                f"(Δ PR-AUC {d_pr:+.3f}, Δ Brier {d_brier:+.4f})."
            )
    return {
        "winner": winner,
        "reason": why,
        "delta_macro_pr_auc_gbdt_minus_lr": d_pr,
        "delta_macro_brier_lr_minus_gbdt": d_brier,
        "delta_macro_log_loss_lr_minus_gbdt": d_ll,
        "close_call": close,
        "gbdt_clear_win": gbdt_clear,
        "catastrophic_gbdt_drop": catastrophic,
        "calibration": "none_extra",
        "calibration_note": (
            "No extra Platt layer. Logistic probabilities are used raw if logistic wins; "
            "GBDT probabilities are used raw if GBDT wins. Isotonic was not applied."
        ),
    }


def error_table(frame: pd.DataFrame, p_col: str = "p_hat") -> pd.DataFrame:
    out = frame.copy()
    y = out["overtake_success"].astype(int)
    p = out[p_col].astype(float)
    out["error_tag"] = "other"
    out.loc[(y == 0) & (p >= 0.50), "error_tag"] = "false_positive"
    out.loc[(y == 1) & (p < 0.20), "error_tag"] = "false_negative"
    out.loc[(y == 1) & (p >= 0.50), "error_tag"] = "true_positive"
    out.loc[(y == 0) & (p < 0.20), "error_tag"] = "true_negative"
    return out


def error_summary(tagged: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    merged = tagged.merge(
        feature_frame.loc[:, ["event_id"] + [c for c in FEATURE_ALLOWLIST if c in feature_frame.columns]],
        on="event_id",
        how="left",
    )
    rows = []
    for tag, grp in merged.groupby("error_tag"):
        rec = {
            "error_tag": tag,
            "n": int(len(grp)),
            "mean_p": float(grp["p_hat"].mean()),
            "median_gap_m": float(pd.to_numeric(grp.get("gap_at_start_m"), errors="coerce").median()),
            "median_closing_kmh": float(pd.to_numeric(grp.get("closing_speed_at_start_kmh"), errors="coerce").median()),
            "median_relative_kmh": float(pd.to_numeric(grp.get("relative_speed_same_position_kmh"), errors="coerce").median()),
            "median_energy": float(pd.to_numeric(grp.get("EstimatedEnergyIndex"), errors="coerce").median()),
            "median_lookahead_deploy": float(pd.to_numeric(grp.get("LookaheadDeployProxy"), errors="coerce").median()),
            "median_tyre_age": float(pd.to_numeric(grp.get("tyre_age"), errors="coerce").median()),
            "median_track_frac": float(pd.to_numeric(grp.get("track_position_frac"), errors="coerce").median()),
        }
        if "zone_type" in grp.columns and len(grp):
            rec["top_zone_type"] = str(grp["zone_type"].value_counts().index[0])
        if "race_id" in grp.columns and len(grp):
            rec["top_race"] = str(grp["race_id"].value_counts().index[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    if not ML_SAFE_CSV.exists():
        raise FileNotFoundError(f"ML-safe dataset not found: {ML_SAFE_CSV}")
    df = pd.read_csv(ML_SAFE_CSV)
    df["overtake_success"] = pd.to_numeric(df["overtake_success"], errors="coerce")
    if df["overtake_success"].isna().any():
        raise RuntimeError("ML-safe table has null labels; expected only 0/1 rows.")
    missing = [c for c in FEATURE_ALLOWLIST + ["race_id", "event_id", "overtake_success"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Required columns missing from ML-safe table: {missing}")
    return df


def main() -> int:
    t0 = time.time()
    for d in (OUT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    header("INTEGRITY")
    before = hashes()
    for name, digest in before.items():
        print(f"  {name:16s} {digest[:16]}...")

    header("DATASET")
    df = load_dataset()
    print(f"  path      : {ML_SAFE_CSV}")
    print(f"  rows      : {len(df)}")
    print(f"  positives : {int((df.overtake_success == 1).sum())}")
    print(f"  negatives : {int((df.overtake_success == 0).sum())}")
    print(f"  races     : {sorted(df.race_id.unique())}")
    print(f"  n_races   : {df.race_id.nunique()}")

    unknown = set(df.race_id.unique()) - set(DEV_RACES) - set(TEST_RACES)
    missing_split = [r for r in DEV_RACES + TEST_RACES if r not in set(df.race_id.unique())]
    if unknown or missing_split:
        raise RuntimeError(f"Race split mismatch. unknown={unknown} missing={missing_split}")

    future_or_id = [
        c for c in df.columns
        if c in EXCLUDED_FROM_X or c.endswith("_audit") or c in {
            "event_end_time", "event_duration_s", "pass_after_horizon", "actual_swap_delay_s",
        }
    ]
    leaked = [c for c in FEATURE_ALLOWLIST if c in future_or_id]
    if leaked:
        print("STOP: proposed feature is future/circuit-identity derived:", leaked)
        return 2

    header("FINAL FEATURE LIST")
    for col in FEATURE_ALLOWLIST:
        print(f"  {col}")
    print("  baseline subset:", ", ".join(BASELINE_FEATURES))
    print("  excluded circuit-identity / absolute distance / covered-m")

    dev = df.loc[df["race_id"].isin(DEV_RACES)].copy()
    test = df.loc[df["race_id"].isin(TEST_RACES)].copy()
    print(f"\n  development rows : {len(dev)}  pos={int((dev.overtake_success == 1).sum())}")
    print(f"  locked test rows : {len(test)}  pos={int((test.overtake_success == 1).sum())}")

    models = {
        "baseline": make_pipeline("logistic", BASELINE_FEATURES),
        "logistic": make_pipeline("logistic", FEATURE_ALLOWLIST),
        "gbdt": make_pipeline("gbdt", FEATURE_ALLOWLIST),
    }
    feature_sets = {
        "baseline": BASELINE_FEATURES,
        "logistic": FEATURE_ALLOWLIST,
        "gbdt": FEATURE_ALLOWLIST,
    }

    loro_out = {}
    for name, pipe in models.items():
        header(f"LORO  {name}")
        loro_out[name] = loro(name, pipe, dev, feature_sets[name])
        s = loro_out[name]["summary"]
        print(
            f"  MACRO  PR-AUC={s['macro_pr_auc']:.3f}  Brier={s['macro_brier']:.4f}  "
            f"LogLoss={s['macro_log_loss']:.3f}  ROC={s['macro_roc_auc']:.3f}"
        )
        print(
            f"  MICRO  PR-AUC={s['micro_pr_auc']:.3f}  Brier={s['micro_brier']:.4f}  "
            f"LogLoss={s['micro_log_loss']:.3f}"
        )

    header("CALIBRATION (OOF, development)")
    rel_tables = {}
    for name in models:
        oof = loro_out[name]["oof"]
        rel = reliability_table(oof["overtake_success"].to_numpy(), oof["p_hat"].to_numpy())
        rel_tables[name] = rel
        ece = expected_calibration_error(rel)
        loro_out[name]["summary"]["ece_oof"] = ece
        print(f"  {name:<12s} ECE={ece:.4f}")
        if not rel.empty:
            rel.to_csv(OUT_DIR / f"calibration_bins_{name}.csv", index=False)
            save_reliability_plot(
                rel, f"{name} OOF reliability", FIG_DIR / f"reliability_{name}.png"
            )

    header("SELECTION")
    summaries = {k: v["summary"] for k, v in loro_out.items()}
    decision = select_winner(
        {k: summaries[k] for k in ("logistic", "gbdt")},
        {k: loro_out[k]["folds"] for k in ("logistic", "gbdt")},
    )
    # Baseline is not a selection candidate.
    winner = decision["winner"]
    print(f"  winner : {winner}")
    print(f"  reason : {decision['reason']}")
    print("  Belgium/Miami were not used.")

    header("FINAL TRAIN (7 development races) + LOCKED TEST")
    x_dev, y_dev = prepare_xy(dev)
    x_dev = x_dev.loc[:, feature_sets[winner]]
    final_pipe = clone(models[winner])
    final_pipe.fit(x_dev, y_dev)

    test_parts = []
    test_metrics = {}
    for race in TEST_RACES:
        sub = test.loc[test["race_id"] == race]
        x_te, y_te = prepare_xy(sub)
        x_te = x_te.loc[:, feature_sets[winner]]
        p = final_pipe.predict_proba(x_te)[:, 1]
        m = metrics(y_te, p)
        m["model"] = winner
        m["race_id"] = race
        m["circuit"] = str(sub["circuit"].iloc[0])
        test_metrics[race] = m
        part = sub.loc[:, ["event_id", "race_id", "circuit", "overtake_success"]].copy()
        part["p_hat"] = p
        part["split"] = "locked_test"
        test_parts.append(part)
        print(
            f"  {race:22s} n={m['n']:4d} pos={m['n_pos']:3d}  "
            f"PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.4f}  "
            f"LogLoss={m['log_loss']:.3f}  ROC={m['roc_auc']:.3f}"
        )
    test_pred = pd.concat(test_parts, ignore_index=True)
    pooled_test = metrics(test_pred["overtake_success"].to_numpy(), test_pred["p_hat"].to_numpy())
    print(
        f"  pooled test          n={pooled_test['n']:4d} pos={pooled_test['n_pos']:3d}  "
        f"PR-AUC={pooled_test['pr_auc']:.3f}  Brier={pooled_test['brier']:.4f}"
    )
    rel_test = reliability_table(
        test_pred["overtake_success"].to_numpy(), test_pred["p_hat"].to_numpy(), n_bins=8
    )
    if not rel_test.empty:
        rel_test.to_csv(OUT_DIR / "calibration_bins_locked_test.csv", index=False)
        save_reliability_plot(
            rel_test, f"{winner} locked-test reliability", FIG_DIR / "reliability_locked_test.png"
        )

    header("ERROR ANALYSIS")
    oof_w = loro_out[winner]["oof"]
    tagged_oof = error_table(oof_w)
    tagged_test = error_table(test_pred)
    feat_dev = pd.concat([dev.loc[:, ["event_id"] + FEATURE_ALLOWLIST],
                          test.loc[:, ["event_id"] + FEATURE_ALLOWLIST]], ignore_index=True)
    err_oof = error_summary(tagged_oof, feat_dev)
    err_test = error_summary(tagged_test, feat_dev)
    print("  OOF error tags:")
    print(err_oof.loc[:, ["error_tag", "n", "median_gap_m", "median_closing_kmh", "median_energy"]].to_string(index=False))
    print("  Locked-test error tags:")
    print(err_test.loc[:, ["error_tag", "n", "median_gap_m", "median_closing_kmh", "median_energy"]].to_string(index=False))

    header("SAVE")
    fold_all = pd.concat([loro_out[k]["folds"] for k in models], ignore_index=True)
    oof_all = pd.concat([loro_out[k]["oof"] for k in models], ignore_index=True)
    fold_all.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)
    oof_all.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    tagged_oof.to_csv(OUT_DIR / "error_analysis_oof.csv", index=False)
    tagged_test.to_csv(OUT_DIR / "error_analysis_test.csv", index=False)
    err_oof.to_csv(OUT_DIR / "error_summary_oof.csv", index=False)
    err_test.to_csv(OUT_DIR / "error_summary_test.csv", index=False)
    test_pred.to_csv(OUT_DIR / "locked_test_predictions.csv", index=False)
    pd.DataFrame({"column": FEATURE_ALLOWLIST, "role": "feature_t0_ml"}).to_csv(
        OUT_DIR / "ml_feature_manifest.csv", index=False
    )

    import sklearn
    after = hashes()
    unchanged = {k: before.get(k) == after.get(k) for k in before}
    payload = {
        "version": VERSION,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "path": str(ML_SAFE_CSV),
            "n_rows": int(len(df)),
            "n_pos": int((df.overtake_success == 1).sum()),
            "n_neg": int((df.overtake_success == 0).sum()),
            "n_races": int(df.race_id.nunique()),
            "races": sorted(df.race_id.unique().tolist()),
            "dev_n": int(len(dev)),
            "dev_pos": int((dev.overtake_success == 1).sum()),
            "test_n": int(len(test)),
            "test_pos": int((test.overtake_success == 1).sum()),
        },
        "split": {"development": DEV_RACES, "locked_test": TEST_RACES},
        "features": {
            "allowlist": FEATURE_ALLOWLIST,
            "baseline": BASELINE_FEATURES,
            "excluded": EXCLUDED_FROM_X,
            "winsorized": WINSOR_FEATURES,
        },
        "hyperparameters": {
            "logistic_C": LOGISTIC_C,
            "gbdt": GBDT_PARAMS,
            "class_weight": None,
        },
        "loro_summaries": summaries,
        "loro_folds": {k: loro_out[k]["folds"].to_dict(orient="records") for k in models},
        "selection": decision,
        "winner": winner,
        "locked_test": {"per_race": test_metrics, "pooled": pooled_test},
        "error_summary_oof": err_oof.to_dict(orient="records"),
        "error_summary_test": err_test.to_dict(orient="records"),
        "integrity": {
            "hashes_before": before,
            "hashes_after": after,
            "protected_files_unchanged": unchanged,
            "all_protected_unchanged": all(unchanged.values()),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "ml_selection_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"  wrote {OUT_DIR}")
    print(f"  protected files unchanged: {payload['integrity']['all_protected_unchanged']}")
    print(f"  elapsed {payload['elapsed_s']}s")
    print("\n  No strategy engine. No UI. No detector change. No dataset overwrite.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
