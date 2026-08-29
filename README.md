# ApexIQ

A historical replay prototype that estimates overtaking opportunity and converts it into a resource-aware ATTACK/SAVE decision.

This repository contains the TrackShift science pipeline and the ApexIQ frontend. The public demo is **historical replay of frozen, precomputed predictions**. It is **not live telemetry** and **not live inference**.

## What ApexIQ does

At a genuine on-track battle (`t0`), ApexIQ estimates the probability that a corroborated pass completes within 30 seconds, then applies a deterministic rule layer using a simulated energy index and the next 400 m of track. The output is **ATTACK** or **SAVE**. DEFEND is not implemented.

## Pipeline

```
Event detection
     ↓
Feature extraction (19 t0 features)
     ↓
Frozen shallow GBDT
     ↓
P(pass within 30s)  →  p_hat
     ↓
Strategy layer (energy + 400 m context)
     ↓
ATTACK / SAVE
```

The UI does not retrain or call a model. It reads `frontend/public/replay.json`, which is copied from frozen strategy and ML artifacts.

## Dataset

Public FastF1 2026 race sessions (`R`), nine Grands Prix:

Australia, China, Japan, Miami, Canada, Monaco, Austria, Belgium, Netherlands.

| | |
|---|---:|
| Detected events | 10,435 |
| Labelled events | 6,785 |
| Successful passes | 1,528 |
| No-pass events | 5,257 |
| ML features at t0 | 19 |

An overtake is labelled successful only when race-distance displacement and DriverAhead both corroborate a pass inside 30 seconds of `t0`.

## Model

Shallow `HistGradientBoostingClassifier` (depth 3, 150 iterations, learning rate 0.05). Hyperparameters were precommitted; there was no grid search.

Compared with a 4-feature domain logistic baseline and a full L2 logistic model on leave-one-race-out development races. The GBDT was selected.

Development LORO (7 races): PR-AUC **0.567**, Brier **0.098**.

## Validation

- **Development:** 7 races, leave-one-race-out
- **Locked confirmation:** Belgium and Miami, unused for model or strategy selection

Locked test: Belgium PR-AUC **0.648**, Miami PR-AUC **0.515**, pooled PR-AUC **0.576**.

PR-AUC and Brier are the reported metrics. Accuracy is not used as the headline metric.

## Strategy

ATTACK / SAVE only. Gates (chosen on development OOF, not on locked races):

- `p_hat` below 0.45 → SAVE
- simulated energy index below 0.10 → SAVE
- next 400 m not a spend window → SAVE
- otherwise ATTACK, if `p_hat` is valid

`EstimatedEnergyIndex` is a **simulated 0–1 index**, not battery percentage, SOC, or ERS energy.

Development ATTACK rate ablation: P-only 21.7% → P+energy 21.1% → P+energy+400 m **15.7%**. This does not prove optimal race strategy.

## Replay / demo

The React + Vite app in `frontend/` is a static historical replay:

- Race Replay: circuit, battle, `p_hat`, energy, 400 m context, ATTACK/SAVE, historical outcome (when labelled)
- Model Validation: frozen LORO / locked / ablation numbers

Runtime needs only static files. No Python backend is required to view the demo.

## Run locally

**Frontend demo**

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173). The app loads `/replay.json` from `frontend/public/replay.json`.

If that file is missing:

```bash
python scripts/build_frontend_replay.py
```

That script copies frozen outputs. It does not retrain the model or rewrite scientific CSVs.

**Python pipeline** (optional; not required for the demo)

Python 3.12+, FastF1, and the packages used by `src/trackshift/`. Energy, events, ML, and strategy scripts live under `scripts/`. Do not rerun them unless you intend to regenerate frozen artifacts.

## Build the frontend

```bash
cd frontend
npm install
npm run build
```

Output: `frontend/dist/` (gitignored). Production hosting should serve that build with `replay.json` included from `public/`.

## Public demo

**frontend-zeta-pearl-9ctxk2h38z.vercel.app**

The deployed site is the same historical replay as local `frontend/`. No backend is required at runtime.
