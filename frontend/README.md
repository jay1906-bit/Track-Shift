# ApexIQ frontend

Historical replay dashboard for the frozen TrackShift pipeline.

**Product name:** ApexIQ  
**Mode:** historical replay, not live inference.

The UI reads `public/replay.json`. It does not retrain the GBDT or call a live model.

## Setup

From `frontend/`:

```bash
npm install
npm run dev
```

Open the printed local URL (defaults to http://localhost:5173).

If `public/replay.json` is missing:

```bash
python scripts/build_frontend_replay.py
```

That script only copies frozen outputs. It does not change scientific CSVs.

## What the screen shows

- P(overtake within 30s) — frozen GBDT `p_hat`
- Simulated Energy Index — not battery %
- Along-track schematic — not GPS
- Immediate 400m track context
- ATTACK / SAVE and the actual `strategy_reason`

DEFEND is not implemented.
