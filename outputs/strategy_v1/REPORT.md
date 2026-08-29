# TrackShift strategy engine v1

**Version:** `strategy_v1`  
**Frozen ML:** GBDT OOF + locked-test probabilities from `outputs/ml_selection_v1/`  
**DEFEND:** not implemented

Protected energy/event/ML artifacts hashed before and after: **UNCHANGED**.

---

### STRATEGY SPECIFICATION

Two actions only: **ATTACK** and **SAVE**.

1. Invalid `p_hat` (missing or outside [0, 1]) → `INVALID` (no action).
2. Hard infeasibility → SAVE
   - missing energy
   - `EstimatedEnergyIndex` < `0.1` (resource floor)
   - lookahead truncated/unusable **and** energy < `0.5`
3. Opportunity gate → SAVE if `p_hat` < `0.45`
4. Immediate spend window → SAVE unless the next 400 m is a deploy-compatible context:
   - `zone_type == straight`, **or**
   - deploy share = LookaheadDeployProxy / (Deploy + Harvest) ≥ `0.4`
5. Otherwise ATTACK.

High raw deploy does **not** force SAVE. Gap and closing speed are not strategy gates.

---

### THRESHOLDS

Selected on the 7 development races only. Belgium and Miami unused.

| Parameter | Value | Development-only rationale |
|---|---:|---|
| `p_hat_min` | 0.45 | Middle of the precommitted set {0.35, 0.45, 0.55}. OOF `p_hat` median is 0.12; 0.45 sits between the 75th (0.38) and 80th (0.48) percentiles — a selective upper-tail opportunity gate, not 0.70. |
| `energy_floor` | 0.1 | Low-tail of development energy (~5.8% of events ≤ 0.10). 0.05 is rarer; 0.20 starts cutting into the body. Energy is an affordability veto, not a second model. |
| `energy_adequate_if_lookahead_unusable` | 0.5 | Truncation is rare (13 development events). Require clearly adequate energy when the 400 m window is incomplete. |
| `deploy_share_min` | 0.4 | Row-local mix, not a raw cross-race LookaheadDeployProxy cut. Harvest proxies are systematically larger, so 0.50 would be biased; 0.40 means deploy is a substantial fraction of the 400 m mix. Straights remain spend-compatible even with a lower mix. |

**Not selected by overtake_success, F1, or pass count.**

---

### FILES CHANGED

- `src/trackshift/strategy.py` (new)
- `src/trackshift/config.py` (output paths only)
- `scripts/run_strategy_engine.py` (new)
- `tests/test_strategy.py` (new)
- `outputs/strategy_v1/` (new; does not overwrite event/energy/ML files)

---

### UNIT TESTS

Passed.

---

### DEVELOPMENT REPLAY

n = 5473  
ATTACK = 858 (15.7%)  
SAVE = 4615 (84.3%)

Median p_hat: ATTACK 0.677 vs SAVE 0.080  
Median energy: ATTACK 0.946 vs SAVE 0.908  
Energy-floor vetoes (all rows, hierarchy-first): 319  
Energy-floor vetoes among p-gate passers: 33  
Spend-context vetoes among p-gate passers: 298  
Lookahead-unusable vetoes: 1

ATTACK median p_hat is higher than SAVE: **True**.

| Race | n | ATTACK | SAVE | ATTACK % | median p ATTACK | median p SAVE |
|---|---:|---:|---:|---:|---:|---:|
| australia_2026_R | 453 | 56 | 397 | 12.4% | 0.589 | 0.095 |
| austria_2026_R | 538 | 23 | 515 | 4.3% | 0.533 | 0.039 |
| canada_2026_R | 363 | 17 | 346 | 4.7% | 0.535 | 0.026 |
| china_2026_R | 1213 | 283 | 930 | 23.3% | 0.715 | 0.148 |
| japan_2026_R | 1492 | 393 | 1099 | 26.3% | 0.707 | 0.151 |
| monaco_2026_R | 665 | 8 | 657 | 1.2% | 0.496 | 0.061 |
| netherlands_2026_R | 749 | 78 | 671 | 10.4% | 0.615 | 0.082 |

Zone share among ATTACK: {'straight': 0.5921, 'brake': 0.2284, 'corner': 0.1795}

`overtake_success` was not used to decide. Analysis-only pass rate: ATTACK 0.652, SAVE 0.184.

---

### LOCKED TEST CONFIRMATION

One shot on frozen thresholds. Not used to change the engine.

n = 1312  
ATTACK = 95 (7.2%)  
SAVE = 1217 (92.8%)

Median p_hat: ATTACK 0.543 vs SAVE 0.060

| Race | n | ATTACK | SAVE | ATTACK % |
|---|---:|---:|---:|---:|
| belgium_2026_R | 675 | 43 | 632 | 6.4% |
| miami_2026_R | 637 | 52 | 585 | 8.2% |

Two races are confirmation only, not universal generalisation.

---

### ABLATION

| Ablation | ATTACK % | n ATTACK | energy vetoes among p-passers | spend vetoes among p-passers | median p ATTACK |
|---|---:|---:|---:|---:|---:|
| A: P-only | 21.7% | 1190 | 0 | 0 | 0.686 |
| B: P + energy | 21.1% | 1156 | 33 | 0 | 0.688 |
| C: P + energy + 400 m context | 15.7% | 858 | 33 | 298 | 0.677 |

Energy changes few ATTACK decisions because the simulated index is usually near the top of [0, 1]: among events that already clear the p-gate, the floor vetoes only a handful. Immediate 400 m context (straight or deploy-share) removes a larger slice of high-p fights that sit in harvest-heavy brake/corner windows. That is reported, not dressed up.

DEFEND ablation was not run.

---

### LIMITATIONS

- No causal behind-car signal; DEFEND is not implemented.
- `EstimatedEnergyIndex` is a simulated 0–1 resource index, not F1 SOC / ERS.
- Lookahead is the next ~400 m (~6–14 s), not later-lap or race demand.
- Energy is often saturated; affordability vetoes are rare and should stay rare.
- The engine is event-triggered at detector t0, not a full-race optimiser.
- Development replay uses frozen LORO OOF probabilities; locked test uses the saved one-shot GBDT scores. The GBDT was not retrained here.

---

### PPT-READY DESCRIPTION

TrackShift does not attack whenever an overtake looks possible. A frozen model estimates the probability that a pairwise pass completes within 30 seconds. A separate rule layer then asks whether the attacker can afford to spend in the next 400 metres of track — the immediate passing window, not a full-lap energy plan. It attacks only when that probability clears a development-chosen opportunity gate and the simulated energy budget plus immediate track mix support a deploy; otherwise it saves. We do not observe the car behind, so there is no DEFEND mode.

---

### CANDIDATE GRID (development process metrics)

See `threshold_grid.csv`. Selected row: p_hat_min=0.45, energy_floor=0.1, deploy_share_min=0.4.
