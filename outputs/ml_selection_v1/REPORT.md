# TrackShift ML model-selection experiment

**Version:** `ml_selection_v1`  
**Seed:** 42  
**sklearn:** 1.9.0  
**Script:** `scripts/run_ml_selection.py`  
**Source data (read-only):** `data/processed/multirace_v1/overtake_events_ml_safe_v1.csv`

Protected energy files, Australia v4 events, and the multi-race tables were hashed before and after. **Unchanged.**

Belgium and Miami were not used for selection, preprocessing, or calibration.

---

### DATASET

| | |
|---|---:|
| Races | 9 |
| Labelled events | **6,785** |
| Positives | **1,528** (22.5%) |
| Negatives | **5,257** |
| Development (7 races) | 5,473 rows / 1,406 positives |
| Locked test (Belgium + Miami) | 1,312 rows / 122 positives |

Counts match the frozen multi-race headline. No dataset was regenerated.

---

### FEATURES

**Allowlist used as X (19 portable t0 columns):**

gap_at_start_m, attacker_speed_at_start_kmh, target_speed_at_attacker_position_kmh, relative_speed_same_position_kmh, closing_speed_at_start_kmh, spatial_time_gap_s, spatial_alignment_ok, closing_speed_available, track_position_frac, distance_to_next_corner_m, distance_to_next_straight_m, tyre_age, stint, EstimatedEnergyIndex, LookaheadDeployProxy, LookaheadHarvestProxy, lookahead_truncated, zone_type, compound

**Domain baseline subset:** gap_at_start_m, closing_speed_at_start_kmh, EstimatedEnergyIndex, LookaheadDeployProxy

**Removed from X, and why:**

| Column / family | Why |
|---|---|
| attacker, target, ids, pair_id, reciprocal_event_id | identity / grouping, not portable situation features |
| race_id, circuit, year | grouping only; would leak circuit identity |
| circuit_length_m, corner_count, longest_straight_m, n_long_straights_over_500m | constant within race = circuit ID |
| track_distance_at_start_m | not portable across circuits |
| LookaheadCoveredM | near-constant 400 m; truncation flag kept |
| episode / audit / outcome fields | future or label-derived; not in the ML-safe matrix |

Relative speed and closing speed were winsorized at the **training-fold** 1st/99th percentiles. Imputation, one-hot encoding, and (for logistic) scaling were fit on the training races of each fold only. No `class_weight`.

---

### VALIDATION

Leave-one-race-out on the 7 development races:

Australia, China, Japan, Canada, Monaco, Austria, Netherlands.

Each fold trains on 6 races and validates on 1. Same folds for every model.

Canada, Monaco, Austria have 19–23 positives and are marked **NOISY**. They are reported but did not single-handedly decide the winner. Stable-size folds (pos ≥ 40): Australia (61), China (545), Japan (545), Netherlands (194).

---

### MODEL RESULTS (7-race LORO)

| Model | Macro PR-AUC | Macro Brier | Macro log loss | Macro ROC-AUC | OOF ECE | Micro PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Domain baseline (4-feature logistic) | 0.554 | 0.116 | 0.380 | 0.855 | 0.028 | 0.579 |
| L2 logistic (full X, C=1) | 0.462 | 0.126 | 0.400 | 0.820 | 0.044 | 0.556 |
| Shallow GBDT (depth 3, 150 iter, lr 0.05) | **0.567** | **0.098** | **0.316** | **0.878** | **0.029** | **0.694** |

Stable-race-only macro PR-AUC: baseline 0.621 · logistic 0.587 · **GBDT 0.693**.

Calibration: extra Platt/isotonic **not** applied. GBDT OOF ECE (0.029) matched the domain baseline and beat full logistic (0.044). Raw GBDT probabilities were kept.

Random Forest was not run.

**Note:** The 4-feature domain baseline beat **full** logistic on every co-primary metric. Extra linear features, especially on Monaco, made logistic worse. GBDT still beat both the baseline and full logistic on Brier, log loss, and stable-race PR-AUC.

---

### PER-RACE RESULTS

PR-AUC (NOISY = fewer than 40 positives):

| Race | n | pos | prev. | Baseline | Logistic | GBDT |
|---|---:|---:|---:|---:|---:|---:|
| Australia | 453 | 61 | 13.5% | 0.403 | 0.428 | **0.501** |
| China | 1213 | 545 | 44.9% | 0.704 | 0.700 | **0.780** |
| Japan | 1492 | 545 | 36.5% | 0.592 | 0.621 | **0.684** |
| Canada NOISY | 363 | 19 | 5.2% | 0.387 | 0.377 | **0.396** |
| Monaco NOISY | 665 | 19 | 2.9% | **0.503** | 0.084 | 0.371 |
| Austria NOISY | 538 | 23 | 4.3% | **0.507** | 0.427 | 0.430 |
| Netherlands | 749 | 194 | 25.9% | 0.784 | 0.598 | **0.808** |

Brier (lower is better):

| Race | Baseline | Logistic | GBDT |
|---|---:|---:|---:|
| Australia | 0.111 | 0.115 | **0.097** |
| China | 0.227 | 0.218 | **0.194** |
| Japan | 0.208 | 0.206 | **0.184** |
| Canada NOISY | 0.058 | 0.048 | **0.041** |
| Monaco NOISY | 0.035 | 0.083 | **0.028** |
| Austria NOISY | 0.053 | 0.053 | **0.037** |
| Netherlands | 0.120 | 0.160 | **0.107** |

GBDT: best PR-AUC race = Netherlands (0.808); worst = Monaco (0.371, noisy). Median PR-AUC 0.501. Spread 0.436. **No catastrophic drop vs logistic on a stable-size race** (Australia/China/Japan/Netherlands all improve).

Full logistic **collapsed on Monaco** (PR-AUC 0.084 vs baseline 0.503). That fold is noisy; it is reported, not hidden, and was not used as a tie-break against GBDT (GBDT already won the stable races).

---

### WINNER

**Shallow gradient-boosted trees** (`HistGradientBoostingClassifier`, max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=40, l2=1.0).

**Why (precommitted hierarchy):**

1. Macro PR-AUC: GBDT 0.567 vs logistic 0.462 (**+0.105**, above the 0.02 “close call” margin).
2. Macro Brier: GBDT 0.098 vs logistic 0.126 (**+0.028**).
3. Macro log loss and ECE also favour GBDT.
4. On every stable-size LORO race GBDT ≥ logistic on PR-AUC and Brier.
5. Complexity: GBDT is the heavier model, but the gain is not a tiny decimal — the prefer-logistic-if-close rule does not apply.

The domain baseline is **not** the selected production model. It remains the honesty check: ranking vs a 4-feature logistic is only modestly better in macro PR-AUC (+0.013), while **probability quality (Brier/log loss) improves clearly**, and stable-race PR-AUC improves from 0.621 to 0.693.

No extra calibration layer.

Belgium and Miami were not inspected until this freeze.

---

### FINAL TEST (confirmation only)

Frozen GBDT trained on all 7 development races, scored **once**.

| | n | pos | prev. | PR-AUC | Brier | Log loss | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Belgium | 675 | 75 | 11.1% | 0.648 | 0.066 | 0.229 | 0.926 |
| Miami | 637 | 47 | 7.4% | 0.515 | 0.059 | 0.215 | 0.901 |
| Pooled | 1312 | 122 | 9.3% | 0.576 | 0.063 | — | — |

This is a two-circuit confirmation, not proof of universal generalisation. Miami’s 47 positives make its PR-AUC noisier than Belgium’s.

---

### ERROR ANALYSIS (frozen GBDT; no refit)

**Development OOF** (high-P false positive = P≥0.50 and no pass; false negative = P<0.20 and pass):

| Tag | n | median gap (m) | median closing (km/h) | median energy | typical zone |
|---|---:|---:|---:|---:|---|
| True negative | 3042 | 96 | +47 | 0.89 | brake |
| True positive | 728 | 13 | −60 | 0.94 | straight |
| False positive | 311 | 13 | −24 | 0.94 | straight |
| False negative | 247 | 86 | +19 | 0.91 | brake |

False positives look like true positives: already close, on a straight, energy high — the cars simply did not complete a corroborated pass in 30 s. False negatives look like true negatives: large gap, not closing. The model is mostly a **close-and-closing** detector; it misses passing from farther back and over-calls close battles that stay behind.

**Locked test:** 32 FP (close/straight, mostly Miami), 13 FN (larger gap, often early-lap Miami). Same pattern. Not used to change the model.

---

### LIMITATIONS

- Only **9 race groups**; events inside a race are clustered (pairs, reciprocals).
- **China + Japan** hold most development positives; pooled/micro metrics overstate a “typical” GP.
- Canada / Monaco / Austria LORO folds are **noisy** (~19–23 positives). Full logistic failed badly on Monaco.
- Locked test is **two races** (122 positives). Do not claim all-circuit performance from that.
- GBDT vs the 4-feature baseline is a **clear Brier win**, a **modest macro PR-AUC win**. The value is better probabilities and better ranking on high-count races, not magic.
- Energy numbers in this table are the multi-race walk, not spliced frozen v4 energy.

---

### PPT-READY MODEL JUSTIFICATION

We compared a racing baseline, penalized logistic regression, and a shallow gradient-boosted tree using leave-one-race-out on seven development Grands Prix, with Belgium and Miami locked until the end. The tree model improved unseen-race ranking and probability scores over the full logistic model by a clear margin (macro PR-AUC 0.57 vs 0.46, macro Brier 0.098 vs 0.126), including on every development race with enough positives to be informative. A four-feature gap/closing/energy/lookahead logistic was a strong baseline, but the shallow tree was better calibrated in Brier/log loss and stronger on stable-size races, so it was frozen. One confirmation pass on Belgium and Miami (PR-AUC 0.65 and 0.51) was consistent with that choice; those two races were not used to pick the model.

---

### ARTIFACTS

All under `outputs/ml_selection_v1/`:

- `ml_selection_summary.json` — machine-readable full result
- `selected_model_config.json` — frozen winner
- `ml_feature_manifest.csv`
- `per_fold_metrics.csv`, `oof_predictions.csv`
- `locked_test_predictions.csv`
- `calibration_bins_*.csv`, `figures/reliability_*.png`
- `error_analysis_oof.csv`, `error_analysis_test.csv`, `error_summary_*.csv`
- this report

No strategy engine, UI, detector change, or extra model search was run.
