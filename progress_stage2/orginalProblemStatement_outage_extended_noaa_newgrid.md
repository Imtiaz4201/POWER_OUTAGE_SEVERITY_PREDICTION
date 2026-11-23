# Process: `orginalProblemStatement_outage_extended_noaa_newgrid.ipynb`

**Base model:** extended outage data + NOAA storms + NWS watches / warnings.
**Proposed model:** base + new grid data (EIA-861 + FERC Form 1).

---

### 1. Load the datasets
- **Outages:** `Data/outage_extended.xlsx`, **4,279 outage reports**, 2000–2024.
  - Purdue rows 2000 – Jul 2016, unchanged.
  - DOE OE-417 rows Jul 2016 – 2024, added by `build_outage_extended.py`.
- **Storms:** NOAA Storm Events, 25 yearly CSV files (2000–2024), **1,552,468 storm records**. Only these columns are read: `STATE, BEGIN_DATE_TIME, END_DATE_TIME, EVENT_TYPE, MAGNITUDE, CZ_TYPE, CZ_FIPS`.
- **NWS watches / warnings / advisories:** `Data/nws_raw/nws_events.parquet`, downloaded from the Iowa Environmental Mesonet by `download_nws_warnings.py` (2005–2024, all states). Columns: `state, phenomena, significance, issue, init_expire, wfo, area`.
- **New grid data:** `Data/pudl_raw/` (PUDL parquet files):
  - EIA-861: reliability (SAIDI / SAIFI / CAIDI), smart meters, distribution circuits, utility → county service territory, sales
  - FERC Form 1: plant in service (overhead / underground lines), operating expenses (line maintenance)
  - PUDL utility-ID links between FERC and EIA
- **Economy / population / climate (for the "as-of" features):** EIA-861 sales, BEA GDP, Census population, NOAA ONI (`Data/ext_raw/`).

### 2. Merge duplicate outage reports
- Reports with the same `OUTAGE.START.DATE` + `OUTAGE.START.TIME` + `U.S._STATE` describe the same event.
- The most complete one (customers and duration known) is kept: **4,279 → 4,195 rows**.
- `N.STATES.AFFECTED` missing (Purdue rows) is set to 1.

### 3. Clean the storm data (NOAA)
- State names to upper case, so they match the outage `CLEAN_STATE` (e.g. `TEXAS`).
- `BEGIN_DATE_TIME`, `END_DATE_TIME` parsed to datetimes; rows without valid times dropped.
- **Droughts removed:** they last months and are listed once per zone.
- `MAGNITUDE` converted to a number.
- `EVENT_TYPE` grouped into **families**: wind, winter, cold, tropical, tornado, flood, hail, heat, lightning, fire, other.
- Zone id = `CZ_TYPE` + `CZ_FIPS`.
- Storms grouped by state and sorted by begin time.

### 4. Clean the NWS warning data
- Kept only **warnings (W), watches (A) and advisories (Y)**; statements dropped: **1,352,918 products**, 52 states / territories.
- `phenomena` code grouped into families, e.g. `SV`, `HW` → wind; `TO` → tornado; `WS`, `BZ`, `IS`, `SQ` → winter; `HU`, `TR` → tropical; `FF`, `FL` → flood; `EH`, `HT` → heat; `WC`, `FZ` → cold; `FW` → fire; else other.
- **Only real-time times used:**
  - `issue`: when it was issued;
  - `init_expire`: the expiry announced **at issue**.
  - The final `expire` (extended later) is **not** used.
- Times converted from UTC to each state's main time zone (daylight saving included; up to 1 h off in states with two zones).
- Sorted by state and issue time.

### 5. Create the targets (what we predict)
- `OUTAGE_START` = start date + time; `OUTAGE_RESTORATION` built the same way.
- **customers** → `log_customers` = log(1 + customers affected per 100,000 state population).
- **duration** → `log_duration` = log(1 + outage minutes).
- 0 customers / 0 minutes = no real interruption, so set to missing for that target.
- Kept only rows with a start time and at least one target: **4,195 → 3,817 rows**.
- Rows sorted by start time.
- **is_severe** is created later (step 13), from training rows only.

### 6. Add time features
`start_month`, `start_hour`, `is_weekend`, cyclic month/hour (sin, cos), `post_regime` (year ≥ 2011), `regime_2024` (OE-417 form change in 2024).

### 7. Add outage-history features (only past information)
Earlier outages count **only if restored before this outage started** (unknown restoration = known 7 days after start). Small samples are smoothed toward a broader average.

| Group | Features |
|---|---|
| State history | past outages (all / last 365 days), mean customers / duration, zero-customer share |
| Global and recent history | all-time means, last-365-day means, 3-year state means, outage counts (30 days, 365 days) |
| Storm-status history | past outcomes for the same storm / no-storm status |
| Area history (from the extended file) | `area_hist_n`, `area_hist_n_365d`, `area_hist_log_cust`, `area_hist_log_dur`, `area_hist_attack_share`, plus `n_states_affected` |
| Cause history (scenario B only) | past outcomes for the same cause, globally and in the state |

### 8. Clean the outage columns
- Column names to lower case (`.` → `_`).
- Economic / population columns converted to numbers; missing values filled with the state's earlier value.
- log1p applied to skewed columns.
- Cause features (scenario B): `cause_reported`, `is_named_hurricane`.

### 9. "As-of" replacement (used by the strict and completed tables)

| Feature | Replaced with |
|---|---|
| EIA prices, sales, customers, sector shares | **previous year**, annual EIA-861 |
| BEA GDP, per-capita GDP, utility share | **previous year** |
| population | **previous year** (Census) |
| ONI anomaly, climate category | season **ending the month before** the outage |

### 10. Merge outages and NOAA storms (by state + time), one table per rule
Matched on **same state** (`clean_state` = NOAA `STATE`) and time. Look-back window = 24 h.

| Rule | Storm counted if | Outages with a storm |
|---|---|---|
| R0 proposal | storm **began** within the 24 h before the outage | 62.7% |
| R1 overlap (W2 fix) | began before the outage **and** still active in the last 24 h | 71.2% |
| R2 strict (W1, operational) | as R1, but began **≥ 1 h before** the outage | 70.4% |
| R3 completed | storm had **ended** before the outage | 57.7% |

- **Overlap vs proposal:** 1,378 more outages (36.1%) get storms, 20,345 extra storm records.
- **Storm features created:**
  - counts (24 h, 7 days), zones, event types, counts per family
  - still active (overall and per family), hours since the last storm
  - max wind and hail (**completed storms only**), most common type
- **Never used:** damage, injuries, deaths.
- R0 / R1 use the original economy columns; R2 / R3 use the as-of ones (step 9).

### 11. Merge outages and NWS warnings (by state + time)
Matched on **same state** (`postal_code` = NWS `state`) at the outage start time; added to every rule's table.

| Feature | Meaning |
|---|---|
| `nws_warn_active`, `nws_watch_active`, `nws_adv_active` | in effect at outage start (issued before it, announced expiry after it) |
| `nws_warn_active_<family>` | warnings in effect, by family (9 families) |
| `nws_warn_24h`, `nws_warn_7d` | warnings issued in the last 24 h / 7 days |
| `nws_hours_since_warning` | hours since the last warning (capped at 168) |

Outages **before 2006** get missing values, because the archive is complete only from Nov 2005.

### 12. Merge the new grid data (proposed model), previous year only
Each outage gets the values from the **year before** it (outage year − 1).
- **EIA-861, state level:** SAIDI, SAIFI, CAIDI (with / without major-event days), averaged over the state's utilities, weighted by customers.
- **EIA-861, county level:**
  - utilities serving the outage's county (from `area_key`, via the EIA service-territory list) → county SAIDI / SAIFI / CAIDI and smart-meter share (state value if the county is unknown);
  - state smart-meter share, distribution circuits per 1,000 customers, share of circuits with voltage optimisation.
- **FERC Form 1, state level:** linked to EIA utilities via PUDL ids, customer-weighted:
  - underground share of distribution lines
  - distribution plant per customer (log)
  - overhead-line maintenance per customer (log)
  - maintenance per $ of overhead plant
- **Join keys:** state (`postal_code`) + year − 1; county features also use the county from `area_key`.

### 13. Split train / test (by time, not random)
- **Train:** outages before 2022-01-01, **2,722 outages** (1,466 with customers known, 2,613 with duration known).
- **Test:** 2022–2024, **1,095 outages** (320 customers, 964 duration, 189 with both, used for is_severe).
- **is_severe label:** Q1 / Q3 computed **from training rows only**; severe = above Q3 in customers **or** duration.

### 14. Columns kept and dropped (leakage control)
Only a **whitelist** of feature columns goes into the model; everything else is dropped. The code also stops if a leakage column slips in.

**Dropped, because they're the answer or are known only after the outage:**
- `customers_affected`, `outage_duration`, `log_customers`, `log_duration`, `demand_loss_mw`, `cust_reported_zero`
- `outage_restoration`, `outage_restoration_date`, `outage_restoration_time`
- `cause_category`, `cause_category_detail`, `hurricane_names` (raw cause text; scenario B uses the cleaned `cause_reported` instead)
- `year`, `start_year`, `obs`
- From the extended file: `source`, `area_affected`, `area_key`, `event_type_raw`, `alert_criteria`, `same_start_reports`
- From NOAA: damage, injuries, deaths (never loaded), and magnitudes of ongoing storms.
- From NWS: the final `expire` time (only `issue` and `init_expire` are used).

**Kept (numeric):**

| Group | Count |
|---|---|
| state statistics | 38 |
| time | 9 |
| NOAA storm features | 25 |
| history | 17 |
| area history | 6 |
| NWS warnings | 15 |
| **total, scenario A** | **110** |
| + cause (scenario B) | +5 = 115 |
| + new grid (proposed model) | +16 = 126 (A) / 131 (B) |

**Kept (categorical):** `nerc_region`, `climate_region`, `climate_category`, `clean_state`, `noaa_dominant_event_type`, plus `cause_reported` in scenario B.

### 15. Prepare features for the models (fitted on training rows only)
- **Numeric:** missing values filled with the median plus a "was missing" flag, then standardised.
- **Categorical:** one-hot encoded; categories seen fewer than 10 times are grouped.

### 16. Train the models
- **Ensemble (main model)** = average of:
  - Random Forest (500 trees)
  - LightGBM (800 trees)
  - ExtraTrees (500 trees)
- Logistic / ridge regression as a simple comparison.
- **Recency weights:** an outage 4 years older counts half.
- Classifiers use balanced class weights. The is_severe threshold is tuned for accuracy on training data only (inner 4-fold out-of-fold predictions).

### 17. Experiments
- **Experiment 1, base (outage + NOAA + NWS):** R0, R1, R2, R3, plus references:
  - `REF_noaa_only`: strict NOAA, no NWS
  - `REF_nws_only`: NWS, no NOAA
  - `REF_no_weather`
- **Experiment 2, proposed (+ new grid):** R0, R1, R2, R3.
- Each run is done for **scenario A** (pre-event) and **scenario B** (cause known), for 3 targets.
- **Base vs proposed:** paired by storm rule (proposed − base).
- **Severe detection (strict rule):**
  - baselines (always / never severe, random, storm rule, cause rule) and logistic regression
  - ensemble uncalibrated / Platt-calibrated
  - cost-tuned thresholds (missed severe : false alarm = 1:1, 3:1, 5:1)

### 18. Evaluate on the test set
- **Regression (customers, duration; log scale):** RMSE, R², MAE, Spearman.
- **is_severe:** precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, accuracy, confusion matrix (graph), plus expected cost and a reliability (calibration) plot in the severe-detection section.

### 19. Outputs (`progress_stage2/results_outage_extended_noaa_nws_newgrid/`)
| File | Content |
|---|---|
| `exp1_base_regression.csv`, `exp1_base_is_severe.csv` | Experiment 1 metrics |
| `exp2_proposed_regression.csv`, `exp2_proposed_is_severe.csv` | Experiment 2 metrics |
| `comparison_regression.csv`, `comparison_is_severe.csv` | proposed − base |
| `severe_detection.csv` | baselines, calibration, cost thresholds |
| `*_confusion_*.png`, `reliability.png` | figures |
