# Process: `orginalProblemStatementwith_outage(old)_noaa_hfield.ipynb`

**Base model:** original outage data + NOAA storms. **Proposed model:** base + HIFLD transmission-line features.

---

### 1. Load the datasets
- **Outages:** `Data/outage.xlsx` (Purdue, Mukherjee et al., 2018), **1,534 major outages**, Jan 2000 – Jul 2016, 56 columns.
- **Storms:** NOAA Storm Events, 25 yearly CSV files (2000–2024), **1,552,468 storm records**. Only these columns are read: `STATE, BEGIN_DATE_TIME, END_DATE_TIME, EVENT_TYPE, MAGNITUDE, CZ_TYPE, CZ_FIPS`.
- **Grid (proposed model only):** `Data/hifld_transmission_state_features.csv`, one row per state (transmission lines clipped to state boundaries).
- **Economy / population / climate (for the "as-of" features):**
  - EIA-861 annual sales (`Data/pudl_raw/`)
  - BEA state GDP, Census population, NOAA ONI climate index (`Data/ext_raw/`)

### 2. Merge duplicate outage reports
- Reports with the same `OUTAGE.START.DATE` + `OUTAGE.START.TIME` + `U.S._STATE` describe the same event.
- The most complete one (customers and duration known) is kept: **1,534 → 1,503 rows**.

### 3. Clean the storm data
- State names to upper case, so they match the outage `CLEAN_STATE` (e.g. `TEXAS`).
- `BEGIN_DATE_TIME`, `END_DATE_TIME` parsed to datetimes; rows without valid times dropped.
- **Droughts removed:** they last months and are listed once per zone.
- `MAGNITUDE` converted to a number.
- `EVENT_TYPE` grouped into **families**: wind, winter, cold, tropical, tornado, flood, hail, heat, lightning, fire, other.
- Zone id = `CZ_TYPE` + `CZ_FIPS`.
- Storms grouped by state and sorted by begin time, for fast matching.

### 4. Create the targets (what we predict)
- `OUTAGE_START` = `OUTAGE.START.DATE` + `OUTAGE.START.TIME`; `OUTAGE_RESTORATION` built the same way.
- **customers** → `log_customers` = log(1 + customers affected per 100,000 state population).
- **duration** → `log_duration` = log(1 + outage minutes).
- 0 customers / 0 minutes = no real interruption, so set to missing for that target.
- Kept only rows with a start time and at least one target: **1,503 → 1,392 rows**.
- Rows sorted by start time, because every split is chronological.
- **is_severe** is created later (step 11), using training rows only.

### 5. Add time features
`start_month`, `start_hour`, `is_weekend`, cyclic month/hour (sin, cos), `post_regime` (year ≥ 2011, a reporting change), `regime_2024` (always 0 here).

### 6. Add outage-history features (only past information)
For each outage, earlier outages count **only if they were restored before this one started** (unknown restoration = known 7 days after its start). Small samples are smoothed toward a broader average.

| Group | Features |
|---|---|
| State history | number of past outages (all / last 365 days), mean customers, mean duration, share of zero-customer reports |
| Global history | mean customers / duration of all past outages |
| Recent history | last-365-day means, 3-year state means, outages in the last 30 days, outages nationally in the last 365 days |
| Storm-status history | past outcomes for outages with the same storm / no-storm status (globally, then in the state) |
| Cause history (scenario B only) | past outcomes for the same cause, globally and in the state |

### 7. Clean the outage columns
- Column names to lower case (`.` → `_`).
- Economic / population columns converted to numbers; missing values filled with the same state's earlier value.
- log1p applied to skewed columns (sales, customers, GDP, population, densities).
- Cause features (scenario B): `cause_reported` (lower case) and `is_named_hurricane`.

### 8. "As-of" replacement (used by the strict and completed tables)
The original file gives economic values **for the outage month or year**, which are published later. They were replaced with values that existed before the outage:

| Feature | Replaced with |
|---|---|
| EIA prices, sales, customers (res / com / ind / total), sector shares | **previous year**, annual EIA-861 |
| BEA real GDP, per-capita GDP, utility share | **previous year** |
| population | **previous year** (Census) |
| ONI anomaly, climate category | season **ending the month before** the outage |

### 9. Merge outages and storms (by state + time), one table per rule
Matched on **same state** (`clean_state` = NOAA `STATE`) and time. Look-back window = 24 h.

| Rule | Storm counted if | Outages with a storm |
|---|---|---|
| R0 proposal | storm **began** within the 24 h before the outage | 68.0% |
| R1 overlap (W2 fix) | began before the outage **and** still active in the last 24 h | 76.0% |
| R2 strict (W1, operational) | as R1, but began **≥ 1 h before** the outage | 74.6% |
| R3 completed | storm had **ended** before the outage | 59.6% |

- **Overlap vs proposal:** 509 more outages (36.6%) get storms, 5,812 extra storm records.
- **Storm features created:**
  - counts in the last 24 h and 7 days, number of zones and event types, counts per family
  - storm still active (overall and per family), hours since the last storm
  - max wind and hail (**only from storms already ended**), most common storm type
- **Never used:** damage, injuries, deaths.
- R0 / R1 use the original economy columns; R2 / R3 use the as-of ones (step 8).

### 10. Merge HIFLD (proposed model)
- Joined to every outage **by state** (`clean_state`).
- Features: `grid_transmission_km` (log), `grid_hv_km` (log), `grid_km_per_1000_km2`, `grid_hv_share`, `grid_km_per_100k_pop`.
- Same value in every year (one 2022 snapshot).

### 11. Split train / test (by time, not random)
- **Train:** outages before 2014-01-01, **1,129 outages** (749 with customers known, 1,117 with duration known).
- **Test:** 2014-01-01 to Jul 2016, **263 outages** (108 customers, 251 duration, 96 with both, used for is_severe).
- **is_severe label:** Q1 / Q3 computed **from training rows only**; severe = above Q3 in customers **or** duration.

### 12. Columns kept and dropped (leakage control)
Only a **whitelist** of feature columns goes into the model; everything else is dropped. The code also stops if a leakage column slips in.

**Dropped, because they're the answer or are known only after the outage:**
- `customers_affected`, `outage_duration`, `log_customers`, `log_duration`, `demand_loss_mw`, `cust_reported_zero`
- `outage_restoration`, `outage_restoration_date`, `outage_restoration_time`
- `cause_category`, `cause_category_detail`, `hurricane_names` (raw cause text; scenario B uses the cleaned `cause_reported` instead)
- `year`, `start_year`, `obs`
- From NOAA: damage, injuries and deaths (never loaded), and magnitudes of storms still ongoing.

**Kept (numeric):**

| Group | Count |
|---|---|
| state statistics: economy, population, land use, climate anomaly | 38 |
| time | 9 |
| NOAA storm features | 25 |
| history | 17 |
| **total, scenario A** | **89** |
| + cause (scenario B) | +5 = 94 |
| + HIFLD (proposed model) | +5 = 94 (A) / 99 (B) |

**Kept (categorical):** `nerc_region`, `climate_region`, `climate_category`, `clean_state`, `noaa_dominant_event_type`, plus `cause_reported` in scenario B.

### 13. Prepare features for the models (fitted on training rows only)
- **Numeric:** missing values filled with the median plus a "was missing" flag, then standardised.
- **Categorical:** one-hot encoded; categories seen fewer than 10 times are grouped.

### 14. Train the models
- **Ensemble (main model)** = average of:
  - Random Forest (500 trees)
  - LightGBM (800 trees)
  - ExtraTrees (500 trees)
- Logistic / ridge regression as a simple comparison.
- **Recency weights:** an outage 4 years older counts half.
- Classifiers use balanced class weights. The is_severe threshold is tuned for accuracy on training data only (inner 4-fold out-of-fold predictions).

### 15. Experiments
- **Experiment 1, base (outage + NOAA):** R0, R1, R2, R3, and a no-weather reference.
- **Experiment 2, proposed (+ HIFLD):** R0, R1, R2, R3.
- Each run is done for **scenario A** (pre-event) and **scenario B** (cause known), for 3 targets.
- **Base vs proposed:** paired by storm rule (proposed − base).
- **Severe detection (strict rule):**
  - baselines (always / never severe, random, storm rule, cause rule) and logistic regression
  - ensemble uncalibrated / Platt-calibrated
  - cost-tuned thresholds (missed severe : false alarm = 1:1, 3:1, 5:1)

### 16. Evaluate on the test set
- **Regression (customers, duration; log scale):** RMSE, R², MAE, Spearman.
- **is_severe:** precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy, accuracy, confusion matrix (graph), plus expected cost and a reliability (calibration) plot in the severe-detection section.

### 17. Outputs (`progress_stage2/results_outage_old_noaa_hifld/`)
| File | Content |
|---|---|
| `exp1_base_regression.csv`, `exp1_base_is_severe.csv` | Experiment 1 metrics |
| `exp2_proposed_regression.csv`, `exp2_proposed_is_severe.csv` | Experiment 2 metrics |
| `comparison_regression.csv`, `comparison_is_severe.csv` | proposed − base |
| `severe_detection.csv` | baselines, calibration, cost thresholds |
| `*_confusion_*.png`, `reliability.png` | figures |
