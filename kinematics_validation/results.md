# Kinematic Synthesis Validation (Option B)

Dataset: `trinity_ev_dataset.csv`, 1668 rows, 6 sessions at 1 Hz.

## 1. Reproduction of stored columns

- Max |re-computed speed − stored speed| = 7.11e-15 km/h (reproduced exactly).

- Stored braking events: 396; re-computed (original): 396.

## 2. Plausibility checks

Limits used: acceleration ≤ 4.0 m/s², deceleration ≥ -7.0 m/s², speed ≤ v_max = 55.0 km/h.

| Variant | Max_speed_kmh | Mean_speed_kmh | Max_accel_ms2 | Min_accel_ms2 | Rows_accel_gt_limit | Rows_decel_lt_limit | P99_abs_accel | Braking_events |
|---|---|---|---|---|---|---|---|---|
| Original (upgrade_trinity.py) | 53.63 | 22.79 | 11.361 | -1.371 | 5 | 0 | 0.647 | 396 |
| Corrected (per-session reset) | 53.63 | 22.29 | 0.797 | -1.112 | 0 | 0 | 0.459 | 354 |



**Outlier diagnosis:** the max acceleration (11.36 m/s²) occurs at row 1116 (Time_s = 1, session 5), the first row after a session boundary. In `upgrade_trinity.py` the speed is zeroed at the boundary *after* the EMA loop, so the next row still integrates the previous session's speed and the jump from 0 creates the spike. Resetting the EMA state inside the loop (corrected variant) removes it.

## 3. Sensitivity to α and v_max (corrected variant)

| alpha | v_max | Max_speed | Mean_speed | P99_abs_accel | Max_abs_accel | Braking_events | Total_distance_km | Wh_per_km_power |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 45.0 | 36.7 | 17.1 | 0.21 | 0.408 | 247 | 7.93 | 4.4 |
| 0.05 | 55.0 | 44.8 | 20.9 | 0.257 | 0.499 | 291 | 9.69 | 3.6 |
| 0.05 | 65.0 | 53.0 | 24.7 | 0.304 | 0.59 | 320 | 11.45 | 3.05 |
| 0.05 | 75.0 | 61.1 | 28.5 | 0.351 | 0.68 | 351 | 13.21 | 2.64 |
| 0.1 | 45.0 | 42.0 | 17.93 | 0.311 | 0.703 | 323 | 8.31 | 4.2 |
| 0.1 | 55.0 | 51.3 | 21.92 | 0.38 | 0.86 | 346 | 10.16 | 3.43 |
| 0.1 | 65.0 | 60.6 | 25.9 | 0.449 | 1.016 | 362 | 12.0 | 2.91 |
| 0.1 | 75.0 | 69.9 | 29.89 | 0.518 | 1.172 | 378 | 13.85 | 2.52 |
| 0.15 | 45.0 | 43.9 | 18.24 | 0.375 | 0.91 | 334 | 8.45 | 4.13 |
| 0.15 | 55.0 | 53.6 | 22.29 | 0.459 | 1.112 | 354 | 10.33 | 3.38 |
| 0.15 | 65.0 | 63.4 | 26.34 | 0.542 | 1.315 | 361 | 12.2 | 2.86 |
| 0.15 | 75.0 | 73.1 | 30.39 | 0.625 | 1.517 | 369 | 14.08 | 2.48 |
| 0.2 | 45.0 | 44.6 | 18.39 | 0.445 | 1.084 | 327 | 8.52 | 4.09 |
| 0.2 | 55.0 | 54.5 | 22.48 | 0.544 | 1.325 | 347 | 10.42 | 3.35 |
| 0.2 | 65.0 | 64.4 | 26.57 | 0.643 | 1.566 | 361 | 12.31 | 2.83 |
| 0.2 | 75.0 | 74.3 | 30.65 | 0.742 | 1.807 | 369 | 14.2 | 2.46 |
| 0.3 | 45.0 | 44.9 | 18.55 | 0.633 | 1.394 | 332 | 8.6 | 4.06 |
| 0.3 | 55.0 | 54.9 | 22.68 | 0.773 | 1.704 | 346 | 10.51 | 3.32 |
| 0.3 | 65.0 | 64.9 | 26.8 | 0.914 | 2.014 | 353 | 12.42 | 2.81 |
| 0.3 | 75.0 | 74.9 | 30.92 | 1.055 | 2.324 | 358 | 14.33 | 2.43 |



At the default (α = 0.15, v_max = 55.0): total distance 10.33 km, 354 braking events. Distance scales linearly with v_max; α mainly changes acceleration peaks and braking counts, not distance.

## 4. Energy-consistency check

Pack energy assumed for SOC-based energy: 3810 Wh (backend model).

| Session | Config | Mode | Duration_s | Mean_speed_kmh | Distance_km | Mean_power_W | Energy_from_power_Wh | SOC_drop_pct | Energy_from_SOC_Wh | Wh_per_km_power | Wh_per_km_SOC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Single | ECO | 280 | 22.43 | 1.745 | 67.9 | 5.28 | 0.4 | 15.24 | 3.03 | 8.74 |
| 2 | Single | SPORT | 286 | 30.67 | 2.437 | 129.4 | 10.28 | 0.4 | 15.24 | 4.22 | 6.25 |
| 3 | Series | ECO | 281 | 18.87 | 1.473 | 51.1 | 3.99 | 0.3 | 11.43 | 2.71 | 7.76 |
| 4 | Series | SPORT | 268 | 21.9 | 1.63 | 82.3 | 6.13 | 0.3 | 11.43 | 3.76 | 7.01 |
| 5 | Parallel | ECO | 267 | 17.92 | 1.329 | 53.0 | 3.93 | 0.2 | 7.62 | 2.96 | 5.73 |
| 6 | Parallel | SPORT | 286 | 21.55 | 1.712 | 66.2 | 5.26 | 0.3 | 11.43 | 3.07 | 6.68 |



- Aggregate implied consumption: **3.38 Wh/km** (from measured power) and **7.01 Wh/km** (from SOC drop × 3810 Wh).

- DTE model assumes 150 Wh/km (ECO) / 250 Wh/km (SPORT): 44× / 74× the power-based value.

- Power-based vs SOC-based energy disagree by 2.1×, meaning the logged current/power and the logged SOC drop are themselves inconsistent for a 3.81 kWh pack (the SOC readout has 0.1 % resolution; ~0.4 % drops are near quantization).


## Figures

- `fig_speed_trace.png`: current vs synthesized speed with braking events (one session).

- `fig_sensitivity.png`: total distance and braking events across α × v_max.

- `fig_accel_hist.png`: acceleration distribution, original vs corrected.
