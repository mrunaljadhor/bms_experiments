# Experiment 3: AMSA on real routes

Model: pack 30.9 Ah × 63.5 V, c0 ECO 25.8 / SPORT 31.6 Wh/km, avg speed 60.0 km/h, SoH 100.0 %. Routing: TomTom (live traffic); elevation: OpenTopoData SRTM 30 m (srtm30m).

## Route physics and AMSA thresholds

Route,Type,Queried_at,Distance_km,Climb_m,Descent_m,Uphill_grade_pct,Downhill_grade_pct,Traffic_jam_ratio,ECO_DTE_100_km,SPORT_DTE_100_km,Naive_ECO_DTE_100_km,Energy_ECO_Wh,Energy_naive_ECO_Wh,SOC_min_SAFE_pct,SOC_min_ECO_pct,Naive_SOC_min_SAFE_pct,Naive_SOC_min_ECO_pct,Endpoint_status_at_100,Verify_abs_err_km
New Delhi → Noida,Flat urban,2026-10-16T11:49:16,28.38,120,135,0.42,0.48,0.0,73.2,59.8,76.1,760.0,732.0,47.5,38.8,45.7,37.3,SAFE,0.080647
Noida → New Delhi,Flat urban,2026-10-16T11:49:19,27.09,132,118,0.49,0.44,0.055,74.4,60.8,76.1,714.0,699.0,44.6,36.4,43.6,35.6,SAFE,0.047626
Dehradun → Mussoorie,Hill,2026-10-16T11:49:21,33.25,1375,42,4.13,0.13,0.014,47.6,38.8,76.1,1371.0,858.0,85.6,69.9,53.5,43.7,SAFE,0.075247
Mussoorie → Dehradun,Hill,2026-10-16T11:49:23,28.33,130,1472,0.46,5.2,0.004,95.0,77.6,76.1,585.0,731.0,36.5,29.8,45.6,37.3,SAFE,0.060339
Shimla → Kufri,Hill,2026-10-16T11:49:25,13.38,695,352,5.19,2.63,0.0,48.9,40.0,76.1,536.0,345.0,33.5,27.3,21.5,17.6,SAFE,0.022664
Kufri → Shimla,Hill,2026-10-16T11:49:27,13.38,351,688,2.62,5.14,0.0,72.5,59.2,76.1,362.0,345.0,22.6,18.5,21.5,17.6,SAFE,0.031563
Siliguri → Darjeeling,Hill,2026-10-16T11:49:29,60.37,2292,296,3.8,0.49,0.0,49.6,40.5,76.1,2388.0,1558.0,149.1,121.7,97.2,79.4,IMPOSSIBLE,0.029242
Darjeeling → Siliguri,Hill,2026-10-16T11:49:31,59.22,279,2277,0.47,3.85,0.0,87.1,71.2,76.1,1333.0,1528.0,83.2,68.0,95.4,77.9,SAFE,0.069954


## Naive (flat, no-traffic) DTE vs AMSA: number of 1 % SOC steps (5–100 %) classified differently

Route,SOC_points_misclassified_by_naive,SOC_points_naive_SAFE_but_AMSA_not,SOC_points_naive_reachable_but_actually_impossible
New Delhi → Noida,3,2,1
Noida → New Delhi,2,1,1
Dehradun → Mussoorie,42,32,26
Mussoorie → Dehradun,16,0,0
Shimla → Kufri,16,12,10
Kufri → Shimla,2,1,1
Siliguri → Darjeeling,21,3,21
Darjeeling → Siliguri,22,0,0


## Endpoint spot-check at mid-CRITICAL SOC

[
  {
    "Route": "New Delhi \u2192 Noida",
    "SOC": 43.1,
    "Endpoint_status": "CRITICAL",
    "Endpoint_action": "FORCE_ECO_MODE"
  },
  {
    "Route": "Dehradun \u2192 Mussoorie",
    "SOC": 77.8,
    "Endpoint_status": "CRITICAL",
    "Endpoint_action": "FORCE_ECO_MODE"
  },
  {
    "Route": "Shimla \u2192 Kufri",
    "SOC": 30.4,
    "Endpoint_status": "CRITICAL",
    "Endpoint_action": "FORCE_ECO_MODE"
  }
]