# Experiment 3: AMSA on real routes

Model: pack 30.9 Ah × 63.5 V, c0 ECO 25.8 / SPORT 31.6 Wh/km, avg speed 60.0 km/h, SoH 100.0 %. Routing: TomTom (live traffic); elevation: OpenTopoData SRTM 30 m (srtm30m).

## Route physics and AMSA thresholds

Route,Type,Queried_at,Distance_km,Climb_m,Descent_m,Uphill_grade_pct,Downhill_grade_pct,Traffic_jam_ratio,ECO_DTE_100_km,SPORT_DTE_100_km,Naive_ECO_DTE_100_km,Energy_ECO_Wh,Energy_naive_ECO_Wh,SOC_min_SAFE_pct,SOC_min_ECO_pct,Naive_SOC_min_SAFE_pct,Naive_SOC_min_ECO_pct,Endpoint_status_at_100,Verify_abs_err_km
New Delhi → Noida,Flat urban,2026-10-21T08:54:11,25.86,116,130,0.45,0.5,0.0,73.0,59.6,76.1,695.0,667.0,43.4,35.4,41.6,34.0,SAFE,0.04432
Noida → New Delhi,Flat urban,2026-10-21T08:54:14,25.73,145,133,0.56,0.52,0.043,73.5,60.0,76.1,687.0,664.0,42.9,35.0,41.4,33.8,SAFE,0.103057
Dehradun → Mussoorie,Hill,2026-10-21T08:54:17,32.98,1427,104,4.33,0.32,0.0,46.8,38.2,76.1,1382.0,851.0,86.3,70.4,53.1,43.4,SAFE,0.002018
Mussoorie → Dehradun,Hill,2026-10-21T08:54:20,28.53,131,1478,0.46,5.18,0.0,94.7,77.3,76.1,591.0,736.0,36.9,30.1,45.9,37.5,SAFE,0.027366
Shimla → Kufri,Hill,2026-10-21T08:54:22,13.38,695,352,5.19,2.63,0.0,48.9,40.0,76.1,536.0,345.0,33.5,27.3,21.5,17.6,SAFE,0.022664
Kufri → Shimla,Hill,2026-10-21T08:54:24,13.38,351,688,2.62,5.14,0.0,72.5,59.2,76.1,362.0,345.0,22.6,18.5,21.5,17.6,SAFE,0.031563
Siliguri → Darjeeling,Hill,2026-10-21T08:54:26,60.43,2278,282,3.77,0.47,0.0,49.7,40.6,76.1,2386.0,1559.0,148.9,121.6,97.3,79.5,IMPOSSIBLE,0.013826
Darjeeling → Siliguri,Hill,2026-10-21T08:54:28,59.3,276,2274,0.47,3.83,0.0,87.0,71.1,76.1,1337.0,1530.0,83.4,68.1,95.5,78.0,SAFE,0.143281


## Naive (flat, no-traffic) DTE vs AMSA: number of 1 % SOC steps (5–100 %) classified differently

Route,SOC_points_misclassified_by_naive,SOC_points_naive_SAFE_but_AMSA_not,SOC_points_naive_reachable_but_actually_impossible
New Delhi → Noida,3,2,1
Noida → New Delhi,2,1,1
Dehradun → Mussoorie,43,33,27
Mussoorie → Dehradun,15,0,0
Shimla → Kufri,16,12,10
Kufri → Shimla,2,1,1
Siliguri → Darjeeling,21,3,21
Darjeeling → Siliguri,21,0,0


## Endpoint spot-check at mid-CRITICAL SOC

[
  {
    "Route": "New Delhi \u2192 Noida",
    "SOC": 39.4,
    "Endpoint_status": "CRITICAL",
    "Endpoint_action": "FORCE_ECO_MODE"
  },
  {
    "Route": "Dehradun \u2192 Mussoorie",
    "SOC": 78.3,
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