"""
Experiment 3: AMSA on real routes (TomTom live traffic + OpenTopoData SRTM 30 m)
===========================================================================
Starts the real Express backend (backend/server.js) and queries POST /api/feasibility
once per route and direction. Route physics (distance, cumulative climb/descent,
traffic-jam ratio) therefore come from the production code path.

DTE is linear in SOC in the backend model, so the AMSA SOC thresholds follow
analytically from DTE at 100 % SOC:
    SAFE        if SOC >= 100 * d / DTE_SPORT(100)
    CRITICAL    if 100 * d / DTE_ECO(100) <= SOC < 100 * d / DTE_SPORT(100)   (ECO-Lock)
    IMPOSSIBLE  if SOC < 100 * d / DTE_ECO(100)                               (charge required)
A sweep over SOC is also evaluated with a re-implementation of calculateDTE that is
verified against the endpoint output. A "naive" estimator (flat road, no traffic)
is compared to show the decisions that a terrain/traffic-unaware DTE would get wrong.

Run:  python experiments/exp3_amsa_routes/run_exp3.py
"""

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PORT = 5055
BASE = f"http://127.0.0.1:{PORT}"

# Must match backend/server.js
PACK_AH, PACK_V = 30.9, 63.5
C0 = {"ECO": 25.8, "SPORT": 31.6}
AVG_SPEED = 60.0   # backend default (calibration point of the 76/62 km display)
BASE_SOH = 100.0

PLACES = {
    "New Delhi": (28.6139, 77.2090), "Noida": (28.5355, 77.3910),
    "Dehradun": (30.3165, 78.0322), "Mussoorie": (30.4598, 78.0664),
    "Shimla": (31.1048, 77.1734), "Kufri": (31.1011, 77.2666),
    "Siliguri": (26.7271, 88.3953), "Darjeeling": (27.0360, 88.2627),
}
ROUTES = [("New Delhi", "Noida", "Flat urban"),
          ("Dehradun", "Mussoorie", "Hill"),
          ("Shimla", "Kufri", "Hill"),
          ("Siliguri", "Darjeeling", "Hill")]



def road_load(v_kmh, m=155, g=9.81, crr=0.012, rho=1.2, cda=0.5, p_aux=40):
    """Road load per metre (N): rolling + aero + auxiliary. Same as roadLoadPerKm() in backend/server.js."""
    v = max(5.0, v_kmh) / 3.6
    return m * g * crr + 0.5 * rho * cda * v * v + p_aux / v

def dte(soc, mode, ug=0.0, dg=0.0, jam=0.0, soh=BASE_SOH, speed=AVG_SPEED, accel=0.0, dT=0.0):
    """Exact port of calculateDTE() in backend/server.js."""
    e_av = soc / 100 * PACK_AH * PACK_V * soh / 100
    v_eff = speed * (1 - jam * 0.5)
    aero = road_load(v_eff) / road_load(60.0)
    sg = 1 + jam * (0.3 + accel / 100 * 0.4)
    ba = 1 + accel / 100 * 0.15
    up = 1 + ug * 0.15
    regen = 0.6 + jam * 0.2
    dn = max(0.2, 1 - dg * 0.08 * regen)
    tp = 1 + max(0, dT * 0.005)
    return e_av / (C0[mode] * aero * sg * ba * up * dn * tp)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wait_for_server(proc, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("backend exited early")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("backend did not start")


def classify(soc, d, e100, s100):
    if soc >= 100 * d / s100:
        return "SAFE"
    if soc >= 100 * d / e100:
        return "CRITICAL"
    return "IMPOSSIBLE"


def main():
    env = dict(os.environ, PORT=str(PORT))
    log = open(OUT / "backend_log.txt", "w", encoding="utf-8")
    proc = subprocess.Popen(["node", "server.js"], cwd=ROOT / "backend", env=env, stdout=log, stderr=log)
    rows, sweep = [], []
    try:
        wait_for_server(proc)
        for a, b, kind in ROUTES:
            for o, d_ in [(a, b), (b, a)]:
                body = {"current_soc": 100, "origin": "%f,%f" % PLACES[o],
                        "destination": "%f,%f" % PLACES[d_], "baseSoh": BASE_SOH, "avgSpeedKmh": AVG_SPEED}
                stamp = datetime.now().isoformat(timespec="seconds")
                r = post("/api/feasibility", body)
                pm, f = r["physics_metrics"], r["feasibility"]
                dist = float(r["route_distance_km"])
                ug, dg, jam = pm["uphill_grade_pct"], pm["downhill_grade_pct"], pm["traffic_jam_ratio"]
                e100, s100 = dte(100, "ECO", ug, dg, jam), dte(100, "SPORT", ug, dg, jam)
                n_e100, n_s100 = dte(100, "ECO"), dte(100, "SPORT")   # naive: flat, no traffic
                check = abs(e100 - f["eco_dte"]) + abs(s100 - f["sport_dte"])
                rows.append({
                    "Route": f"{o} → {d_}", "Type": kind, "Queried_at": stamp,
                    "Distance_km": round(dist, 2),
                    "Climb_m": pm["uphill_climb_meters"], "Descent_m": pm["downhill_descent_meters"],
                    "Uphill_grade_pct": ug, "Downhill_grade_pct": dg, "Traffic_jam_ratio": jam,
                    "ECO_DTE_100_km": round(e100, 1), "SPORT_DTE_100_km": round(s100, 1),
                    "Naive_ECO_DTE_100_km": round(n_e100, 1),
                    "Energy_ECO_Wh": round(dist / e100 * PACK_AH * PACK_V, 0),
                    "Energy_naive_ECO_Wh": round(dist / n_e100 * PACK_AH * PACK_V, 0),
                    "SOC_min_SAFE_pct": round(100 * dist / s100, 1),
                    "SOC_min_ECO_pct": round(100 * dist / e100, 1),
                    "Naive_SOC_min_SAFE_pct": round(100 * dist / n_s100, 1),
                    "Naive_SOC_min_ECO_pct": round(100 * dist / n_e100, 1),
                    "Endpoint_status_at_100": f["status"], "Verify_abs_err_km": round(check, 6),
                })
                for soc in range(5, 101):
                    true_c = classify(soc, dist, e100, s100)
                    naive_c = classify(soc, dist, n_e100, n_s100)
                    sweep.append({"Route": f"{o} → {d_}", "SOC": soc,
                                  "ECO_DTE": dte(soc, "ECO", ug, dg, jam), "SPORT_DTE": dte(soc, "SPORT", ug, dg, jam),
                                  "Naive_ECO_DTE": dte(soc, "ECO"), "Naive_SPORT_DTE": dte(soc, "SPORT"),
                                  "Distance": dist, "AMSA": true_c, "Naive": naive_c})
                print(rows[-1])
                time.sleep(1.5)  # OpenTopoData public API: max 1 request / s

            # spot-check the endpoint at a mid SOC against the analytic classification (forward direction)
        mid = []
        for a, b, _ in ROUTES:
            row = next(x for x in rows if x["Route"] == f"{a} → {b}")
            soc = round((row["SOC_min_SAFE_pct"] + row["SOC_min_ECO_pct"]) / 2, 1)
            if 0 < soc <= 100:
                r = post("/api/feasibility", {"current_soc": soc, "origin": "%f,%f" % PLACES[a],
                                               "destination": "%f,%f" % PLACES[b], "baseSoh": BASE_SOH,
                                               "avgSpeedKmh": AVG_SPEED})
                mid.append({"Route": row["Route"], "SOC": soc, "Endpoint_status": r["feasibility"]["status"],
                            "Endpoint_action": r["amsa_decision"]["action"]})
                time.sleep(1.5)
    finally:
        proc.terminate()
        log.close()

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "routes_summary.csv", index=False)
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "soc_sweep.csv", index=False)
    mis = (sw.assign(Wrong=sw.AMSA != sw.Naive, Dangerous=(sw.Naive == "SAFE") & (sw.AMSA != "SAFE"),
                     Strand=(sw.Naive != "IMPOSSIBLE") & (sw.AMSA == "IMPOSSIBLE"))
           .groupby("Route", sort=False)[["Wrong", "Dangerous", "Strand"]].sum()
           .rename(columns={"Wrong": "SOC_points_misclassified_by_naive",
                            "Dangerous": "SOC_points_naive_SAFE_but_AMSA_not",
                            "Strand": "SOC_points_naive_reachable_but_actually_impossible"}))
    mis.to_csv(OUT / "naive_vs_amsa.csv")
    (OUT / "midpoint_endpoint_check.json").write_text(json.dumps(mid, indent=2))

    # ---------------- figure: forward routes
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    colors = {"SAFE": "#dcfce7", "CRITICAL": "#fef3c7", "IMPOSSIBLE": "#fee2e2"}
    for ax, (a, b, _) in zip(axes.ravel(), ROUTES):
        name = f"{a} → {b}"
        s = sw[sw.Route == name]
        row = tab[tab.Route == name].iloc[0]
        for lo, hi, c in [(5, row.SOC_min_ECO_pct, "IMPOSSIBLE"), (row.SOC_min_ECO_pct, row.SOC_min_SAFE_pct, "CRITICAL"),
                          (row.SOC_min_SAFE_pct, 100, "SAFE")]:
            lo, hi = max(5, lo), min(100, hi)
            if hi > lo:
                ax.axvspan(lo, hi, color=colors[c], lw=0)
        ax.plot(s.SOC, s.ECO_DTE, color="#16a34a", lw=1.6, label="DTE ECO (terrain+traffic)")
        ax.plot(s.SOC, s.SPORT_DTE, color="#dc2626", lw=1.6, label="DTE SPORT (terrain+traffic)")
        ax.plot(s.SOC, s.Naive_ECO_DTE, color="#16a34a", lw=1, ls=":", label="DTE ECO (naive)")
        ax.plot(s.SOC, s.Naive_SPORT_DTE, color="#dc2626", lw=1, ls=":", label="DTE SPORT (naive)")
        ax.axhline(row.Distance_km, color="k", lw=1.2, ls="--", label="Route distance")
        ax.set_title(f"{name}  ({row.Distance_km:.1f} km, +{row.Climb_m:.0f} m, J={row.Traffic_jam_ratio:.2f})", fontsize=8.5)
        ax.set_xlim(5, 100)
        ax.set_ylim(0, max(s.Naive_ECO_DTE.max(), row.Distance_km) * 1.05)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("State of charge (%)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Distance (km)", fontsize=8)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_amsa_routes.png", dpi=300)
    plt.close(fig)

    md = ["# Experiment 3: AMSA on real routes\n",
          f"Model: pack {PACK_AH} Ah × {PACK_V} V, c0 ECO {C0['ECO']} / SPORT {C0['SPORT']} Wh/km, "
          f"avg speed {AVG_SPEED} km/h, SoH {BASE_SOH} %. Routing: TomTom (live traffic); elevation: OpenTopoData SRTM 30 m (srtm30m).\n",
          "## Route physics and AMSA thresholds\n", tab.to_csv(index=False),
          "\n## Naive (flat, no-traffic) DTE vs AMSA: number of 1 % SOC steps (5–100 %) classified differently\n",
          mis.to_csv(), "\n## Endpoint spot-check at mid-CRITICAL SOC\n", json.dumps(mid, indent=2)]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(tab.to_string())
    print(mis.to_string())
    print(json.dumps(mid, indent=2))


if __name__ == "__main__":
    main()
