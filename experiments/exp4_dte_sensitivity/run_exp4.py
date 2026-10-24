"""
Experiment 4: one-at-a-time sensitivity of the backend DTE model (tornado chart)
================================================================================
Baseline: SOC 80 %, ECO, SoH 100 %, 60 km/h, flat road, no traffic, no aggression, ambient delta 0.
Each factor is moved over a realistic range while the others stay at baseline.
The DTE function is an exact port of calculateDTE() in backend/server.js (canonical parameters).

Run:  python experiments/exp4_dte_sensitivity/run_exp4.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PACK_AH, PACK_V = 30.9, 63.5
C0 = {"ECO": 25.8, "SPORT": 31.6}
SOC = 80.0



def road_load(v_kmh, m=155, g=9.81, crr=0.012, rho=1.2, cda=0.5, p_aux=40):
    """Road load per metre (N): rolling + aero + auxiliary. Same as roadLoadPerKm() in backend/server.js."""
    v = max(5.0, v_kmh) / 3.6
    return m * g * crr + 0.5 * rho * cda * v * v + p_aux / v

def dte(mode="ECO", soc=SOC, ug=0.0, dg=0.0, jam=0.0, soh=100.0, speed=60.0, accel=0.0, dT=0.0, c0_scale=1.0):
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
    return e_av / (C0[mode] * c0_scale * aero * sg * ba * up * dn * tp)


# (label, kwarg, low value, high value, unit)
FACTORS = [
    ("Uphill grade", "ug", 0.0, 5.0, "%"),
    ("Downhill grade", "dg", 0.0, 5.0, "%"),
    ("Average speed", "speed", 40.0, 90.0, "km/h"),
    ("Traffic-jam ratio J", "jam", 0.0, 0.8, ""),
    ("Acceleration aggression", "accel", 0.0, 50.0, "%"),
    ("Ambient temperature rise", "dT", 0.0, 15.0, "°C"),
    ("State of health", "soh", 80.0, 100.0, "%"),
    ("Base consumption c0 (±10 %)", "c0_scale", 0.9, 1.1, "×"),
]


def main():
    rows = []
    for mode in ["ECO", "SPORT"]:
        base = dte(mode)
        for label, key, lo, hi, unit in FACTORS:
            d_lo, d_hi = dte(mode, **{key: lo}), dte(mode, **{key: hi})
            rows.append({"Mode": mode, "Factor": label, "Low": lo, "High": hi, "Unit": unit,
                         "Baseline_km": round(base, 2), "DTE_at_low_km": round(d_lo, 2),
                         "DTE_at_high_km": round(d_hi, 2),
                         "Change_low_pct": round(100 * (d_lo / base - 1), 1),
                         "Change_high_pct": round(100 * (d_hi / base - 1), 1),
                         "Swing_km": round(abs(d_hi - d_lo), 2)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "sensitivity_table.csv", index=False)

    # curves for each factor (ECO)
    curves = []
    for label, key, lo, hi, unit in FACTORS:
        for x in np.linspace(lo, hi, 41):
            curves.append({"Factor": label, "Value": round(float(x), 4), "DTE_ECO_km": dte("ECO", **{key: x}),
                           "DTE_SPORT_km": dte("SPORT", **{key: x})})
    pd.DataFrame(curves).to_csv(OUT / "sensitivity_curves.csv", index=False)

    # tornado (ECO)
    eco = tab[tab.Mode == "ECO"].sort_values("Swing_km")
    base = eco.Baseline_km.iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(eco))
    for i, r in enumerate(eco.itertuples()):
        for val, x, col in [(r.Low, r.DTE_at_low_km, "#2563eb"), (r.High, r.DTE_at_high_km, "#f97316")]:
            ax.barh(i, x - base, left=base, color=col, height=0.6)
            fmt = f"{val:g}{r.Unit}" if r.Unit not in ("", "×") else f"{val:g}{r.Unit}"
            ax.text(x + (0.8 if x >= base else -0.8), i, fmt, va="center",
                    ha="left" if x >= base else "right", fontsize=6.5)
    ax.axvline(base, color="k", lw=1)
    ax.set_yticks(y, eco.Factor, fontsize=7.5)
    ax.set_xlabel(f"ECO distance-to-empty at SOC {SOC:.0f} % (km)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    lo_all = min(eco.DTE_at_low_km.min(), eco.DTE_at_high_km.min())
    hi_all = max(eco.DTE_at_low_km.max(), eco.DTE_at_high_km.max())
    ax.set_xlim(lo_all - 12, hi_all + 14)
    ax.set_ylim(-0.6, len(eco) + 0.1)
    ax.text(base, len(eco) - 0.45, f" baseline {base:.1f} km", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2563eb", label="Low value"), Patch(color="#f97316", label="High value")],
              fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dte_tornado.png", dpi=300)
    plt.close(fig)

    md = [f"# Experiment 4: DTE sensitivity (one-at-a-time)\n",
          f"Baseline: SOC {SOC:.0f} %, SoH 100 %, 60 km/h, flat, no traffic. Pack {PACK_AH} Ah × {PACK_V} V, "
          f"c0 ECO {C0['ECO']} / SPORT {C0['SPORT']} Wh/km.\n",
          tab.to_csv(index=False)]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(tab.to_string())


if __name__ == "__main__":
    main()
