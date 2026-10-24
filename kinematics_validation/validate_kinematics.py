"""
Zero-Cost Kinematic Synthesis: validation without a vehicle (Option B)
=====================================================================
1. Reproduces the kinematic synthesis (upgrade_trinity.py) from raw Current_A.
2. Plausibility checks (speed / acceleration limits, outlier diagnosis).
3. Sensitivity analysis over alpha (EMA inertia) and v_max.
4. Energy-consistency check: implied Wh/km vs the DTE model's 150/250 Wh/km.

Run:  python kinematics_validation/validate_kinematics.py
Outputs go to kinematics_validation/ (CSV tables + PNG figures + results.md).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "trinity_ev_dataset.csv"

ALPHA = 0.15
V_MAX = 55.0
BRAKE_ACCEL = -0.05
BRAKE_THROTTLE = 0.2
PACK_WH = 60 * 63.5          # backend pack (3810 Wh)
ACCEL_LIMIT = 4.0            # m/s^2, generous upper bound for a light two-wheeler
DECEL_LIMIT = -7.0           # m/s^2, emergency braking bound


def load_sessions(df):
    """Split into riding sessions wherever Time_s goes backwards."""
    session = (df["Time_s"].diff() < 0).cumsum()
    return df.assign(Session=session)


def throttle_proxy(current):
    load = np.where(current < 0, -current, 0.0)
    i_max, i_min = np.quantile(load, 0.99), load.min()
    if i_max == i_min:
        i_max = i_min + 1
    return np.clip((load - i_min) / (i_max - i_min), 0, 1)


def synthesize(df, alpha=ALPHA, v_max=V_MAX, per_session_reset=True, original_bug=False):
    """EMA kinematic synthesis.

    original_bug=True reproduces upgrade_trinity.py exactly: the speed is zeroed at
    session boundaries *after* the EMA loop, so the row after a boundary still
    integrates the previous session's speed (creates the 11.36 m/s^2 spike).
    """
    theta = throttle_proxy(df["Current_A"].to_numpy())
    target = np.sqrt(theta) * v_max
    t = df["Time_s"].to_numpy()
    new_trip = np.r_[False, np.diff(t) < 0]

    v = np.zeros(len(df))
    for i in range(1, len(df)):
        prev = 0.0 if (new_trip[i] and per_session_reset and not original_bug) else v[i - 1]
        v[i] = alpha * target[i] + (1 - alpha) * prev
        if new_trip[i] and per_session_reset and not original_bug:
            v[i] = 0.0
    if original_bug:
        # zeroed after the loop; the following row was already computed from the unreset value
        v[new_trip] = 0.0

    dt = np.diff(t, prepend=t[0]).astype(float)
    dt[dt <= 0] = 1.0
    a = np.r_[0.0, np.diff(v / 3.6)] / dt
    a[new_trip] = 0.0
    braking = (a < BRAKE_ACCEL) & (theta < BRAKE_THROTTLE)
    return v, a, braking, theta


def md(df):
    """Markdown table without the optional 'tabulate' dependency."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def session_energy_table(df, v, label):
    rows = []
    for s, g in df.groupby("Session"):
        idx = g.index.to_numpy()
        dur_s = len(idx)  # 1 Hz
        dist_km = v[idx].sum() / 3600.0
        e_power_wh = g["Power_KW"].sum() * 1000 / 3600.0
        dsoc = g["Actual_SOC"].iloc[0] - g["Actual_SOC"].iloc[-1]
        e_soc_wh = dsoc / 100 * PACK_WH
        rows.append({
            "Model": label,
            "Session": int(s) + 1,
            "Config": g["Cell_Config"].iloc[0],
            "Mode": g["Drive_Mode"].iloc[0],
            "Duration_s": dur_s,
            "Mean_speed_kmh": round(v[idx].mean(), 2),
            "Distance_km": round(dist_km, 3),
            "Mean_power_W": round(g["Power_KW"].mean() * 1000, 1),
            "Energy_from_power_Wh": round(e_power_wh, 2),
            "SOC_drop_pct": round(dsoc, 2),
            "Energy_from_SOC_Wh": round(e_soc_wh, 2),
            "Wh_per_km_power": round(e_power_wh / dist_km, 2) if dist_km else np.nan,
            "Wh_per_km_SOC": round(e_soc_wh / dist_km, 2) if dist_km else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(DATA)
    df = load_sessions(raw)
    lines = ["# Kinematic Synthesis Validation (Option B)\n",
             f"Dataset: `{DATA.name}`, {len(df)} rows, {df['Session'].nunique()} sessions at 1 Hz.\n"]

    # ---------------- 1. Reproduction ----------------
    v_bug, a_bug, b_bug, _ = synthesize(df, original_bug=True)
    v_fix, a_fix, b_fix, theta = synthesize(df, original_bug=False)
    rep_err = np.abs(v_bug - df["Speed(km/h)"].to_numpy()).max()
    lines += ["## 1. Reproduction of stored columns\n",
              f"- Max |re-computed speed − stored speed| = {rep_err:.2e} km/h "
              f"({'reproduced exactly' if rep_err < 1e-6 else 'differs'}).\n",
              f"- Stored braking events: {int(df['Braking'].sum())}; re-computed (original): {int(b_bug.sum())}.\n"]

    # ---------------- 2. Plausibility ----------------
    def stats(v, a, b, name):
        return {
            "Variant": name,
            "Max_speed_kmh": round(v.max(), 2),
            "Mean_speed_kmh": round(v.mean(), 2),
            "Max_accel_ms2": round(a.max(), 3),
            "Min_accel_ms2": round(a.min(), 3),
            "Rows_accel_gt_limit": int((a > ACCEL_LIMIT).sum()),
            "Rows_decel_lt_limit": int((a < DECEL_LIMIT).sum()),
            "P99_abs_accel": round(np.quantile(np.abs(a), 0.99), 3),
            "Braking_events": int(b.sum()),
        }
    plaus = pd.DataFrame([stats(v_bug, a_bug, b_bug, "Original (upgrade_trinity.py)"),
                          stats(v_fix, a_fix, b_fix, "Corrected (per-session reset)")])
    plaus.to_csv(OUT / "plausibility.csv", index=False)

    spike = int(np.argmax(a_bug))
    lines += ["## 2. Plausibility checks\n",
              f"Limits used: acceleration ≤ {ACCEL_LIMIT} m/s², deceleration ≥ {DECEL_LIMIT} m/s², "
              f"speed ≤ v_max = {V_MAX} km/h.\n",
              md(plaus), "\n",
              f"\n**Outlier diagnosis:** the max acceleration ({a_bug.max():.2f} m/s²) occurs at row {spike} "
              f"(Time_s = {df['Time_s'].iloc[spike]}, session {df['Session'].iloc[spike] + 1}), "
              f"{'the first row after a session boundary' if df['Time_s'].iloc[spike] <= 1 else 'not at a boundary'}. "
              "In `upgrade_trinity.py` the speed is zeroed at the boundary *after* the EMA loop, so the next row "
              "still integrates the previous session's speed and the jump from 0 creates the spike. "
              "Resetting the EMA state inside the loop (corrected variant) removes it.\n"]

    # ---------------- 3. Sensitivity ----------------
    alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
    vmaxes = [45.0, 55.0, 65.0, 75.0]
    sens = []
    for al in alphas:
        for vm in vmaxes:
            v, a, b, _ = synthesize(df, alpha=al, v_max=vm)
            e = session_energy_table(df, v, "")
            sens.append({
                "alpha": al, "v_max": vm,
                "Max_speed": round(v.max(), 1),
                "Mean_speed": round(v.mean(), 2),
                "P99_abs_accel": round(np.quantile(np.abs(a), 0.99), 3),
                "Max_abs_accel": round(np.abs(a).max(), 3),
                "Braking_events": int(b.sum()),
                "Total_distance_km": round(e["Distance_km"].sum(), 2),
                "Wh_per_km_power": round(e["Energy_from_power_Wh"].sum() / e["Distance_km"].sum(), 2),
            })
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    base = sens[(sens.alpha == ALPHA) & (sens.v_max == V_MAX)].iloc[0]
    lines += ["## 3. Sensitivity to α and v_max (corrected variant)\n",
              md(sens), "\n",
              f"\nAt the default (α = {ALPHA}, v_max = {V_MAX}): total distance {base.Total_distance_km} km, "
              f"{int(base.Braking_events)} braking events. Distance scales linearly with v_max; "
              "α mainly changes acceleration peaks and braking counts, not distance.\n"]

    # ---------------- 4. Energy consistency ----------------
    energy = session_energy_table(df, v_fix, "Corrected")
    energy.to_csv(OUT / "energy_consistency.csv", index=False)
    tot_d = energy["Distance_km"].sum()
    whkm_p = energy["Energy_from_power_Wh"].sum() / tot_d
    whkm_s = energy["Energy_from_SOC_Wh"].sum() / tot_d
    lines += ["## 4. Energy-consistency check\n",
              f"Pack energy assumed for SOC-based energy: {PACK_WH:.0f} Wh (backend model).\n",
              md(energy.drop(columns=["Model"])), "\n",
              f"\n- Aggregate implied consumption: **{whkm_p:.2f} Wh/km** (from measured power) and "
              f"**{whkm_s:.2f} Wh/km** (from SOC drop × {PACK_WH:.0f} Wh).\n",
              f"- DTE model assumes 150 Wh/km (ECO) / 250 Wh/km (SPORT): "
              f"{150 / whkm_p:.0f}× / {250 / whkm_p:.0f}× the power-based value.\n",
              f"- Power-based vs SOC-based energy disagree by {whkm_s / whkm_p:.1f}×, "
              "meaning the logged current/power and the logged SOC drop are themselves inconsistent "
              "for a 3.81 kWh pack (the SOC readout has 0.1 % resolution; ~0.4 % drops are near quantization).\n"]

    # ---------------- Figures ----------------
    s0 = df[df["Session"] == 1]
    idx = s0.index.to_numpy()
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(s0["Time_s"], -s0["Current_A"], color="#9ca3af", lw=1, label="Discharge current (A)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)")
    ax2 = ax1.twinx()
    ax2.plot(s0["Time_s"], v_fix[idx], color="#2563eb", lw=1.6, label="Synthesized speed (km/h)")
    br = idx[b_fix[idx]]
    ax2.scatter(df.loc[br, "Time_s"], v_fix[br], s=8, color="#dc2626", label="Braking event", zorder=3)
    ax2.set_ylabel("Speed (km/h)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, frameon=False)
    ax1.set_title(f"Session 2 ({s0['Cell_Config'].iloc[0]}, {s0['Drive_Mode'].iloc[0]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_speed_trace.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, col, title in [(axes[0], "Total_distance_km", "Total distance (km)"),
                           (axes[1], "Braking_events", "Braking events")]:
        piv = sens.pivot(index="alpha", columns="v_max", values=col)
        im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(vmaxes)), [int(x) for x in vmaxes])
        ax.set_yticks(range(len(alphas)), alphas)
        ax.set_xlabel("v_max (km/h)")
        ax.set_ylabel("α")
        ax.set_title(title, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                ax.text(j, i, f"{val:.1f}" if col.endswith("km") else f"{int(val)}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if val > piv.values.mean() else "black")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist(a_bug, bins=80, color="#f97316", alpha=0.6, label="Original")
    ax.hist(a_fix, bins=80, color="#2563eb", alpha=0.6, label="Corrected")
    ax.set_yscale("log")
    ax.axvline(ACCEL_LIMIT, color="k", ls="--", lw=0.8)
    ax.axvline(DECEL_LIMIT, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Acceleration (m/s²)")
    ax.set_ylabel("Count (log)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_accel_hist.png", dpi=300)
    plt.close(fig)

    lines += ["\n## Figures\n",
              "- `fig_speed_trace.png`: current vs synthesized speed with braking events (one session).\n",
              "- `fig_sensitivity.png`: total distance and braking events across α × v_max.\n",
              "- `fig_accel_hist.png`: acceleration distribution, original vs corrected.\n"]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
