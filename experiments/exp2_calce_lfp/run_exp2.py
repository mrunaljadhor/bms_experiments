"""
Experiment 2: SOC estimation on a public real LiFePO4 benchmark (CALCE A123, 25 °C)
==================================================================================
Data: CALCE A123 LiFePO4 cells (1.1 Ah), DST / US06 / FUDS full discharges (100 % to 2.0 V)
      from cells A1-007 and A1-008, plus a C/20 low-current OCV test (cell A1-007).
      https://calce.umd.edu/data. Cite Xing et al., Appl. Energy 113 (2014) and
      He et al., IJEPES 62 (2014).

Protocol: leave-one-profile-out. Train on two drive cycles (both cells), test on the third.
Model: the same leak-free LSTM as Exp 1 (inputs V, I, T; window 50 s; SOC never an input).
Baselines on the same test windows:
  - Coulomb counting with true SOC0 and with SOC0 errors of -2 % and -10 %;
  - EKF with the MEASURED OCV curve (C/20 charge/discharge average) and R_int fitted on the
    training profiles, initialised with the true SOC0 and with a -10 % error;
  - a constant predictor (mean training SOC).

Run:  python experiments/exp2_calce_lfp/run_exp2.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
WINDOW, BATCH, EPOCHS, PATIENCE, SEED = 50, 32, 40, 5, 42
PROFILES = ["DST", "US06", "FUDS"]          # order of the discharge steps in each file
CELLS = {"A1-007": "DST-US06-FUDS-25/A1-007-DST-US06-FUDS-25-20120827.xlsx",
         "A1-008": "DST-US06-FUDS-25/A1-008-DST-US06-FUDS-25-20120827.xlsx"}
OCV_FILE = "OCV25-20120905/A1-007-OCV-25-20120905.xlsx"


def load_xlsx(rel):
    cache = DATA / (Path(rel).stem + ".pkl")
    if cache.exists():
        return pd.read_pickle(cache)
    xl = pd.ExcelFile(DATA / rel)
    sheets = [s for s in xl.sheet_names if s.startswith("Channel")]
    d = pd.concat([xl.parse(s) for s in sheets], ignore_index=True).sort_values("Test_Time(s)")
    d.to_pickle(cache)
    return d


def extract_runs(d):
    """Return the three full drive-cycle discharges as 1 Hz frames with true SOC."""
    d = d.copy()
    d["net"] = d["Discharge_Capacity(Ah)"] - d["Charge_Capacity(Ah)"]
    g = d.groupby("Step_Index").agg(t0=("Test_Time(s)", "min"), t1=("Test_Time(s)", "max"),
                                    n=("Test_Time(s)", "size"), Imean=("Current(A)", "mean"))
    steps = g[(g.n > 1000) & (g.Imean < -0.1)].sort_values("t0")
    assert len(steps) == 3, steps
    runs = {}
    for name, (_, s) in zip(PROFILES, steps.iterrows()):
        seg = d[(d["Test_Time(s)"] >= s.t0) & (d["Test_Time(s)"] <= s.t1)]
        t = seg["Test_Time(s)"].to_numpy() - s.t0
        grid = np.arange(0, np.floor(t[-1]) + 1)
        r = pd.DataFrame({"t": grid})
        for col, src in [("V", "Voltage(V)"), ("I", "Current(A)"), ("T", "Temperature (C)_1"), ("net", "net")]:
            r[col] = np.interp(grid, t, seg[src].to_numpy())
        q = r.net.iloc[-1] - r.net.iloc[0]               # capacity discharged to the 2.0 V cutoff
        r["SOC"] = 100 * (1 - (r.net - r.net.iloc[0]) / q)
        r.attrs["Q"] = q
        runs[name] = r
    return runs


def ocv_curve():
    d = load_xlsx(OCV_FILE)
    d["net"] = d["Discharge_Capacity(Ah)"] - d["Charge_Capacity(Ah)"]
    big = d.groupby("Step_Index").size()
    big = big[big > 5000].index.tolist()               # C/20 discharge, then C/20 charge
    grid = np.linspace(0, 1, 201)
    curves = []
    for st in big:
        s = d[d.Step_Index == st]
        net = s.net.to_numpy()
        frac = (net - net[0]) / (net[-1] - net[0])       # 0 -> 1 along the step
        soc = 1 - frac if s["Current(A)"].mean() < 0 else frac
        o = np.argsort(soc)
        curves.append(np.interp(grid, soc[o], s["Voltage(V)"].to_numpy()[o]))
    v = np.maximum.accumulate(np.mean(curves, axis=0))  # average of charge and discharge, monotone
    return grid, v


def windows(run, scaler=None):
    X = run[["V", "I", "T"]].to_numpy(np.float32)
    y = run.SOC.to_numpy(np.float32)
    if scaler is not None:
        X = (X - scaler[0]) / scaler[1]
    idx = np.arange(WINDOW, len(X) + 1)
    Xw = np.stack([X[i - WINDOW:i] for i in idx])
    return Xw, y[idx - 1]


def build():
    m = Sequential([Input((WINDOW, 3)), LSTM(64, return_sequences=True), Dropout(0.2),
                    LSTM(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def coulomb(run, soc0):
    return soc0 + 100 * np.cumsum(np.r_[0, run.I.to_numpy()[1:]]) / (3600 * run.attrs["Q"])


def ekf(run, soc0, r_int, og, ov, q=1e-7, r_meas=0.01 ** 2, p0=0.01):
    """Scalar EKF on SOC (fraction). V = OCV(SOC) + R_int * I, discharge current negative."""
    docv = np.gradient(ov, og)
    x, p, Q = soc0 / 100, p0, run.attrs["Q"]
    I, V = run.I.to_numpy(), run.V.to_numpy()
    out = np.empty(len(I))
    for k in range(len(I)):
        if k:
            x += I[k] / (3600 * Q)
            p += q
        x = min(max(x, 0.0), 1.0)
        h = np.interp(x, og, docv)
        kg = p * h / (h * h * p + r_meas)
        x += kg * (V[k] - (np.interp(x, og, ov) + r_int * I[k]))
        p *= (1 - kg * h)
        out[k] = 100 * min(max(x, 0.0), 1.0)
    return out


def metrics(e):
    e = np.asarray(e)
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "MaxErr": float(np.abs(e).max())}


def main():
    t_start = time.time()
    tf.keras.utils.set_random_seed(SEED)
    runs = {(c, p): r for c, f in CELLS.items() for p, r in extract_runs(load_xlsx(f)).items()}
    og, ov = ocv_curve()
    info = {f"{c}/{p}": {"duration_s": int(r.t.iloc[-1]), "capacity_Ah": round(r.attrs["Q"], 4),
                         "I_min": round(float(r.I.min()), 3), "I_max": round(float(r.I.max()), 3)}
            for (c, p), r in runs.items()}
    print(json.dumps(info, indent=1))

    errs, rows, keep = {}, [], {}
    for test_p in PROFILES:
        train_keys = [k for k in runs if k[1] != test_p]
        test_keys = [k for k in runs if k[1] == test_p]
        allX = np.concatenate([runs[k][["V", "I", "T"]].to_numpy(np.float32) for k in train_keys])
        scaler = (allX.mean(0), allX.std(0) + 1e-6)
        # validation = 2 of 10 contiguous chunks of every training run (windows never cross chunks)
        Xtr, ytr, Xva, yva = [], [], [], []
        for k in train_keys:
            r = runs[k]
            for j, chunk in enumerate(np.array_split(np.arange(len(r)), 10)):
                if len(chunk) < WINDOW + 5:
                    continue
                Xw, yw = windows(r.iloc[chunk].reset_index(drop=True), scaler)
                (Xva if j in (3, 8) else Xtr).append(Xw)
                (yva if j in (3, 8) else ytr).append(yw)
        Xtr, ytr, Xva, yva = map(np.concatenate, (Xtr, ytr, Xva, yva))
        model = build()
        t0 = time.time()
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH, verbose=2,
                         callbacks=[EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
        train_s = time.time() - t0
        # R_int by least squares on the training profiles, using the measured OCV
        num = den = 0.0
        for k in train_keys:
            r = runs[k]
            res = r.V.to_numpy() - np.interp(r.SOC.to_numpy() / 100, og, ov)
            num += float((res * r.I).sum()); den += float((r.I ** 2).sum())
        r_int = num / den
        const = float(np.concatenate([runs[k].SOC.to_numpy() for k in train_keys]).mean())

        for k in test_keys:
            r = runs[k]
            Xw, yw = windows(r, scaler)
            pred = model.predict(Xw, batch_size=1024, verbose=0).ravel()
            sl = slice(WINDOW - 1, None)
            ests = {
                "LSTM (no SOC0)": pred,
                "Coulomb, true SOC0": coulomb(r, 100.0)[sl],
                "Coulomb, SOC0 -2 %": coulomb(r, 98.0)[sl],
                "Coulomb, SOC0 -10 %": coulomb(r, 90.0)[sl],
                "EKF (measured OCV), true SOC0": ekf(r, 100.0, r_int, og, ov)[sl],
                "EKF (measured OCV), SOC0 -10 %": ekf(r, 90.0, r_int, og, ov)[sl],
                "Constant (mean training SOC)": np.full(len(yw), const),
            }
            for name, e in ests.items():
                err = e - yw
                errs.setdefault(name, []).append(err)
                rows.append({"Test_profile": test_p, "Cell": k[0], "Method": name, **metrics(err)})
            keep[k] = (r.t.to_numpy()[sl], yw, ests)
        rows.append({"Test_profile": test_p, "Cell": "-", "Method": "_train", "MAE": train_s,
                     "RMSE": len(hist.history["loss"]), "MaxErr": r_int})
        print(f"fold {test_p}: {len(hist.history['loss'])} epochs, {train_s:.0f} s, R_int={r_int*1000:.1f} mOhm")

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_run_results.csv", index=False)
    summary = pd.DataFrame([{"Method": n, **metrics(np.concatenate(e))} for n, e in errs.items()]).round(3)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))

    # figure: FUDS, cell A1-008
    t, y, ests = keep[("A1-008", "FUDS")]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(t / 60, y, color="k", lw=1.2, label="True SOC")
    for name, c, ls in [("LSTM (no SOC0)", "#2563eb", "-"), ("EKF (measured OCV), SOC0 -10 %", "#16a34a", "--"),
                        ("Coulomb, SOC0 -10 %", "#dc2626", ":")]:
        ax.plot(t / 60, ests[name], color=c, lw=0.9, ls=ls, label=name.replace("SOC0", "SOC$_0$"))
    ax.set_xlabel("Time (min)", fontsize=7)
    ax.set_ylabel("SOC (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calce_fuds.png", dpi=600)
    fig.savefig(OUT / "fig_calce_fuds.pdf")
    plt.close(fig)

    json.dump({"runs": info, "ocv_soc_grid": og.tolist(), "ocv_V": ov.tolist(),
               "runtime_s": round(time.time() - t_start, 1)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
