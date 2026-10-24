"""
Experiment 2b: a stronger EKF baseline for the CALCE benchmark (no LSTM retraining)
==================================================================================
The R_int-only EKF in run_exp2.py is a weak baseline. Here the standard first-order RC
(Thevenin) model is used:
    V_k  = OCV(SOC_k) + R0*I_k + V1_k + b          (discharge current negative)
    V1_k = a*V1_{k-1} + R1*(1-a)*I_k,   a = exp(-dt/tau)
OCV is measured (C/20 charge/discharge average). R0, R1, b and tau are fitted by least squares
on the TRAINING drive cycles of each fold, and the EKF noise settings (q, r) are chosen on
the training cycles too. Leave-one-profile-out folds, identical test windows to run_exp2.py.

Run after run_exp2.py:  python experiments/exp2_calce_lfp/run_exp2b_ekf.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_exp2 import CELLS, PROFILES, WINDOW, extract_runs, load_xlsx, metrics, ocv_curve

OUT = Path(__file__).resolve().parent


def filt(I, tau):
    a = np.exp(-1.0 / tau)
    x = np.zeros_like(I)
    for k in range(1, len(I)):
        x[k] = a * x[k - 1] + (1 - a) * I[k]
    return x


def fit_ecm(train_runs, og, ov):
    best = None
    for tau in [5, 10, 20, 40, 60, 100, 150, 250, 400]:
        A, y = [], []
        for r in train_runs:
            I = r.I.to_numpy()
            A.append(np.c_[I, filt(I, tau), np.ones_like(I)])
            y.append(r.V.to_numpy() - np.interp(r.SOC.to_numpy() / 100, og, ov))
        A, y = np.concatenate(A), np.concatenate(y)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, tau, *coef)
    return {"rmse_V": best[0], "tau": best[1], "R0": best[2], "R1": best[3], "b": best[4]}


def ekf_rc(r, soc0, p, og, ov, q, rv, p0=0.01):
    docv = np.gradient(ov, og)
    a = np.exp(-1.0 / p["tau"])
    Q = r.attrs["Q"]
    I, V = r.I.to_numpy(), r.V.to_numpy()
    x = np.array([soc0 / 100, 0.0])
    P = np.diag([p0, 1e-4])
    F = np.diag([1.0, a])
    Qn = np.diag([q, 1e-6])
    out = np.empty(len(I))
    for k in range(len(I)):
        if k:
            x = np.array([x[0] + I[k] / (3600 * Q), a * x[1] + p["R1"] * (1 - a) * I[k]])
            P = F @ P @ F.T + Qn
        x[0] = min(max(x[0], 0.0), 1.0)
        H = np.array([np.interp(x[0], og, docv), 1.0])
        S = H @ P @ H + rv
        K = P @ H / S
        x = x + K * (V[k] - (np.interp(x[0], og, ov) + p["R0"] * I[k] + x[1] + p["b"]))
        P = (np.eye(2) - np.outer(K, H)) @ P
        out[k] = 100 * min(max(x[0], 0.0), 1.0)
    return out


def main():
    runs = {(c, pr): r for c, f in CELLS.items() for pr, r in extract_runs(load_xlsx(f)).items()}
    og, ov = ocv_curve()
    grid_q, grid_r = [1e-8, 1e-7, 1e-6, 1e-5], [0.005 ** 2, 0.01 ** 2, 0.02 ** 2, 0.05 ** 2]
    rows, errs, fits = [], {}, {}
    for test_p in PROFILES:
        train = [runs[k] for k in runs if k[1] != test_p]
        p = fit_ecm(train, og, ov)
        # choose (q, r) on the training cycles, starting from a -10 % SOC0 error
        scores = {}
        for q, rv in itertools.product(grid_q, grid_r):
            e = np.concatenate([ekf_rc(r, 90.0, p, og, ov, q, rv) - r.SOC.to_numpy() for r in train])
            scores[(q, rv)] = float(np.abs(e).mean())
        q, rv = min(scores, key=scores.get)
        fits[test_p] = {**{k: round(v, 5) for k, v in p.items()}, "q": q, "r": rv, "train_MAE": round(scores[(q, rv)], 3)}
        print(test_p, fits[test_p])
        for k in [k for k in runs if k[1] == test_p]:
            r = runs[k]
            y = r.SOC.to_numpy()[WINDOW - 1:]
            for name, soc0 in [("EKF-1RC (measured OCV), true SOC0", 100.0), ("EKF-1RC (measured OCV), SOC0 -10 %", 90.0)]:
                err = ekf_rc(r, soc0, p, og, ov, q, rv)[WINDOW - 1:] - y
                errs.setdefault(name, []).append(err)
                rows.append({"Test_profile": test_p, "Cell": k[0], "Method": name, **metrics(err)})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_run_results_ekf1rc.csv", index=False)
    summ = pd.DataFrame([{"Method": n, **metrics(np.concatenate(e))} for n, e in errs.items()]).round(3)
    old = pd.read_csv(OUT / "summary.csv")
    allsum = pd.concat([old, summ], ignore_index=True)
    allsum.to_csv(OUT / "summary_with_ekf1rc.csv", index=False)
    json.dump(fits, open(OUT / "ekf1rc_fits.json", "w"), indent=1)
    print(allsum.to_string(index=False))
    print(per.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
