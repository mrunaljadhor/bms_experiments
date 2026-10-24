"""
Experiment 1: leak-free SOC LSTM with scenario-level hold-out + real stand-test evaluation
=========================================================================================
Fixes the issues of bms_lstm_pipeline.py:
  * the target (SOC) is NOT part of the input window (inputs = V, I, T only)
  * windows never cross scenario / session boundaries
  * scaler fitted on training scenarios only
  * hold-out is by whole scenario (synthetic) and by whole session (real, leave-one-session-out)

Baselines on the 6 real stand-test sessions: Coulomb counting and an EKF
(pack-level, OCV curve of the synthetic generator, R_int fitted on training sessions),
with the true initial SOC and with a +2 % initial-SOC error.

Run:  python experiments/exp1_soc_lstm/run_exp1.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import synthetic_data_generator as sdg  # noqa: E402  (reuses the project's generator functions)

SEED = 42
WINDOW = 50
BATCH = 32
EPOCHS = 50
PATIENCE = 5
CAPACITY_AH = 30.9
V_NOM = 63.5
FEATURES = ["Voltage_V", "Current_A", "Max_Temp_C"]  # current: discharge-positive


def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------- data
def build_synthetic(real_df, n_scenarios=100):
    set_seed()
    profiles = sdg.extract_discharge_profiles(real_df)
    scenarios = [sdg.generate_synthetic_scenario(i, profiles, None) for i in range(n_scenarios)]
    df = pd.concat(scenarios, ignore_index=True)
    return df[["Scenario_ID", "Time_s", *FEATURES, "Actual_SOC", "Drive_Mode"]]


def load_real():
    df = pd.read_csv(ROOT / "trinity_ev_dataset.csv")
    df["Session"] = (df["Time_s"].diff() < 0).cumsum()
    df["Current_A"] = -df["Current_A"]  # convert to discharge-positive (synthetic convention)
    return df


def windows(df, group_col, scaler):
    X, y, g = [], [], []
    for key, part in df.groupby(group_col, sort=True):
        feats = scaler.transform(part[FEATURES].to_numpy())
        soc = part["Actual_SOC"].to_numpy() / 100.0
        for i in range(WINDOW, len(part) + 1):
            X.append(feats[i - WINDOW:i])
            y.append(soc[i - 1])  # SOC at the last step of the window (estimation, not forecasting)
            g.append(key)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(g)


def build_model():
    model = Sequential([
        Input((WINDOW, len(FEATURES))),
        LSTM(64, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def metrics(y_true_pct, y_pred_pct):
    y_true_pct, y_pred_pct = np.asarray(y_true_pct), np.asarray(y_pred_pct)
    return {
        "MAE": float(mean_absolute_error(y_true_pct, y_pred_pct)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_pct, y_pred_pct))),
        "MaxErr": float(np.max(np.abs(y_true_pct - y_pred_pct))),
        "R2": float(r2_score(y_true_pct, y_pred_pct)) if np.var(y_true_pct) > 0 else float("nan"),
        "N": int(len(y_true_pct)),
    }


# ----------------------------------------------------------------------------- baselines
def ocv(soc):
    """Pack OCV curve used by the synthetic generator (noise-free)."""
    soc = np.asarray(soc, float)
    return np.where(soc >= 80, V_NOM * (0.98 + 0.02 * (soc - 80) / 20),
                    np.where(soc >= 20, V_NOM * (0.90 + 0.08 * (soc - 20) / 60),
                             V_NOM * (0.50 + 0.40 * soc / 20)))


def docv(soc):
    return V_NOM * 0.02 / 20 if soc >= 80 else (V_NOM * 0.08 / 60 if soc >= 20 else V_NOM * 0.40 / 20)


def coulomb(session, soc0):
    i = session["Current_A"].to_numpy()
    return soc0 - np.cumsum(np.r_[0.0, i[1:]]) / (3600 * CAPACITY_AH) * 100


def fit_rint(train_sessions):
    """Least squares V = OCV(SOC) - R*I on training sessions."""
    v = np.concatenate([s["Voltage_V"].to_numpy() for s in train_sessions])
    i = np.concatenate([s["Current_A"].to_numpy() for s in train_sessions])
    soc = np.concatenate([s["Actual_SOC"].to_numpy() for s in train_sessions])
    resid = ocv(soc) - v
    return float(np.dot(i, resid) / np.dot(i, i))


def ekf(session, soc0, r_int, q=1e-4, r_meas=0.05 ** 2, p0=4.0):
    i = session["Current_A"].to_numpy()
    v = session["Voltage_V"].to_numpy()
    x, p, out = soc0, p0, []
    for k in range(len(i)):
        if k:
            x -= i[k] / (3600 * CAPACITY_AH) * 100
            p += q
        h = docv(x)
        s = h * p * h + r_meas
        kg = p * h / s
        x += kg * (v[k] - (float(ocv(x)) - r_int * i[k]))
        p *= (1 - kg * h)
        out.append(x)
    return np.asarray(out)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    real = load_real()
    real_raw = pd.read_csv(ROOT / "trinity_ev_dataset.csv")
    syn = build_synthetic(real_raw)
    syn.to_csv(OUT / "synthetic_with_scenario_ids.csv", index=False)

    ids = np.random.RandomState(SEED).permutation(syn["Scenario_ID"].unique())
    n = len(ids)
    tr_ids, va_ids, te_ids = ids[:int(.7 * n)], ids[int(.7 * n):int(.85 * n)], ids[int(.85 * n):]
    tr = syn[syn.Scenario_ID.isin(tr_ids)]
    va = syn[syn.Scenario_ID.isin(va_ids)]
    te = syn[syn.Scenario_ID.isin(te_ids)]

    scaler = MinMaxScaler().fit(tr[FEATURES].to_numpy())
    Xtr, ytr, _ = windows(tr, "Scenario_ID", scaler)
    Xva, yva, _ = windows(va, "Scenario_ID", scaler)
    Xte, yte, gte = windows(te, "Scenario_ID", scaler)
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")

    # ---------------- 1a. synthetic scenario hold-out
    set_seed()
    model = build_model()
    tt = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
                     callbacks=[EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)],
                     verbose=2)
    train_time = time.time() - tt
    model.save(OUT / "soc_lstm_leakfree.keras")
    pred_te = model.predict(Xte, verbose=0).ravel() * 100
    res = {"synthetic_test": metrics(yte * 100, pred_te),
           "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)), "test_windows": int(len(Xte)),
           "scenarios": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
           "epochs_run": len(hist.history["loss"]), "train_time_s": round(train_time, 1),
           "params": int(model.count_params())}

    # inference latency (single window, CPU)
    one = Xte[:1]
    model.predict(one, verbose=0)
    lat = []
    for _ in range(50):
        s = time.perf_counter()
        model(one, training=False)
        lat.append((time.perf_counter() - s) * 1000)
    res["latency_ms_single_window_median"] = round(float(np.median(lat)), 2)
    res["model_size_kb"] = round((OUT / "soc_lstm_leakfree.keras").stat().st_size / 1024, 1)

    # ---------------- 1b. real stand-test sessions
    sessions = {k: g.reset_index(drop=True) for k, g in real.groupby("Session")}
    Xr, yr, gr = windows(real, "Session", scaler)
    zero = model.predict(Xr, verbose=0).ravel() * 100
    rows, traces = [], {}
    for k, sess in sessions.items():
        mask = gr == k
        others = [s for j, s in sessions.items() if j != k]
        truth_w = yr[mask] * 100
        # fine-tune a copy on the other 5 sessions (leave-one-session-out)
        set_seed()
        ft = tf.keras.models.clone_model(model)
        ft.set_weights(model.get_weights())
        ft.compile(optimizer=Adam(1e-4), loss="mse")
        ft.fit(Xr[~mask], yr[~mask], epochs=10, batch_size=BATCH, verbose=0)
        pred_ft = ft.predict(Xr[mask], verbose=0).ravel() * 100

        soc_true = sess["Actual_SOC"].to_numpy()
        soc0 = soc_true[0]
        r_int = fit_rint(others)
        cc = coulomb(sess, soc0)[WINDOW - 1:]
        cc_err = coulomb(sess, soc0 + 2)[WINDOW - 1:]
        kf = ekf(sess, soc0, r_int)[WINDOW - 1:]
        kf_err = ekf(sess, soc0 + 2, r_int)[WINDOW - 1:]
        label = f"S{k + 1} {sess['Cell_Config'][0]}-{sess['Drive_Mode'][0]}"
        for name, pred in [("LSTM zero-shot (synthetic-trained)", zero[mask]),
                           ("LSTM fine-tuned (LOSO)", pred_ft),
                           ("Coulomb counting (true SOC0)", cc),
                           ("Coulomb counting (SOC0 +2%)", cc_err),
                           ("EKF (true SOC0)", kf),
                           ("EKF (SOC0 +2%)", kf_err)]:
            rows.append({"Session": label, "Method": name, **metrics(truth_w, pred)})
        traces[label] = {"t": sess["Time_s"].to_numpy()[WINDOW - 1:], "truth": truth_w,
                         "zero": zero[mask], "ft": pred_ft, "cc": cc, "kf_err": kf_err, "r_int": r_int}

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "real_sessions_per_session.csv", index=False)
    agg = per.groupby("Method", sort=False).apply(
        lambda d: pd.Series({"MAE": np.average(d.MAE, weights=d.N),
                             "RMSE": np.sqrt(np.average(d.RMSE ** 2, weights=d.N)),
                             "MaxErr": d.MaxErr.max()}), include_groups=False).round(3)
    agg.to_csv(OUT / "real_sessions_summary.csv")
    res["real_sessions_summary"] = agg.reset_index().to_dict(orient="records")
    res["total_runtime_s"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    h = hist.history
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(np.array(h["loss"]), label="Train")
    ax.plot(np.array(h["val_loss"]), label="Validation")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (SOC fraction²)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_training_history.png", dpi=300); plt.close(fig)

    sid = gte[0]
    m = gte == sid
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(yte[m] * 100, color="k", lw=1.5, label="True SOC")
    ax.plot(pred_te[m], color="#2563eb", lw=1, ls="--", label="LSTM estimate")
    ax.set_xlabel("Time step (s)"); ax.set_ylabel("SOC (%)")
    ax.set_title(f"Held-out synthetic scenario {sid}", fontsize=9)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_synthetic_test_scenario.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=False)
    for ax, (label, tr_) in zip(axes.ravel(), traces.items()):
        ax.step(tr_["t"], tr_["truth"], color="k", lw=1.4, where="post", label="BMS SOC")
        ax.plot(tr_["t"], tr_["ft"], color="#2563eb", lw=1, label="LSTM fine-tuned")
        ax.plot(tr_["t"], tr_["cc"], color="#16a34a", lw=1, ls="--", label="Coulomb (true SOC0)")
        ax.plot(tr_["t"], tr_["kf_err"], color="#dc2626", lw=1, ls=":", label="EKF (SOC0 +2%)")
        ax.set_title(label, fontsize=9); ax.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("SOC (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_sessions.png", dpi=300); plt.close(fig)

    print(json.dumps(res, indent=2))
    print(per.round(3).to_string())


if __name__ == "__main__":
    main()
