"""
Experiment 10: uncertainty-aware SOC (MC dropout + split-conformal calibration) and risk-aware AMSA
====================================================================================================
1. Retrain the multi-temperature CALCE LSTM of run_exp2c_multitemp.py (same folds, seed, windows,
   determinism) and SAVE the model of each fold (resumable: a saved fold is not retrained).
2. Monte-Carlo dropout: T = 30 stochastic passes (dropout active) give a mean and a std per window.
3. Split-conformal calibration on the validation windows of the TRAINING cycles (chunks 3 and 8):
     two-sided 95 % interval  mu +/- q2 * sigma,  q2 = conformal quantile of |y - mu| / sigma
     one-sided 95 % lower bound  mu - q1 * sigma, q1 = conformal quantile of (mu - y) / sigma
   Also reported: the uncalibrated Gaussian interval mu +/- 1.96 sigma.
4. Evaluation on the unseen drive cycle (all temperatures 0-50 °C; 25 °C reported separately):
   MAE of the MC mean, interval coverage and width, lower-bound violation rate.
5. Risk-aware AMSA: for the 8 route directions of Exp 3, the decision "reachable in ECO" taken from
   (a) the point estimate mu and (b) the lower bound is compared with the decision from the true SOC
   on every 25 °C test window: unsafe acceptances (estimate says reachable, truth says not) and
   false refusals (estimate says not reachable, truth says reachable).

Run:  python experiments/exp2_calce_lfp/run_exp10_uq.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from run_exp2 import PROFILES, WINDOW, build, extract_runs, load_xlsx, windows
from run_exp2c_multitemp import BATCH, EPOCHS, PATIENCE, SEED, STRIDE, TEMPS

OUT = Path(__file__).resolve().parent
MODELS = OUT / "uq_models"
ROUTES = OUT.parent / "exp3_amsa_routes" / "routes_summary.csv"
T_MC, ALPHA = 30, 0.05


def conformal_q(scores, alpha):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def mc_predict(model, X, T=T_MC, bs=4096):
    out = np.empty((T, len(X)), np.float32)
    for t in range(T):
        out[t] = np.concatenate([model(X[i:i + bs], training=True).numpy().ravel() for i in range(0, len(X), bs)])
    return out.mean(0), out.std(0) + 1e-3


def main():
    t_start = time.time()
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    MODELS.mkdir(exist_ok=True)
    runs = {}
    for temp, date in TEMPS.items():
        for cell in ["A1-007", "A1-008"]:
            rel = f"DST-US06-FUDS-{temp}/{cell}-DST-US06-FUDS-{temp}-{date}.xlsx"
            for p, r in extract_runs(load_xlsx(rel)).items():
                runs[(temp, cell, p)] = r

    per_run, store = [], {}
    for test_p in PROFILES:
        ck = OUT / f"uq_fold_{test_p}.npz"
        train_keys = [k for k in runs if k[2] != test_p]
        allX = np.concatenate([runs[k][["V", "I", "T"]].to_numpy(np.float32) for k in train_keys])
        scaler = (allX.mean(0), allX.std(0) + 1e-6)
        if ck.exists():
            store[test_p] = dict(np.load(ck, allow_pickle=True))
            print(f"[{test_p}] loaded from checkpoint", flush=True)
            continue
        Xtr, ytr, Xva, yva = [], [], [], []
        for k in train_keys:
            r = runs[k]
            for j, chunk in enumerate(np.array_split(np.arange(len(r)), 10)):
                if len(chunk) < WINDOW + 5:
                    continue
                Xw, yw = windows(r.iloc[chunk].reset_index(drop=True), scaler)
                Xw, yw = Xw[::STRIDE], yw[::STRIDE]
                (Xva if j in (3, 8) else Xtr).append(Xw)
                (yva if j in (3, 8) else ytr).append(yw)
        Xtr, ytr, Xva, yva = map(np.concatenate, (Xtr, ytr, Xva, yva))
        mpath = MODELS / f"lstm_{test_p}.keras"
        if mpath.exists():
            model = tf.keras.models.load_model(mpath)
        else:
            model = build()
            model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH, verbose=2,
                      callbacks=[EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
            model.save(mpath)
        del Xtr, ytr
        # calibration on the validation windows
        mu_v, sd_v = mc_predict(model, Xva)
        q2 = conformal_q(np.abs(yva - mu_v) / sd_v, ALPHA)
        q1 = conformal_q((mu_v - yva) / sd_v, ALPHA)
        # test: the unseen drive cycle at every temperature
        keys, ys, mus, sds, det = [], [], [], [], []
        for k in [k for k in runs if k[2] == test_p]:
            X, y = windows(runs[k], scaler)
            mu, sd = mc_predict(model, X)
            d = model.predict(X, batch_size=4096, verbose=0).ravel()
            keys += [f"{k[0]}|{k[1]}"] * len(y); ys.append(y); mus.append(mu); sds.append(sd); det.append(d)
        store[test_p] = {"key": np.array(keys), "y": np.concatenate(ys), "mu": np.concatenate(mus),
                         "sd": np.concatenate(sds), "det": np.concatenate(det), "q2": q2, "q1": q1}
        np.savez(ck, **store[test_p])
        print(f"[{test_p}] q2={q2:.2f} q1={q1:.2f}  ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- metrics ----------------
    rows = []
    for p, s in store.items():
        y, mu, sd, q2, q1 = s["y"], s["mu"], s["sd"], float(s["q2"]), float(s["q1"])
        temp = np.array([k.split("|")[0] for k in s["key"]])
        for tsel in sorted(set(temp), key=int) + ["all"]:
            m = np.ones(len(y), bool) if tsel == "all" else temp == tsel
            e = mu[m] - y[m]
            rows.append({"Test_profile": p, "Temp_C": tsel, "N": int(m.sum()),
                         "MAE_det": float(np.abs(s["det"][m] - y[m]).mean()), "MAE_mc": float(np.abs(e).mean()),
                         "mean_sigma": float(sd[m].mean()), "q2": q2, "q1": q1,
                         "cov95_gauss": float((np.abs(e) <= 1.96 * sd[m]).mean()),
                         "width95_gauss": float(2 * 1.96 * sd[m].mean()),
                         "cov95_conformal": float((np.abs(e) <= q2 * sd[m]).mean()),
                         "width95_conformal": float(2 * q2 * sd[m].mean()),
                         "lower_margin_mean": float(q1 * sd[m].mean()),
                         "lower_margin_p95": float(np.quantile(q1 * sd[m], 0.95)),
                         "lower_violation": float((y[m] < mu[m] - q1 * sd[m]).mean())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "uq_metrics.csv", index=False)

    def pooled(temp_filter):
        parts = [(s, np.array([k.split("|")[0] for k in s["key"]])) for s in store.values()]
        sel = [(s, (t == temp_filter) if temp_filter else np.ones(len(t), bool)) for s, t in parts]
        y = np.concatenate([s["y"][m] for s, m in sel]); mu = np.concatenate([s["mu"][m] for s, m in sel])
        sd = np.concatenate([s["sd"][m] for s, m in sel])
        q2 = np.concatenate([np.full(m.sum(), float(s["q2"])) for s, m in sel])
        q1 = np.concatenate([np.full(m.sum(), float(s["q1"])) for s, m in sel])
        e = mu - y
        return y, mu, sd, q1, {
            "MAE_mc": float(np.abs(e).mean()), "mean_sigma": float(sd.mean()),
            "cov95_gauss": float((np.abs(e) <= 1.96 * sd).mean()), "width95_gauss": float(2 * 1.96 * sd.mean()),
            "cov95_conformal": float((np.abs(e) <= q2 * sd).mean()), "width95_conformal": float(2 * (q2 * sd).mean()),
            "lower_margin_mean": float((q1 * sd).mean()), "lower_margin_median": float(np.median(q1 * sd)),
            "lower_margin_p95": float(np.quantile(q1 * sd, 0.95)), "lower_violation": float((y < mu - q1 * sd).mean())}

    y25, mu25, sd25, q1_25, s25 = pooled("25")
    _, _, _, _, sall = pooled(None)

    # ---------------- risk-aware AMSA on the 8 routes ----------------
    routes = pd.read_csv(ROUTES)
    lb25 = mu25 - q1_25 * sd25
    amsa = []
    for _, r in routes.iterrows():
        th = r.SOC_min_ECO_pct          # minimum true SOC to arrive in ECO mode
        truth = y25 >= th
        for name, est in [("point estimate", mu25), ("95 % lower bound", lb25)]:
            ok = est >= th
            amsa.append({"Route": r.Route, "ECO_threshold_pct": th, "Estimator": name,
                         "unsafe_accept_pct": 100 * float((ok & ~truth).mean()),
                         "false_refusal_pct": 100 * float((~ok & truth).mean()),
                         "truth_reachable_pct": 100 * float(truth.mean())})
    amsa = pd.DataFrame(amsa)
    amsa.to_csv(OUT / "uq_amsa_routes.csv", index=False)
    agg = amsa.groupby("Estimator")[["unsafe_accept_pct", "false_refusal_pct"]].mean().round(3).to_dict("index")

    summary = {"T_MC": T_MC, "alpha": ALPHA, "test_25C": {k: round(v, 3) for k, v in s25.items()},
               "test_all_temps": {k: round(v, 3) for k, v in sall.items()},
               "amsa_mean_over_8_routes_25C": agg, "runtime_s": round(time.time() - t_start, 1)}
    json.dump(summary, open(OUT / "uq_summary.json", "w"), indent=1)
    print(met[met.Temp_C.isin(["25", "all"])].round(3).to_string(index=False))
    print(amsa.round(2).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
