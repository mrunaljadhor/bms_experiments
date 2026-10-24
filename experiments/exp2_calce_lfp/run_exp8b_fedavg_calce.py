"""
Experiment 8b: federated averaging on the CALCE A123 LiFePO4 benchmark
=====================================================================
Replaces the stand-test FedAvg comparison (Exp 8), whose sessions span < 0.5 % SOC and where a
constant predictor (1.22 %) beats every variant. Here SOC spans 100 -> 0 %.

Clients: 14 = 2 cells x 7 temperatures (0-50 °C); each client holds only its own records.
Folds: leave-one-drive-cycle-out, exactly as run_exp2c_multitemp.py (the centralized reference):
       clients train on the two training cycles; test = unseen cycle at 25 °C (both cells).
Compared:
  - FedAvg: shared random init, up to 40 rounds x 2 local epochs (Adam 1e-3), sample-weighted
            averaging; stops after 6 rounds without improvement of the federated validation loss,
            and the best round is kept.
  - Local-only: each client trains alone (early stopping on its own validation chunks);
            reported as the mean over clients.
  - Centralized: taken from run_exp2c_multitemp.py (same folds, windows and architecture).
Communication: float32 weights, download + upload per client per round, vs raw 1 Hz V/I/T/SOC.

Run:  python experiments/exp2_calce_lfp/run_exp8b_fedavg_calce.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from run_exp2 import PROFILES, WINDOW, build, extract_runs, load_xlsx, metrics, windows
from run_exp2c_multitemp import STRIDE, TEMPS

OUT = Path(__file__).resolve().parent
SEED, ROUNDS, BATCH, LOCAL_EPOCHS_MAX, PATIENCE = 42, 40, 64, 30, 5
LOCAL_EPOCHS, ROUND_PATIENCE = 2, 6


def client_data(runs, keys, scaler):
    Xtr, ytr, Xva, yva = [], [], [], []
    for k in keys:
        r = runs[k]
        for j, chunk in enumerate(np.array_split(np.arange(len(r)), 10)):
            if len(chunk) < WINDOW + 5:
                continue
            Xw, yw = windows(r.iloc[chunk].reset_index(drop=True), scaler)
            Xw, yw = Xw[::STRIDE], yw[::STRIDE]
            (Xva if j in (3, 8) else Xtr).append(Xw)
            (yva if j in (3, 8) else ytr).append(yw)
    return tuple(map(np.concatenate, (Xtr, ytr, Xva, yva)))


def main():
    t_start = time.time()
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    runs = {}
    for temp, date in TEMPS.items():
        for cell in ["A1-007", "A1-008"]:
            rel = f"DST-US06-FUDS-{temp}/{cell}-DST-US06-FUDS-{temp}-{date}.xlsx"
            for p, r in extract_runs(load_xlsx(rel)).items():
                runs[(temp, cell, p)] = r
    clients = [(t, c) for t in TEMPS for c in ["A1-007", "A1-008"]]

    rows, curves, comm = [], [], {}
    for test_p in PROFILES:
        ck = OUT / f"fedavg_fold_{test_p}.json"
        if ck.exists():                      # resume: fold already finished
            d = json.load(open(ck)); rows += d["rows"]; curves += d["curves"]; comm[test_p] = d["comm"]
            print(f"[{test_p}] loaded from checkpoint"); continue
        fold_rows0, fold_curves0 = len(rows), len(curves)
        train_keys = [k for k in runs if k[2] != test_p]
        allX = np.concatenate([runs[k][["V", "I", "T"]].to_numpy(np.float32) for k in train_keys])
        scaler = (allX.mean(0), allX.std(0) + 1e-6)   # global feature scaling (shared constants)
        data = {cl: client_data(runs, [k for k in train_keys if (k[0], k[1]) == cl], scaler) for cl in clients}
        n = np.array([len(data[cl][0]) for cl in clients], float)
        test = {k: windows(runs[k], scaler) for k in runs if k[2] == test_p and k[0] == "25"}

        def evaluate(model):
            errs = {k: model.predict(X, batch_size=2048, verbose=0).ravel() - y for k, (X, y) in test.items()}
            return metrics(np.concatenate(list(errs.values()))), errs

        # ---------------- FedAvg ----------------
        glob = build()
        w_glob = glob.get_weights()
        best = (np.inf, None, 0)
        local = build()
        t0 = time.time()
        for rnd in range(1, ROUNDS + 1):
            new_w, val_loss = [], 0.0
            for i, cl in enumerate(clients):
                Xtr, ytr, Xva, yva = data[cl]
                local.set_weights(w_glob)
                local.compile(optimizer=Adam(1e-3), loss="mse")
                local.fit(Xtr, ytr, epochs=LOCAL_EPOCHS, batch_size=BATCH, verbose=0, shuffle=True)
                new_w.append(local.get_weights())
            w_glob = [sum(n[i] * new_w[i][j] for i in range(len(clients))) / n.sum() for j in range(len(w_glob))]
            glob.set_weights(w_glob)
            glob.compile(optimizer=Adam(1e-3), loss="mse")
            nv = 0
            for cl in clients:
                Xva, yva = data[cl][2], data[cl][3]
                val_loss += float(np.atleast_1d(glob.evaluate(Xva, yva, batch_size=2048, verbose=0))[0]) * len(Xva)
                nv += len(Xva)
            val_loss /= nv
            m, _ = evaluate(glob)
            curves.append({"Test_profile": test_p, "Round": rnd, "Fed_val_MSE": val_loss, "Test_MAE": m["MAE"]})
            print(f"[{test_p}] round {rnd:2d}  val MSE {val_loss:7.3f}  (test MAE {m['MAE']:.3f})", flush=True)
            if val_loss < best[0]:
                best = (val_loss, [w.copy() for w in w_glob], rnd)
            elif rnd - best[2] >= ROUND_PATIENCE:
                print(f"[{test_p}] early stop at round {rnd}", flush=True)
                break
        glob.set_weights(best[1])
        m, errs = evaluate(glob)
        rows.append({"Test_profile": test_p, "Method": "FedAvg", **m, "best_round": best[2],
                     "train_s": round(time.time() - t0, 1)})
        np.savez(OUT / f"fedavg_errs_{test_p}.npz", **{f"{k[1]}": v for k, v in errs.items()})

        # ---------------- Local-only ----------------
        loc = []
        for cl in clients:
            Xtr, ytr, Xva, yva = data[cl]
            mdl = build()
            mdl.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=LOCAL_EPOCHS_MAX, batch_size=BATCH, verbose=0,
                    callbacks=[EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
            mm, _ = evaluate(mdl)
            loc.append(mm["MAE"])
            print(f"[{test_p}] local-only client {cl}: test MAE {mm['MAE']:.3f}", flush=True)
        rows.append({"Test_profile": test_p, "Method": "Local-only (mean over clients)", "MAE": float(np.mean(loc)),
                     "RMSE": np.nan, "MaxErr": np.nan, "best_round": np.nan, "train_s": np.nan,
                     "Local_min": float(np.min(loc)), "Local_max": float(np.max(loc))})

        # ---------------- communication ----------------
        model_bytes = sum(w.size for w in w_glob) * 4
        raw_bytes = sum(len(runs[k]) * 4 * 4 for k in train_keys)  # V, I, T, SOC as float32 at 1 Hz
        comm[test_p] = {"model_KB": model_bytes / 1024, "per_round_MB": 2 * model_bytes * len(clients) / 2 ** 20,
                        "rounds_to_best": best[2], "total_to_best_MB": 2 * model_bytes * len(clients) * best[2] / 2 ** 20,
                        "raw_upload_MB": raw_bytes / 2 ** 20}
        json.dump({"rows": rows[fold_rows0:], "curves": curves[fold_curves0:], "comm": comm[test_p]},
                  open(ck, "w"), default=float)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fedavg_calce_results.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "fedavg_calce_rounds.csv", index=False)
    # pooled metrics over the three folds (FedAvg), for comparison with the centralized 3.17 %
    pooled = np.concatenate([np.concatenate(list(np.load(OUT / f"fedavg_errs_{p}.npz").values())) for p in PROFILES])
    cen = json.load(open(OUT / "multitemp_results.json"))["LSTM_multitemp_25C"]
    summary = {"FedAvg_25C": {k: round(v, 3) for k, v in metrics(pooled).items()},
               "Centralized_25C (run_exp2c)": cen,
               "Local_only_mean_MAE_per_fold": res[res.Method.str.startswith("Local")].set_index("Test_profile").MAE.round(3).to_dict(),
               "Local_only_mean_MAE": round(float(res[res.Method.str.startswith("Local")].MAE.mean()), 3),
               "communication": comm, "clients": len(clients), "rounds": ROUNDS,
               "runtime_s": round(time.time() - t_start, 1)}
    json.dump(summary, open(OUT / "fedavg_calce_summary.json", "w"), indent=1, default=float)
    print(res.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
