"""
Experiments 6, 7, 8 on the leak-free SOC LSTM from Experiment 1
==============================================================
6. Edge deployment: TFLite conversion (float32 / dynamic-range int8 / full int8), size, latency, accuracy.
7. SHAP on the LSTM: expected-gradients SHAP values (shap.GradientExplainer, with a manual fallback),
   attribution per feature and per time step.
8. Federated learning: the 6 real stand-test sessions are 6 clients. Leave-one-client-out:
   FedAvg (multi-round) vs centralized vs local-only fine-tuning from the synthetic-pretrained
   model, plus a communication-cost analysis.

Run:  python experiments/exp678_edge_shap_fl/run_exp678.py
"""

import json
import os
import subprocess
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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP1 = ROOT / "experiments" / "exp1_soc_lstm"
sys.path.insert(0, str(EXP1))
sys.path.insert(0, str(ROOT))
import run_exp1 as e1  # noqa: E402

RES = {}


def rebuild_data():
    syn = pd.read_csv(EXP1 / "synthetic_with_scenario_ids.csv")
    ids = np.random.RandomState(e1.SEED).permutation(syn["Scenario_ID"].unique())
    n = len(ids)
    tr_ids, te_ids = ids[:int(.7 * n)], ids[int(.85 * n):]
    tr, te = syn[syn.Scenario_ID.isin(tr_ids)], syn[syn.Scenario_ID.isin(te_ids)]
    scaler = MinMaxScaler().fit(tr[e1.FEATURES].to_numpy())
    Xtr, ytr, _ = e1.windows(tr, "Scenario_ID", scaler)
    Xte, yte, _ = e1.windows(te, "Scenario_ID", scaler)
    real = e1.load_real()
    Xr, yr, gr = e1.windows(real, "Session", scaler)
    return Xtr, ytr, Xte, yte, Xr, yr, gr, real


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# ============================================================================ Exp 6
def tflite_convert(model, mode, rep_data=None):
    fn = tf.function(lambda x: model(x, training=False))
    concrete = fn.get_concrete_function(tf.TensorSpec([1, e1.WINDOW, len(e1.FEATURES)], tf.float32))
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    if mode in ("dynamic_int8", "full_int8"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "full_int8":
        def rep():
            for i in range(0, len(rep_data), max(1, len(rep_data) // 300)):
                yield [rep_data[i:i + 1]]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
    return conv.convert()


def tflite_run(blob, X, threads=1):
    it = tf.lite.Interpreter(model_content=blob, num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = np.empty(len(X), np.float32)
    for i in range(len(X)):
        it.set_tensor(inp["index"], X[i:i + 1])
        it.invoke()
        preds[i] = it.get_tensor(out["index"]).ravel()[0]
    lat = []
    for _ in range(20):
        it.set_tensor(inp["index"], X[:1]); it.invoke()
    for k in range(500):
        it.set_tensor(inp["index"], X[k % len(X):k % len(X) + 1])
        s = time.perf_counter(); it.invoke(); lat.append((time.perf_counter() - s) * 1000)
    return preds, float(np.median(lat)), float(np.percentile(lat, 95))


def exp6(model, Xtr, Xte, yte, Xr, yr):
    out = OUT / "exp6_edge"; out.mkdir(exist_ok=True)
    rows = []
    keras_te = model.predict(Xte, verbose=0).ravel()
    keras_r = model.predict(Xr, verbose=0).ravel()
    one = Xte[:1]
    fn = tf.function(lambda x: model(x, training=False))
    for _ in range(10):
        model(one, training=False); fn(one)
    lat_eager, lat_graph = [], []
    for _ in range(200):
        s = time.perf_counter(); model(one, training=False); lat_eager.append((time.perf_counter() - s) * 1000)
        s = time.perf_counter(); fn(one); lat_graph.append((time.perf_counter() - s) * 1000)
    keras_size = (EXP1 / "soc_lstm_leakfree.keras").stat().st_size
    rows.append({"Variant": "Keras (eager)", "Size_KB": round(keras_size / 1024, 1),
                 "Latency_median_ms": round(np.median(lat_eager), 3), "Latency_p95_ms": round(np.percentile(lat_eager, 95), 3),
                 "MAE_synthetic_test_pct": round(100 * mae(keras_te, yte), 3), "MAE_real_zero_shot_pct": round(100 * mae(keras_r, yr), 3),
                 "Max_abs_diff_vs_Keras_pct": 0.0})
    rows.append({"Variant": "Keras (tf.function graph)", "Size_KB": round(keras_size / 1024, 1),
                 "Latency_median_ms": round(np.median(lat_graph), 3), "Latency_p95_ms": round(np.percentile(lat_graph, 95), 3),
                 "MAE_synthetic_test_pct": round(100 * mae(keras_te, yte), 3), "MAE_real_zero_shot_pct": round(100 * mae(keras_r, yr), 3),
                 "Max_abs_diff_vs_Keras_pct": 0.0})
    notes = {}
    for mode in ["float32", "dynamic_int8", "full_int8"]:
        path = out / f"soc_lstm_{mode}.tflite"
        if mode == "full_int8":
            # the full-int8 LSTM conversion can crash the process natively, so isolate it
            if path.exists():
                path.unlink()
            proc = subprocess.run([sys.executable, __file__, "--convert-full-int8"], capture_output=True, text=True)
            if proc.returncode != 0 or not path.exists():
                notes[mode] = f"conversion not supported for this LSTM (converter process exit code {proc.returncode})"
                continue
            blob = path.read_bytes()
        else:
            try:
                blob = tflite_convert(model, mode, Xtr)
            except Exception as ex:
                notes[mode] = f"conversion failed: {type(ex).__name__}: {str(ex)[:200]}"
                continue
            path.write_bytes(blob)
        p_te, l50, l95 = tflite_run(blob, Xte)
        p_r, _, _ = tflite_run(blob, Xr)
        rows.append({"Variant": f"TFLite {mode} (1 thread)", "Size_KB": round(len(blob) / 1024, 1),
                     "Latency_median_ms": round(l50, 3), "Latency_p95_ms": round(l95, 3),
                     "MAE_synthetic_test_pct": round(100 * mae(p_te, yte), 3), "MAE_real_zero_shot_pct": round(100 * mae(p_r, yr), 3),
                     "Max_abs_diff_vs_Keras_pct": round(100 * float(np.max(np.abs(p_te - keras_te))), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "edge_results.csv", index=False)
    RES["exp6"] = {"table": tab.to_dict(orient="records"), "notes": notes, "params": int(model.count_params()),
                   "hardware": "Intel i5-12500H laptop CPU, single-threaded TFLite interpreter"}
    print(tab.to_string(), notes)


# ============================================================================ Exp 7
def expected_gradients(model, X, background, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.zeros_like(X)
    for i in range(len(X)):
        idx = rng.integers(0, len(background), samples)
        ref = background[idx]
        alpha = rng.random((samples, 1, 1)).astype(np.float32)
        xi = tf.convert_to_tensor(ref + alpha * (X[i:i + 1] - ref))
        with tf.GradientTape() as tape:
            tape.watch(xi)
            y = model(xi, training=False)
        g = tape.gradient(y, xi).numpy()
        phi[i] = np.mean((X[i:i + 1] - ref) * g, axis=0)
    return phi


def exp7(model, Xtr, Xte, yte, Xr):
    out = OUT / "exp7_shap"; out.mkdir(exist_ok=True)
    rng = np.random.default_rng(e1.SEED)
    bg = Xtr[rng.choice(len(Xtr), 200, replace=False)]
    ex_syn = Xte[rng.choice(len(Xte), 300, replace=False)]
    ex_real = Xr[rng.choice(len(Xr), 200, replace=False)]
    method = "shap.GradientExplainer (expected gradients)"
    try:
        import shap
        explainer = shap.GradientExplainer(model, bg)
        sv_syn = np.asarray(explainer.shap_values(ex_syn, nsamples=200))
        sv_real = np.asarray(explainer.shap_values(ex_real, nsamples=200))
        sv_syn = sv_syn.reshape(ex_syn.shape)
        sv_real = sv_real.reshape(ex_real.shape)
        method += f", shap {shap.__version__}"
    except Exception as ex:
        method = f"manual expected gradients (shap failed: {type(ex).__name__}: {str(ex)[:120]})"
        sv_syn = expected_gradients(model, ex_syn, bg)
        sv_real = expected_gradients(model, ex_real, bg)

    base = float(model.predict(bg, verbose=0).mean())
    comp_err = float(np.mean(np.abs(sv_syn.sum(axis=(1, 2)) + base - model.predict(ex_syn, verbose=0).ravel())))
    feats = ["Voltage", "Current", "Temperature"]
    summary = {}
    for name, sv in [("synthetic", sv_syn), ("real", sv_real)]:
        per_feat = np.abs(sv).sum(axis=1).mean(axis=0) * 100          # SOC percentage points
        share = per_feat / per_feat.sum() * 100
        per_step = np.abs(sv).sum(axis=2).mean(axis=0)
        last10 = per_step[-10:].sum() / per_step.sum() * 100
        summary[name] = {"mean_abs_shap_pp": dict(zip(feats, np.round(per_feat, 3).tolist())),
                         "share_pct": dict(zip(feats, np.round(share, 1).tolist())),
                         "share_of_last_10_steps_pct": round(float(last10), 1)}
        pd.DataFrame({"step": np.arange(-e1.WINDOW + 1, 1), **{f: np.abs(sv[:, :, j]).mean(axis=0) * 100 for j, f in enumerate(feats)}}
                     ).to_csv(out / f"shap_per_step_{name}.csv", index=False)
    RES["exp7"] = {"method": method, "background": 200, "explained_synthetic": 300, "explained_real": 200,
                   "base_value_soc_pct": round(100 * base, 2), "completeness_mean_abs_err_pp": round(100 * comp_err, 3),
                   "summary": summary}

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1, 2]})
    xs = np.arange(len(feats))
    axes[0].bar(xs - 0.2, [summary["synthetic"]["share_pct"][f] for f in feats], 0.4, label="Synthetic test", color="#2563eb")
    axes[0].bar(xs + 0.2, [summary["real"]["share_pct"][f] for f in feats], 0.4, label="Real stand test", color="#f97316")
    axes[0].set_xticks(xs, feats, fontsize=8); axes[0].set_ylabel("Share of |SHAP| (%)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False); axes[0].tick_params(labelsize=7)
    steps = np.arange(-e1.WINDOW + 1, 1)
    for j, (f, c) in enumerate(zip(feats, ["#2563eb", "#16a34a", "#dc2626"])):
        axes[1].plot(steps, np.abs(sv_syn[:, :, j]).mean(axis=0) * 100, color=c, lw=1.3, label=f)
    axes[1].set_xlabel("Time step in window (0 = current)", fontsize=8)
    axes[1].set_ylabel("Mean |SHAP| (SOC pp)", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False); axes[1].tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_shap_lstm.png", dpi=300); plt.close(fig)
    print(json.dumps(RES["exp7"], indent=2))


# ============================================================================ Exp 8
def fit_on(model, X, y, epochs):
    model.compile(optimizer=Adam(1e-4), loss="mse")
    if len(X):
        model.fit(X, y, epochs=epochs, batch_size=e1.BATCH, verbose=0, shuffle=True)
    return model


def clone(model):
    m = tf.keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    return m


def exp8(model, Xr, yr, gr, real, rounds=10, local_epochs=1):
    out = OUT / "exp8_federated"; out.mkdir(exist_ok=True)
    clients = sorted(np.unique(gr))
    labels = {k: f"S{k + 1} {g['Cell_Config'].iloc[0]}-{g['Drive_Mode'].iloc[0]}" for k, g in real.groupby("Session")}
    rows, curves = [], []
    for test in clients:
        train_clients = [c for c in clients if c != test]
        Xt, yt = Xr[gr == test], yr[gr == test]
        e1.set_seed()
        # centralized: pooled data from the 5 training clients (rounds × local_epochs passes)
        cen = fit_on(clone(model), Xr[np.isin(gr, train_clients)], yr[np.isin(gr, train_clients)], rounds * local_epochs)
        mae_cen = 100 * mae(cen.predict(Xt, verbose=0).ravel(), yt)
        # local-only: each client fine-tunes alone; report the mean over the 5 local models
        loc = []
        for c in train_clients:
            e1.set_seed()
            m = fit_on(clone(model), Xr[gr == c], yr[gr == c], rounds * local_epochs)
            loc.append(100 * mae(m.predict(Xt, verbose=0).ravel(), yt))
        # FedAvg
        e1.set_seed()
        glob = clone(model)
        for r in range(1, rounds + 1):
            ws, ns = [], []
            for c in train_clients:
                m = clone(glob)
                fit_on(m, Xr[gr == c], yr[gr == c], local_epochs)
                ws.append(m.get_weights()); ns.append(int((gr == c).sum()))
            tot = sum(ns)
            glob.set_weights([sum(w[i] * n / tot for w, n in zip(ws, ns)) for i in range(len(ws[0]))])
            curves.append({"Test_client": labels[test], "Round": r,
                           "MAE_pct": 100 * mae(glob.predict(Xt, verbose=0).ravel(), yt)})
        mae_fed = curves[-1]["MAE_pct"]
        zero = 100 * mae(model.predict(Xt, verbose=0).ravel(), yt)
        rows.append({"Held_out_client": labels[test], "Zero_shot": round(zero, 3), "Local_only_mean": round(float(np.mean(loc)), 3),
                     "Local_only_best": round(float(np.min(loc)), 3), "Centralized": round(mae_cen, 3), "FedAvg": round(mae_fed, 3)})
        print(rows[-1])
    tab = pd.DataFrame(rows)
    n = np.array([(gr == c).sum() for c in clients])
    wavg = {col: round(float(np.average(tab[col], weights=n)), 3) for col in ["Zero_shot", "Local_only_mean", "Local_only_best", "Centralized", "FedAvg"]}
    tab.to_csv(out / "federated_loco.csv", index=False)
    cur = pd.DataFrame(curves); cur.to_csv(out / "federated_rounds.csv", index=False)

    # communication: per round each client downloads and uploads the full float32 model
    model_bytes = int(model.count_params()) * 4
    per_round = 2 * model_bytes * (len(clients) - 1)
    rows_per_client = real.groupby("Session").size().mean()
    raw_bytes_all = int(len(real) * (len(e1.FEATURES) + 1) * 4 * (len(clients) - 1) / len(clients))
    one_hz_day = 4 * 4 * 3600 * 24  # 4 float32 values per second for 24 h
    RES["exp8"] = {"setup": f"{len(clients)} clients (real stand-test sessions); leave-one-client-out; FedAvg {rounds} rounds × "
                            f"{local_epochs} local epoch(s), Adam 1e-4, sample-weighted averaging; centralized and local-only use "
                            f"{rounds * local_epochs} epochs; all start from the synthetic-pretrained Exp-1 model",
                   "weighted_mean_MAE_pct": wavg, "per_client": tab.to_dict(orient="records"),
                   "model_size_float32_KB": round(model_bytes / 1024, 1),
                   "traffic_per_round_5_clients_KB": round(per_round / 1024, 1),
                   "traffic_total_KB": round(per_round * rounds / 1024, 1),
                   "raw_data_upload_5_clients_KB": round(raw_bytes_all / 1024, 1),
                   "raw_1Hz_telemetry_per_vehicle_per_day_KB": round(one_hz_day / 1024, 1),
                   "break_even": "one round (download + upload ≈ 2 × model size) costs less than uploading raw 1 Hz V/I/T/SOC "
                                 f"telemetry once a vehicle has logged > {2 * model_bytes / (4 * 4) / 3600:.1f} h of data per round"}

    fig, ax = plt.subplots(figsize=(5, 2.9))
    for name, g in cur.groupby("Test_client"):
        ax.plot(g.Round, g.MAE_pct, lw=1, alpha=0.8, label=name)
    ax.axhline(wavg["Centralized"], color="k", ls="--", lw=1, label="Centralized (mean)")
    ax.set_xlabel("FedAvg round", fontsize=8); ax.set_ylabel("Held-out client MAE (% SOC)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_fedavg_rounds.png", dpi=300); plt.close(fig)
    print(tab.to_string()); print(wavg)


def main():
    t0 = time.time()
    model = tf.keras.models.load_model(EXP1 / "soc_lstm_leakfree.keras")
    Xtr, ytr, Xte, yte, Xr, yr, gr, real = rebuild_data()
    check = 100 * mae(model.predict(Xte, verbose=0).ravel(), yte)
    RES["sanity_synthetic_test_MAE_pct"] = round(check, 4)
    print("sanity synthetic MAE (expect 0.2034):", check)
    exp6(model, Xtr, Xte, yte, Xr, yr)
    exp7(model, Xtr, Xte, yte, Xr)
    exp8(model, Xr, yr, gr, real)
    RES["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(RES, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--convert-full-int8":
        m = tf.keras.models.load_model(EXP1 / "soc_lstm_leakfree.keras")
        Xtr_ = rebuild_data()[0]
        (OUT / "exp6_edge").mkdir(exist_ok=True)
        (OUT / "exp6_edge" / "soc_lstm_full_int8.tflite").write_bytes(tflite_convert(m, "full_int8", Xtr_))
    else:
        main()
