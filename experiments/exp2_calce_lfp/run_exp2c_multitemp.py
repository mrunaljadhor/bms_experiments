"""
Experiment 2c: LSTM trained on all temperatures (0-50 °C), CALCE A123 LiFePO4
=============================================================================
Same architecture and leave-one-profile-out protocol as run_exp2.py, but the training set
contains the two training drive cycles at all seven temperatures 0, 10, 20, 25, 30, 40, 50 °C
(2 cells each). The test drive cycle is never seen at any temperature. -10 °C is excluded:
cell A1-008 delivered only 0.41-0.53 Ah there (about 1.05 Ah otherwise), i.e. an incomplete record.
Training windows use stride 2. Test windows are identical to run_exp2.py (stride 1).

Run:  python experiments/exp2_calce_lfp/run_exp2c_multitemp.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from run_exp2 import PROFILES, WINDOW, build, extract_runs, load_xlsx, metrics, windows

OUT = Path(__file__).resolve().parent
TEMPS = {"0": "20120813", "10": "20120815", "20": "20120817", "25": "20120827",
         "30": "20120820", "40": "20120822", "50": "20120824"}
BATCH, EPOCHS, PATIENCE, SEED, STRIDE = 64, 30, 5, 42, 2


def main():
    t_start = time.time()
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    preds25 = {}
    runs = {}
    for temp, date in TEMPS.items():
        for cell in ["A1-007", "A1-008"]:
            rel = f"DST-US06-FUDS-{temp}/{cell}-DST-US06-FUDS-{temp}-{date}.xlsx"
            for p, r in extract_runs(load_xlsx(rel)).items():
                runs[(temp, cell, p)] = r
    print(len(runs), "runs loaded")

    rows, errs25, history = [], [], {}
    for test_p in PROFILES:
        train_keys = [k for k in runs if k[2] != test_p]
        allX = np.concatenate([runs[k][["V", "I", "T"]].to_numpy(np.float32) for k in train_keys])
        scaler = (allX.mean(0), allX.std(0) + 1e-6)
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
        print(f"fold {test_p}: train {len(Xtr)} windows, val {len(Xva)}")
        model = build()
        t0 = time.time()
        h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH, verbose=2,
                      callbacks=[EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
        history[test_p] = {"epochs": len(h.history["loss"]), "train_s": round(time.time() - t0, 1)}
        for k in [k for k in runs if k[2] == test_p]:
            Xw, yw = windows(runs[k], scaler)
            pred = model.predict(Xw, batch_size=2048, verbose=0).ravel()
            err = pred - yw
            rows.append({"Test_profile": test_p, "Temp_C": int(k[0]), "Cell": k[1], **metrics(err)})
            if k[0] == "25":
                errs25.append(err)
                preds25[f"{k[1]}_{test_p}"] = pred
        print(history[test_p])

    np.savez(OUT / "multitemp_preds_25C.npz", **preds25)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "multitemp_per_run.csv", index=False)
    by_temp = per.groupby("Temp_C")[["MAE", "RMSE", "MaxErr"]].mean().round(3)
    by_temp.to_csv(OUT / "multitemp_by_temp.csv")
    s25 = metrics(np.concatenate(errs25))
    res = {"LSTM_multitemp_25C": {k: round(v, 3) for k, v in s25.items()},
           "mean_over_all_temps_MAE": round(float(per.MAE.mean()), 3),
           "history": history, "runtime_s": round(time.time() - t_start, 1)}
    json.dump(res, open(OUT / "multitemp_results.json", "w"), indent=1)
    print(by_temp.to_string())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
