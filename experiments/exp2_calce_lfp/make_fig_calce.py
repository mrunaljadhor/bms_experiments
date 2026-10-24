"""
Fig. 3 of the paper: SOC on the unseen FUDS cycle (CALCE A123 cell A1-008, 25 °C).
Uses the multi-temperature LSTM predictions (run_exp2c_multitemp.py) and the 1-RC EKF
(run_exp2b_ekf.py parameters for the FUDS fold). Output goes to paper_figures/.

Run:  python experiments/exp2_calce_lfp/make_fig_calce.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_exp2 import CELLS, WINDOW, coulomb, extract_runs, load_xlsx, ocv_curve
from run_exp2b_ekf import ekf_rc

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "paper_figures"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})


def main():
    run = extract_runs(load_xlsx(CELLS["A1-008"]))["FUDS"]
    og, ov = ocv_curve()
    p = json.load(open(HERE / "ekf1rc_fits.json"))["FUDS"]
    lstm = np.load(HERE / "multitemp_preds_25C.npz")["A1-008_FUDS"]
    sl = slice(WINDOW - 1, None)
    t = run.t.to_numpy()[sl] / 60
    y = run.SOC.to_numpy()[sl]
    ekf = ekf_rc(run, 90.0, p, og, ov, p["q"], p["r"])[sl]
    cc = coulomb(run, 90.0)[sl]

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.plot(t, y, color="#111827", lw=1.3, label="Reference SOC")
    ax.plot(t, lstm, color="#2563eb", lw=0.8, label=f"LSTM, no SOC$_0$ (MAE {np.abs(lstm - y).mean():.2f} %)")
    ax.plot(t, ekf, color="#16a34a", lw=0.9, ls="--",
            label=f"EKF-1RC, SOC$_0$ −10 % (MAE {np.abs(ekf - y).mean():.2f} %)")
    ax.plot(t, cc, color="#dc2626", lw=0.9, ls=":",
            label=f"Coulomb, SOC$_0$ −10 % (MAE {np.abs(cc - y).mean():.2f} %)")
    ax.set_xlabel("Time (min)", fontsize=7)
    ax.set_ylabel("SOC (%)", fontsize=7)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig2_calce_fuds.png", dpi=600)
    fig.savefig(FIG / "fig2_calce_fuds.pdf")
    plt.close(fig)
    print("saved", FIG / "fig2_calce_fuds.pdf")


if __name__ == "__main__":
    main()
