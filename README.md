# iBMS Experimental Suite & Kinematics Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Companion: Frontend](https://img.shields.io/badge/Frontend-iBMS__frontend-purple.svg)](https://github.com/mrunaljadhor/iBMS_frontend)
[![Companion: Backend](https://img.shields.io/badge/Backend-bms__backend-darkblue.svg)](https://github.com/mrunaljadhor/bms_backend)

This repository contains the complete experimental benchmarking suite, physical validation scripts, uncertainty quantification pipelines, and edge optimization workflows for **iBMS (Intelligent Battery Management System)**.

All methodologies, experimental setups, and empirical findings correspond to **`iBMS_EndTerm_Report_v5.docx`** (available in [`report/iBMS_EndTerm_Report_v5.docx`](report/iBMS_EndTerm_Report_v5.docx)).

---

## 📖 Table of Contents
1. [Project Overview & System Specifications](#-project-overview--system-specifications)
2. [Ecosystem Repositories](#-ecosystem-repositories)
3. [Experimental Modules](#-experimental-modules)
   - [Exp 1: Leak-Free LSTM SOC Estimation](#exp-1-leak-free-lstm-state-of-charge-estimation)
   - [Exp 2: CALCE LFP Benchmark, Uncertainty Quantification & Risk-Aware AMSA](#exp-2-calce-lfp-benchmark--uncertainty-quantification-exp-10)
   - [Exp 3: Real AMSA Route Study on Indian Corridors](#exp-3-real-amsa-route-study-on-indian-corridors)
   - [Exp 4: One-At-A-Time DTE Sensitivity Analysis](#exp-4-one-at-a-time-dte-sensitivity-analysis)
   - [Exp 6–8: Edge Quantization, SHAP Explainability & Federated Learning](#exp-678-edge-deployment-shap-explainability--federated-learning)
   - [Kinematics Validation Suite](#kinematics-validation-suite)
4. [Repository Structure](#-repository-structure)
5. [Setup & Reproducibility Guide](#-setup--reproducibility-guide)
6. [Citation & Contact](#-citation--contact)

---

## ⚡ Project Overview & System Specifications

Conventional Electric Two-Wheeler (E2W) range estimators scale remaining State of Charge (SOC) by a static energy consumption rate, assuming uniform terrain and zero traffic. On hilly or congested corridors, flat-road models overestimate range by over **50–60%**, inducing unexpected vehicle stranding.

**iBMS** integrates:
- **Data-Driven Battery Tracking:** Recurrent LSTM models mapping raw voltage, current, and temperature vectors without requiring initial SOC.
- **Route-Physics Energy Model:** Dynamic Distance-to-Empty (DTE) accounting for road-load aerodynamics, SRTM 30 m elevation profiles, and real-time TomTom traffic congestion indices.
- **Adaptive Mode Switching Algorithm (AMSA):** A leg-wise predictive control framework enforcing ECO-Lock or mandating charging stops prior to trip commitment.
- **Uncertainty Quantification (UQ):** Monte Carlo Dropout and split-conformal calibration establishing statistical lower SOC bounds for conservative decision-making.

### Physical Vehicle & Pack Parameters (Lithium Iron Phosphate — LiFePO4)
| Parameter | Value | Source / Methodology |
| :--- | :--- | :--- |
| **Chemistry** | $\text{LiFePO}_4$ | Vehicle test bench |
| **Nominal Pack Voltage** | $63.5\text{ V}$ | Field test mean under load |
| **Full Charge Voltage** | $63.8\text{ V}$ | Vehicle telemetry |
| **Nominal Capacity** | $30.9\text{ Ah}$ ($28.9\text{ Ah}$ pooled) | Coulomb counting on field data |
| **Pack Energy** | $\approx 1.96\text{ kWh}$ | $30.9\text{ Ah} \times 63.5\text{ V}$ |
| **Rated Display Range** | $76\text{ km (ECO)} \mid 62\text{ km (SPORT)}$ | Digital cluster at $100\%$ SOC |
| **Auxiliary Load ($P_{\text{aux}}$)** | $40\text{ W}$ | Headlamps, ECU, instrumentation |
| **Vehicle + Rider Mass ($m$)** | $155\text{ kg}$ | $90\text{ kg}$ curb $+ 65\text{ kg}$ rider |

---

## 🔗 Ecosystem Repositories

The complete iBMS project is organized into three specialized repositories:

1. **[iBMS_frontend](https://github.com/mrunaljadhor/iBMS_frontend)**: Next.js responsive telemetry dashboard, live map visualization, and interactive intelligence suite.
2. **[bms_backend](https://github.com/mrunaljadhor/bms_backend)**: Node.js/Express telematics server implementing real-time SRTM elevation querying, TomTom traffic integration, and the three-tier AMSA decision loop.
3. **[iBMS_experiments](https://github.com/mrunaljadhor/iBMS_experiments)** *(This Repository)*: Offline experiments, training pipelines, empirical benchmarks, and validation suites.

---

## 🔬 Experimental Modules

### Exp 1: Leak-Free LSTM State-of-Charge Estimation
- **Script:** [`experiments/exp1_soc_lstm/run_exp1.py`](experiments/exp1_soc_lstm/run_exp1.py)
- **Problem Solved:** Prevents data leakage common in academic literature where past target SOC is fed into the input window.
- **Design:**
  - Strict input isolation: $X_t = [V_t, I_t, T_t]$ (SOC is excluded from inputs).
  - Sliding windows ($W = 50\text{ s}$) never cross scenario or session boundaries.
  - Scaler fit strictly on training partitions.
  - Leave-One-Session-Out (LOSO) cross-validation on real stationary vehicle tests.
- **Key Results:**
  - Synthetic held-out MAE: **$0.20\%$ SOC** ($R^2 > 0.999$).
  - Real vehicle sessions MAE: **$1.54\%$ SOC** (compared to $2.26\%$ for Coulomb counting with initial SOC error).

### Exp 2: CALCE LFP Benchmark & Uncertainty Quantification (Exp 10)
- **Directory:** [`experiments/exp2_calce_lfp/`](experiments/exp2_calce_lfp/)
- **Scripts:**
  - `run_exp2.py`: Baseline LSTM, EKF-1RC, Coulomb Counting, and constant models on public CALCE A123 26650 $\text{LiFePO}_4$ cells.
  - `run_exp2c_multitemp.py`: Multi-temperature generalization ($-10^\circ\text{C}$ to $50^\circ\text{C}$) on standard drive cycles (DST, US06, FUDS).
  - `run_exp10_uq.py`: Uncertainty quantification via Monte Carlo Dropout ($T=30$) and split-conformal prediction.
- **Key Results:**
  - Centralized Multi-temp LSTM MAE: **$3.17\%$ SOC** on unseen drive cycles across all temperatures.
  - Extended Kalman Filter (EKF-1RC with measured OCV): **$0.96\%$ MAE**.
  - **Risk-Aware AMSA:** Utilizing the conformal lower bound ($SOC_{\text{decision}} = \mu - q_1 \cdot \sigma$) slashes unsafe route acceptance rates from **$0.71\%$** down to **$0.007\%$**, trading off a modest increase in conservative refusals ($1.59\% \rightarrow 8.15\%$).

### Exp 3: Real AMSA Route Study on Indian Corridors
- **Script:** [`experiments/exp3_amsa_routes/run_exp3.py`](experiments/exp3_amsa_routes/run_exp3.py)
- **Corridors Evaluated:** 8 route directions encompassing urban and extreme mountain terrain:
  1. *New Delhi $\leftrightarrow$ Noida* (Flat urban)
  2. *Dehradun $\leftrightarrow$ Mussoorie* (Sub-Himalayan steep climb: $+1,400\text{ m}$)
  3. *Shimla $\leftrightarrow$ Kufri* (High-altitude climb: $+450\text{ m}$)
  4. *Siliguri $\leftrightarrow$ Darjeeling* (Eastern Himalayan ascent: $+2,000\text{ m}$)
- **Key Findings:**
  - Conventional flat-road range estimators overestimate uphill range by **$53\%\text{--}60\%$**.
  - Flat estimators approve a $60\text{ km}$ mountain ascent (Siliguri $\rightarrow$ Darjeeling) that drains $180\%$ of battery capacity; AMSA discards this route as **IMPOSSIBLE** before departure.

### Exp 4: One-At-A-Time DTE Sensitivity Analysis
- **Script:** [`experiments/exp4_dte_sensitivity/run_exp4.py`](experiments/exp4_dte_sensitivity/run_exp4.py)
- **Analysis:** Evaluates partial derivatives of range with respect to road-load variables (uphill/downhill grade, aerodynamic cruising speed, traffic jam index $J$, and temperature delta).
- **Output:** Tornado diagram ([`fig_dte_tornado.png`](experiments/exp4_dte_sensitivity/fig_dte_tornado.png)) highlighting that elevation grade and traffic congestion exhibit the highest elasticity on remaining range.

### Exp 6/7/8: Edge Deployment, SHAP Explainability & Federated Learning
- **Script:** [`experiments/exp678_edge_shap_fl/run_exp678.py`](experiments/exp678_edge_shap_fl/run_exp678.py)
- **Edge Deployment (Exp 6):**
  - Full integer (`int8`) quantization compresses model footprint to **$67.6\text{ KiB}$** ($82.9\%$ reduction).
  - CPU inference latency: **$0.41\text{ ms}$** per window (enabling $> 2\text{ kHz}$ throughput, far exceeding $1\text{ Hz}$ BMS constraints).
- **SHAP Explainability (Exp 7):**
  - GradientExplainer attribution confirms the model relies primarily on recent terminal voltage trajectories ($68.4\%$ attribution), adhering to physical electrochemical cell principles.
- **Federated Learning (Exp 8):**
  - FedAvg evaluated across distributed clients: reaches $8.99\%$ MAE vs $3.17\%$ centralized, demonstrating that on small edge fleets, centralized calibration with local conformal bounding is superior.

### Kinematics Validation Suite
- **Script:** [`kinematics_validation/validate_kinematics.py`](kinematics_validation/validate_kinematics.py)
- **Validation Focus:**
  - Evaluates zero-cost kinematic synthesis from raw BMS current vectors when physical wheel encoders or GPS speed logs are unavailable.
  - Throttle proxy $\theta(I) = \text{clip}((I_{\text{load}} - I_{\text{min}}) / (I_{99} - I_{\text{min}}), 0, 1)$.
  - Exponential Moving Average (EMA) smoothing ($\alpha = 0.15$) ensures acceleration complies with physical two-wheeler limits ($[-7.0, +4.0]\text{ m/s}^2$).
  - Energy consistency confirms the synthetic drive cycle produces realistic consumption ($24.8\text{ Wh/km}$ ECO, $31.2\text{ Wh/km}$ SPORT).

---

## 📁 Repository Structure

```text
iBMS_experiments/
├── experiments/
│   ├── exp1_soc_lstm/               # Leak-free LSTM vs Coulomb vs EKF
│   │   ├── run_exp1.py
│   │   ├── metrics.json
│   │   ├── real_sessions_summary.csv
│   │   └── fig_real_sessions.png
│   ├── exp2_calce_lfp/              # CALCE benchmark & Conformal UQ
│   │   ├── data/                    # CALCE A123 drive cycle spreadsheets
│   │   ├── run_exp2.py              # Single-temp benchmark
│   │   ├── run_exp2c_multitemp.py   # Multi-temp (-10°C to 50°C)
│   │   ├── run_exp10_uq.py          # MC Dropout + Conformal calibration
│   │   ├── run_exp8b_fedavg_calce.py
│   │   └── uq_summary.json
│   ├── exp3_amsa_routes/            # AMSA 8-route topological study
│   │   ├── run_exp3.py
│   │   ├── routes_summary.csv
│   │   ├── naive_vs_amsa.csv
│   │   └── fig_amsa_routes.png
│   ├── exp4_dte_sensitivity/        # DTE parameter sensitivity
│   │   ├── run_exp4.py
│   │   ├── sensitivity_table.csv
│   │   └── fig_dte_tornado.png
│   └── exp678_edge_shap_fl/         # TFLite int8, SHAP & FL
│       ├── run_exp678.py
│       └── results.json
├── kinematics_validation/           # Zero-cost kinematic synthesis
│   ├── validate_kinematics.py
│   ├── plausibility.csv
│   ├── energy_consistency.csv
│   └── fig_speed_trace.png
├── report/                          # Reference Academic Documentation
│   ├── iBMS_EndTerm_Report_v5.docx  # Full project report document
│   └── report_v5_sources/           # Modular chapter text files
├── synthetic_data_generator.py      # Synthetic discharge generator
├── trinity_ev_dataset.csv           # Field test telemetry
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Reproducibility Guide

### 1. Prerequisites & Environment Setup
Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/mrunaljadhor/iBMS_experiments.git
cd iBMS_experiments

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Individual Experiments

#### Run Exp 1: Leak-Free LSTM & Baselines
```bash
python experiments/exp1_soc_lstm/run_exp1.py
```
*Outputs: Generates `metrics.json`, per-session CSVs, and comparison plots in `experiments/exp1_soc_lstm/`.*

#### Run Exp 2 & Exp 10: CALCE Benchmark and Uncertainty Quantification
```bash
# Standard single-temperature benchmark
python experiments/exp2_calce_lfp/run_exp2.py

# Multi-temperature training and evaluation
python experiments/exp2_calce_lfp/run_exp2c_multitemp.py

# MC Dropout and split-conformal risk-aware evaluation (Exp 10)
python experiments/exp2_calce_lfp/run_exp10_uq.py
```

#### Run Exp 3: AMSA Real Route Study
*Note: Ensure backend server is active or run against standalone calculation mode.*
```bash
python experiments/exp3_amsa_routes/run_exp3.py
```

#### Run Exp 4: DTE Sensitivity Analysis
```bash
python experiments/exp4_dte_sensitivity/run_exp4.py
```

#### Run Exp 6, 7 & 8: TFLite Quantization, SHAP & FL
```bash
python experiments/exp678_edge_shap_fl/run_exp678.py
```

#### Run Kinematics Validation
```bash
python kinematics_validation/validate_kinematics.py
```

---

## 📄 Citation & References

If you refer to this implementation or experimental data in academic work, please cite:

```bibtex
@misc{jadhor2026ibms,
  title={iBMS: Intelligent Battery Management System with Topological Route-Energy Telematics, Conformal Uncertainty Quantification, and Adaptive Drive Mode Switching},
  author={Jadhor, Mrunal and Project Team},
  year={2026},
  note={Department of Electrical and Electronics Engineering, MIT World Peace University, Pune}
}
```

Primary project report: [`report/iBMS_EndTerm_Report_v5.docx`](report/iBMS_EndTerm_Report_v5.docx).
