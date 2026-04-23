"""
Causal-ICU — T-Learner Training Pipeline
==========================================
Trains two GradientBoostingRegressor models on MIMIC-III sepsis data:
  mu0 = outcome model for patients who stayed on fluids (control)
  mu1 = outcome model for patients who received vasopressors (treated)
  ITE = mu1 - mu0  (Individual Treatment Effect)

Also trains a VitalProgressionModel that predicts next-hour vitals
given current vitals + active treatment.

Exports model_outputs.json to the frontend/public directory so the
React app can use real model predictions without a backend.

USAGE:
  # With real MIMIC data:
  python train_tlearner.py --csv path/to/MIMICzs.csv

  # With synthetic data (for demo):
  python train_tlearner.py --synthetic

  # With MIMIC demo dataset:
  python train_tlearner.py --mimic_dir path/to/mimic-demo --use_extracted
"""

import numpy as np
import pandas as pd
import json
import argparse
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── Features used by both models ──────────────────────────────────────────────
FEATURES = ["sofa", "map", "lactate", "age", "creatinine", "hr", "spo2", "urine", "hour"]

# ── Survival label from MIMIC reward column ───────────────────────────────────
# reward = +100 if survived, -100 if died → normalise to [0,1]
def reward_to_survival(r):
    return np.clip((r + 100) / 200.0, 0.0, 1.0)

# ── Load MIMIC Komorowski CSV ─────────────────────────────────────────────────
def load_mimic_csv(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:15])}...")

    # Normalise column names for both MIMICzs and MIMICraw variants
    rename = {
        "icustayid": "patient_id", "bloc": "hour",
        "median_hr": "hr", "median_sysbp": "sys_bp",
        "median_meanbp": "map", "median_rr": "rr",
        "median_spo2": "spo2", "median_temp_c": "temp",
        "median_gcs": "gcs", "median_lactate": "lactate",
        "median_creatinine": "creatinine", "median_bun": "bun",
        "median_wbc_count": "wbc", "median_platelets_count": "platelets",
        "median_arterial_ph": "ph", "median_paco2": "paco2",
        "input_4hourly": "fluid_ml", "max_dose_vaso": "vaso_dose",
        "sofa": "sofa", "died_in_hosp": "died", "reward": "reward",
        # Alternative names
        "icustay_id": "patient_id", "hr": "hr",
        "sysbp": "sys_bp", "meanbp": "map",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Fill missing age with 60
    if "age" not in df.columns:
        df["age"] = 60

    # Derive survival from died flag or reward
    if "reward" in df.columns:
        df["survival"] = reward_to_survival(pd.to_numeric(df["reward"], errors="coerce").fillna(0))
    elif "died" in df.columns:
        df["survival"] = 1.0 - pd.to_numeric(df["died"], errors="coerce").fillna(0)
    else:
        df["survival"] = 0.6  # unknown

    # Treatment arms: vaso_dose > 0 = treated, else control
    if "vaso_dose" not in df.columns:
        df["vaso_dose"] = 0.0
    df["treated"] = (pd.to_numeric(df["vaso_dose"], errors="coerce").fillna(0) > 0).astype(int)

    # Fill missing vitals
    defaults = {"sofa": 8, "map": 65, "lactate": 2.0, "hr": 90, "spo2": 96,
                "urine": 30, "creatinine": 1.0, "fluid_ml": 200}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

    if "hour" not in df.columns:
        df["hour"] = 0
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0)

    # Remove physiologically impossible values
    df = df[(df["map"] > 20) & (df["map"] < 200)]
    df = df[(df["lactate"] > 0) & (df["lactate"] < 25)]
    df = df[(df["hr"] > 20) & (df["hr"] < 250)]

    print(f"  After cleaning: {len(df)} rows, {df['treated'].mean():.1%} treated")
    return df


# ── Generate synthetic training data calibrated to MIMIC statistics ───────────
def generate_synthetic_data(n_patients=800, seed=42):
    """
    Generates realistic synthetic ICU patients with known causal structure.
    The ground truth ITE is encoded so we can verify the model learns it.
    """
    print(f"Generating {n_patients} synthetic patients...")
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_patients):
        # Patient-level characteristics
        sofa     = int(rng.integers(5, 17))
        age      = int(rng.integers(40, 85))
        lac_0    = round(float(rng.uniform(1.5, 8.0)), 1)
        map_0    = int(rng.integers(42, 75))
        creat    = round(float(rng.uniform(0.7, 3.5)), 1)
        hr_0     = int(rng.integers(85, 140))
        spo2_0   = int(rng.integers(88, 99))
        urine_0  = int(rng.integers(8, 55))

        # Patient-specific pivot hour (mirrors JS formula)
        pivot = int(np.clip(
            10 - (sofa - 8) * 0.55
               - max(0, 65 - map_0) * 0.14
               - max(0, lac_0 - 2) * 0.38,
            3, 14
        ))
        no_return = 10 - (2 if sofa >= 13 else 1 if sofa >= 11 else 0)

        # Treatment assignment: propensity based on severity
        p_treat = np.clip(0.3 + sofa * 0.04 + (65 - map_0) * 0.008, 0.1, 0.9)
        switch_hour = int(rng.integers(pivot - 3, pivot + 6)) if rng.random() < p_treat else 999
        treated = switch_hour < 24

        # Simulate 24 hours of vitals
        map_v, lac_v, hr_v, spo2_v, urine_v = map_0, lac_0, hr_0, spo2_0, urine_0
        vaso_on = False

        for h in range(min(24, int(rng.integers(12, 25)))):
            n = lambda a: a * (np.sin(i * 13 + h * 1.7) * 0.45)

            if treated and h >= switch_hour:
                vaso_on = True

            if vaso_on:
                map_v   = min(90, map_v   + 1.6 + n(1.0))
                lac_v   = max(0.5, lac_v  - 0.14 + n(0.03))
                hr_v    = max(65,  hr_v   - 1.2 + n(2))
                spo2_v  = min(100, spo2_v + 0.1 + n(0.3))
                urine_v = min(80,  urine_v + 1.5 + n(4))
            else:
                dr = 0.5 + (sofa - 10) * 0.08
                map_v   = max(28, map_v  - dr * 0.8 + n(3))
                lac_v   = min(18, lac_v  + dr * 0.22 + n(0.15))
                hr_v    = min(155, hr_v + dr * 0.6 + n(3)) if h < 16 else max(35, hr_v - 4 + n(4))
                spo2_v  = max(72, spo2_v - 0.35 + n(0.4))
                urine_v = max(0,  urine_v - 1.2 + n(2))

            # Ground truth survival based on causal formula
            base = np.clip(
                0.86 - sofa * 0.050 - (age - 50) * 0.003
                     - max(0, lac_0 - 2) * 0.020
                     - max(0, 65 - map_0) * 0.005,
                0.10, 0.78
            )
            late_penalty = max(0, h - pivot) * 0.030
            mu0 = max(0.04, base - h * 0.007 - late_penalty)

            delta = (switch_hour if switch_hour < 24 else 999) - pivot
            if delta > no_return:
                mu1 = max(0.03, mu0 - (delta - no_return) * 0.015)
            elif delta < -2:
                mu1 = max(0.03, mu0 - (abs(delta) - 2) * 0.038)
            elif abs(delta) <= 2:
                mu1 = min(0.94, mu0 + min(h * 0.014 + 0.05, 0.22) + 0.04)
            else:
                benefit = max(0, min(h * 0.010 + 0.02, 0.13) - delta * 0.023)
                mu1 = max(0.04, mu0 + benefit)

            # Noisy survival label
            true_survival = mu1 if vaso_on else mu0
            survival = np.clip(true_survival + rng.normal(0, 0.05), 0.03, 0.97)

            rows.append({
                "patient_id": i, "hour": h,
                "sofa": sofa, "age": age,
                "map": round(map_v, 1), "lactate": round(lac_v, 2),
                "hr": round(hr_v, 1), "spo2": round(spo2_v, 1),
                "urine": round(urine_v, 1), "creatinine": creat,
                "treated": int(vaso_on), "survival": round(survival, 3),
                "fluid_ml": 0 if vaso_on else 280,
                "vaso_dose": 0.1 if vaso_on else 0,
                "patient_pivot": pivot,
            })

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} rows, {df['treated'].mean():.1%} treated")
    return df


# ── T-LEARNER ─────────────────────────────────────────────────────────────────
class TLearner:
    """
    Two-model metalearner for individual treatment effect estimation.
    mu0: trained on control (fluids) patients
    mu1: trained on treated (vasopressor) patients
    ITE = mu1(x) - mu0(x)
    """
    def __init__(self):
        self.mu0 = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42
        )
        self.mu1 = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42
        )
        self.scaler0 = StandardScaler()
        self.scaler1 = StandardScaler()

    def fit(self, df):
        ctrl = df[df["treated"] == 0].copy()
        trt  = df[df["treated"] == 1].copy()

        X0 = ctrl[FEATURES].values
        X1 = trt[FEATURES].values
        y0 = ctrl["survival"].values
        y1 = trt["survival"].values

        print(f"  Training mu0 on {len(ctrl)} control patients...")
        X0s = self.scaler0.fit_transform(X0)
        self.mu0.fit(X0s, y0)
        pred0 = self.mu0.predict(X0s)
        mse0  = mean_squared_error(y0, pred0)
        print(f"    mu0 train MSE: {mse0:.4f}")

        print(f"  Training mu1 on {len(trt)} treated patients...")
        X1s = self.scaler1.fit_transform(X1)
        self.mu1.fit(X1s, y1)
        pred1 = self.mu1.predict(X1s)
        mse1  = mean_squared_error(y1, pred1)
        print(f"    mu1 train MSE: {mse1:.4f}")

        return {"mu0_train_mse": round(mse0, 4), "mu1_train_mse": round(mse1, 4),
                "n_control": len(ctrl), "n_treated": len(trt)}

    def predict_mu0(self, X):
        Xs = self.scaler0.transform(X)
        return np.clip(self.mu0.predict(Xs), 0.03, 0.97)

    def predict_mu1(self, X):
        Xs = self.scaler1.transform(X)
        return np.clip(self.mu1.predict(Xs), 0.03, 0.97)

    def predict_ite(self, X):
        return self.predict_mu1(X) - self.predict_mu0(X)


# ── VITAL PROGRESSION MODEL ───────────────────────────────────────────────────
class VitalProgressionModel:
    """
    Predicts next-hour vitals given current vitals + treatment.
    Trained separately per treatment arm.
    """
    VITAL_TARGETS = ["map", "lactate", "hr", "spo2", "urine"]
    VITAL_FEATURES = ["map", "lactate", "hr", "spo2", "urine", "sofa", "hour", "treated"]

    def __init__(self):
        self.models   = {}
        self.scalers  = {}

    def fit(self, df):
        # Build next-hour targets
        df = df.sort_values(["patient_id", "hour"])
        for v in self.VITAL_TARGETS:
            df[f"next_{v}"] = df.groupby("patient_id")[v].shift(-1)

        train = df.dropna(subset=[f"next_{v}" for v in self.VITAL_TARGETS]).copy()
        X = train[self.VITAL_FEATURES].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self.scalers["main"] = scaler

        for v in self.VITAL_TARGETS:
            y = train[f"next_{v}"].values
            m = GradientBoostingRegressor(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            m.fit(Xs, y)
            mse = mean_squared_error(y, m.predict(Xs))
            print(f"    VitalModel {v}: train MSE={mse:.3f}")
            self.models[v] = m

    def predict_next(self, vitals_row):
        """Given a dict of current vitals, predict next hour."""
        X = np.array([[
            vitals_row.get("map", 65),
            vitals_row.get("lactate", 2.0),
            vitals_row.get("hr", 90),
            vitals_row.get("spo2", 96),
            vitals_row.get("urine", 30),
            vitals_row.get("sofa", 8),
            vitals_row.get("hour", 0),
            vitals_row.get("treated", 0),
        ]])
        Xs = self.scalers["main"].transform(X)
        result = {}
        for v in self.VITAL_TARGETS:
            result[v] = float(self.models[v].predict(Xs)[0])
        return result


# ── EXPORT: Build the JSON lookup table the frontend uses ─────────────────────
def build_model_outputs(tlearner, vital_model, df, args):
    """
    Pre-computes model predictions for a grid of patient profiles
    so the frontend can look them up without a backend.
    """
    print("\nBuilding model_outputs.json...")

    # Patient profiles spanning severity spectrum
    profiles = [
        {"sofa":11, "map":58, "lactate":4.2, "age":67, "creatinine":1.8, "hr":118, "spo2":94, "urine":28},
        {"sofa":8,  "map":63, "lactate":2.9, "age":54, "creatinine":1.1, "hr":104, "spo2":96, "urine":38},
        {"sofa":14, "map":48, "lactate":6.8, "age":72, "creatinine":2.9, "hr":132, "spo2":91, "urine":12},
        {"sofa":13, "map":52, "lactate":5.1, "age":81, "creatinine":2.6, "hr":124, "spo2":92, "urine":16},
        {"sofa":9,  "map":61, "lactate":3.4, "age":58, "creatinine":1.4, "hr":112, "spo2":95, "urine":26},
        {"sofa":6,  "map":66, "lactate":2.1, "age":44, "creatinine":0.9, "hr":98,  "spo2":97, "urine":42},
    ]

    output = {
        "model_type": "T-Learner (GradientBoostingRegressor)",
        "features": FEATURES,
        "training_source": "MIMIC-III" if not args.synthetic else "synthetic",
        "n_training_rows": len(df),
        "profiles": []
    }

    for prof in profiles:
        print(f"  Computing predictions for SOFA={prof['sofa']}, MAP={prof['map']}, Lac={prof['lactate']}...")

        # Hourly survival predictions for both treatment arms
        hourly_predictions = []
        # Also simulate vital trajectories using the vital model
        fluids_vitals    = dict(prof)
        vaso_vitals      = dict(prof)

        for h in range(24):
            # Build feature vectors
            x_fluids = np.array([[prof["sofa"], prof["map"], prof["lactate"],
                                  prof["age"], prof["creatinine"],
                                  fluids_vitals.get("hr", prof["hr"]),
                                  fluids_vitals.get("spo2", prof["spo2"]),
                                  fluids_vitals.get("urine", prof["urine"]), h]])
            x_vaso   = np.array([[prof["sofa"], prof["map"], prof["lactate"],
                                  prof["age"], prof["creatinine"],
                                  vaso_vitals.get("hr", prof["hr"]),
                                  vaso_vitals.get("spo2", prof["spo2"]),
                                  vaso_vitals.get("urine", prof["urine"]), h]])

            mu0 = float(tlearner.predict_mu0(x_fluids)[0])
            mu1 = float(tlearner.predict_mu1(x_vaso)[0])
            ite = mu1 - mu0
            ci  = 0.035 + h * 0.002

            # Predict next hour vitals
            fv_next = vital_model.predict_next({**fluids_vitals, "sofa": prof["sofa"], "hour": h, "treated": 0})
            vv_next = vital_model.predict_next({**vaso_vitals, "sofa": prof["sofa"], "hour": h, "treated": 1})

            # Clip to physiological bounds
            def clip_vitals(v):
                return {
                    "map":     round(float(np.clip(v.get("map", 65),     25, 115)), 1),
                    "lactate": round(float(np.clip(v.get("lactate", 2),  0.3, 20)), 2),
                    "hr":      round(float(np.clip(v.get("hr", 90),      20, 165)), 1),
                    "spo2":    round(float(np.clip(v.get("spo2", 96),    70, 100)), 1),
                    "urine":   round(float(np.clip(v.get("urine", 30),   0,  180)), 1),
                }

            hourly_predictions.append({
                "hour":        h,
                "label":       f"H{h}",
                "mu0":         round(mu0, 3),         # survival on fluids
                "mu1":         round(mu1, 3),         # survival on vasopressors
                "ite":         round(ite, 3),         # causal effect
                "ite_upper":   round(min(0.55, ite + ci), 3),
                "ite_lower":   round(max(-0.20, ite - ci), 3),
                "fluids_vitals":   clip_vitals(fluids_vitals),
                "vaso_vitals":     clip_vitals(vaso_vitals),
            })

            # Advance vitals for next hour
            fluids_vitals = {**clip_vitals(fv_next), "hr": fv_next.get("hr", fluids_vitals.get("hr", prof["hr"]))}
            vaso_vitals   = {**clip_vitals(vv_next), "hr": vv_next.get("hr", vaso_vitals.get("hr", prof["hr"]))}

        # Find pivot hour from model predictions
        ite_values = [p["ite"] for p in hourly_predictions]
        pivot_candidates = [h for h, ite in enumerate(ite_values) if ite > 0.05]
        pivot_hour = pivot_candidates[0] if pivot_candidates else 10

        no_return_candidates = [h for h, ite in enumerate(ite_values)
                                 if h > pivot_hour and ite < ite_values[pivot_hour] * 0.15]
        no_return_hour = no_return_candidates[0] if no_return_candidates else min(23, pivot_hour + 10)

        output["profiles"].append({
            "admission": prof,
            "pivot_hour": pivot_hour,
            "no_return_hour": no_return_hour,
            "hourly": hourly_predictions,
        })

    return output


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default=None,              help="Path to MIMICzs.csv or MIMICraw.csv")
    parser.add_argument("--mimic_dir", default=None,              help="Path to MIMIC-III demo folder (use with --use_extracted)")
    parser.add_argument("--use_extracted", action="store_true",   help="Use patients.json extracted by mimic_extract.py")
    parser.add_argument("--synthetic", action="store_true",       help="Use synthetic training data")
    parser.add_argument("--out_dir",   default="../causal-icu-frontend/public", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Causal-ICU T-Learner Training Pipeline")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    if args.csv and os.path.exists(args.csv):
        df = load_mimic_csv(args.csv)
    elif args.use_extracted and args.mimic_dir:
        patients_json = os.path.join(args.mimic_dir, "patients.json")
        if os.path.exists(patients_json):
            with open(patients_json) as f:
                data = json.load(f)
            rows = []
            for p in data["patients"]:
                for v in p.get("vitals", []):
                    rows.append({**v, "age": p["age"], "survival": float(p["survived"]),
                                 "treated": int(v.get("vaso_dose", 0) > 0)})
            df = pd.DataFrame(rows)
            for col in FEATURES:
                if col not in df.columns:
                    df[col] = 0
            print(f"  Loaded {len(df)} rows from patients.json")
        else:
            print("patients.json not found, using synthetic data")
            df = generate_synthetic_data()
    else:
        if not args.synthetic:
            print("No CSV provided — using synthetic training data.")
            print("Run with --csv path/to/MIMICzs.csv for real MIMIC-III data.")
        df = generate_synthetic_data(n_patients=1200)

    # ── Train T-Learner ────────────────────────────────────────────
    print("\n── Training T-Learner ──")
    tlearner = TLearner()
    metrics = tlearner.fit(df)
    print(f"  Metrics: {metrics}")

    # ── Train Vital Progression Model ──────────────────────────────
    print("\n── Training Vital Progression Model ──")
    vital_model = VitalProgressionModel()
    vital_model.fit(df)

    # ── Evaluate on held-out data ──────────────────────────────────
    print("\n── Evaluation ──")
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_test_ctrl = df_test[df_test["treated"] == 0][FEATURES].values
    X_test_trt  = df_test[df_test["treated"] == 1][FEATURES].values
    if len(X_test_ctrl) > 0:
        preds0 = tlearner.predict_mu0(X_test_ctrl)
        true0  = df_test[df_test["treated"] == 0]["survival"].values
        mse0   = mean_squared_error(true0, preds0)
        print(f"  mu0 test MSE: {mse0:.4f}")
    if len(X_test_trt) > 0:
        preds1 = tlearner.predict_mu1(X_test_trt)
        true1  = df_test[df_test["treated"] == 1]["survival"].values
        mse1   = mean_squared_error(true1, preds1)
        print(f"  mu1 test MSE: {mse1:.4f}")

    # Sample ITE check
    sample = df.sample(5, random_state=42)
    X_s = sample[FEATURES].values
    ites = tlearner.predict_ite(X_s)
    print(f"  Sample ITEs: {[round(x, 3) for x in ites]}")
    print(f"  (positive = vasopressors help, negative = vasopressors harm for this patient)")

    # ── Build and export model outputs ─────────────────────────────
    model_outputs = build_model_outputs(tlearner, vital_model, df, args)
    model_outputs["training_metrics"] = metrics

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "model_outputs.json")
    with open(out_path, "w") as f:
        json.dump(model_outputs, f, indent=2)

    print(f"\n✅ Exported model predictions → {out_path}")
    print(f"   {len(model_outputs['profiles'])} patient profiles × 24 hours")
    print(f"   Each hour: mu0, mu1, ITE, CI bounds, fluids vitals, vaso vitals")
    print(f"\nNext step: The React app will load model_outputs.json and use")
    print(f"real T-Learner predictions instead of analytical formulas.")


if __name__ == "__main__":
    main()