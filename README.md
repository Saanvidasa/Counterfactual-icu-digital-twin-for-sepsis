# 🏥 Counterfactual ICU Digital Twin for Sepsis

> An interactive clinical decision-support dashboard that simulates **"what-if" treatment scenarios** for ICU sepsis patients — powered by real MIMIC-III patient data.

---

## 📌 Overview

Sepsis is a life-threatening emergency where every treatment decision matters. This project builds a **digital twin of ICU sepsis patients** by extracting real clinical data from the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) database and rendering it in an interactive React dashboard.

Clinicians and researchers can explore **counterfactual scenarios** — asking questions like:
- *"What would have happened if we started vasopressors 2 hours earlier?"*
- *"How would fluid resuscitation at a higher rate have changed the patient's MAP and lactate?"*

This makes it a powerful tool for retrospective analysis, clinical training, and AI-assisted treatment planning.

---

## ✨ Features

- 🔬 **Real MIMIC-III data** — extracts vitals, labs, and interventions from actual ICU records
- 📊 **Hour-by-hour patient timeseries** — MAP, heart rate, SpO₂, lactate, creatinine, urine output, and more
- 💉 **Intervention tracking** — vasopressor dosing and IV fluid administration
- 🧮 **Approximate SOFA scoring** — automated severity scoring from available labs
- 🔄 **Counterfactual simulation** — interactive "what-if" treatment exploration
- 🎯 **Smart patient selection** — automatically identifies the most clinically interesting sepsis cases
- ⚡ **Fast React frontend** — built with Vite for instant hot-reload development

---

## 🗂️ Project Structure

```
├── src/                    # React frontend source
│   └── ...                 # Dashboard components, charts, simulation logic
├── public/
│   └── patients.json       # Extracted MIMIC-III patient data (generated)
├── mimic_extract.py        # Python ETL pipeline for MIMIC-III data
├── index.html
├── vite.config.js
└── package.json
```

---

## 🧬 Data Pipeline — `mimic_extract.py`

The Python script reads raw MIMIC-III CSVs and builds a `patients.json` file consumed by the frontend.

### What it extracts

| Category | Variables |
|---|---|
| **Vitals** | MAP, Heart Rate, Respiratory Rate, SpO₂, Temperature, GCS, Systolic BP |
| **Labs** | Lactate, Creatinine, BUN, WBC, Platelets, pH, Bicarbonate, Glucose, PaCO₂ |
| **Outputs** | Urine output (hourly) |
| **Interventions** | Vasopressors (norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine), IV Fluids |

### Required MIMIC-III files

```
ICUSTAYS.csv
CHARTEVENTS.csv
LABEVENTS.csv
OUTPUTEVENTS.csv
INPUTEVENTS_MV.csv
PATIENTS.csv
ADMISSIONS.csv
```

### Running the extractor

```bash
python3 mimic_extract.py \
  --mimic_dir /path/to/mimic-iii-clinical-database-demo-1.4 \
  --n 6 \
  --out ./public/patients.json
```

| Argument | Default | Description |
|---|---|---|
| `--mimic_dir` | `.` | Path to your MIMIC-III folder |
| `--n` | `6` | Number of patients to export |
| `--out` | `../causal-icu-frontend/public/patients.json` | Output path for JSON |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+
- **Python** 3.8+
- Access to the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) (requires PhysioNet credentialing) or the free [MIMIC-III Demo](https://physionet.org/content/mimiciii-demo/)

### 1. Clone the repository

```bash
git clone https://github.com/NavyaS26/Counterfactual-icu-digital-twin-for-sepsis.git
cd Counterfactual-icu-digital-twin-for-sepsis
```

### 2. Install Python dependencies

```bash
pip install pandas numpy
```

### 3. Extract patient data

```bash
python3 mimic_extract.py --mimic_dir /path/to/mimic-iii --n 6
```

### 4. Install and run the frontend

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🩺 How Patient Selection Works

The pipeline automatically identifies the most clinically interesting sepsis patients by scoring each ICU stay on:

- **Low mean arterial pressure (MAP < 70 mmHg)** — a key septic shock criterion
- **High lactate** — indicator of tissue hypoperfusion

Patients are ranked by severity, and a spread of cases (severe → moderate → mild) is selected for the dashboard to give a representative sample.

---

## 📐 SOFA Score Approximation

When full SOFA data isn't available, the system computes an approximate score from available labs:

| Finding | Points |
|---|---|
| MAP < 70 mmHg | +2 |
| Creatinine > 1.2 mg/dL | +2 |
| Platelets < 100 × 10³/µL | +2 |
| pH < 7.3 | +2 |

Diagnosis labels are assigned as: **Sepsis** (SOFA < 8), **Severe Sepsis** (8–10), **Septic Shock** (≥ 11).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite |
| Charts / Visualization | (within `src/`) |
| Data pipeline | Python, Pandas, NumPy |
| Data source | MIMIC-III Clinical Database |
| Linting | ESLint |

---

## ⚠️ Data Access & Ethics

This project uses the **MIMIC-III Clinical Database**, which requires:

1. Completion of a recognized human subjects research training course (e.g. CITI)
2. Signing the PhysioNet data use agreement

> ⚠️ **Do not commit raw MIMIC-III CSV files or any patient data to this repository.** The `patients.json` output is derived and de-identified, but should still be handled responsibly.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the counterfactual simulation logic, add new vitals, or enhance the frontend:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is open source. Please ensure any use of MIMIC-III data complies with the [PhysioNet Data Use Agreement](https://physionet.org/content/mimiciii/view-license/1.4/).

---

## 🙏 Acknowledgements

- [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) — Johnson et al., PhysioNet
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) — frontend tooling
- The critical care and clinical AI community for inspiring this work
