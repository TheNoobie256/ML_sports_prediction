# 📈 ML Sports Predictor & EV Calculator

![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Build Status](https://github.com/YourUsername/YourRepoName/actions/workflows/retrain_models.yml/badge.svg)

An end-to-end Machine Learning Operations (MLOps) pipeline that calculates Expected Value (+EV) and Kelly Criterion bet sizing for live sports betting markets. The platform retrains its models monthly using automated GitHub Actions pipelines to prevent data drift and maximize Return on Investment (ROI).

---

## 🌟 Key Features

* **Live Market Scanning:** Real-time odds ingestion from global bookmakers integrated with injury reports via REST APIs.
* **XGBoost Predictive Engine:** Dual-sport architecture evaluating match outcomes using chronological rolling form (approximating Expected Goals / xG) and implied market probabilities.
* **Quantitative Risk Management:** Automatically calculates the optimal bankroll allocation using the Kelly Criterion.
* **Model Explainability:** Interactive feature-importance breakdown powered by XGBoost, offering transparency into how features (e.g., offensive form vs. injury differentials) drive market edges.
* **Historical Backtesting Engine:** Simulates model performance against past season holdout datasets to evaluate historical ROI and cumulative profit trajectories.
* **Automated MLOps Pipeline:** Monthly automated retraining schedule triggered via GitHub Actions, preventing data drift and syncing binary models (.joblib) back to production.

---

## 🏗️ Project Architecture

```text
├── .github/
│   └── workflows/
│       └── retrain_models.yml
├── app/
│   └── home.py
├── models/
│   ├── xgb_model.joblib
│   └── xgb_nba_model.joblib
├── src/
│   ├── data_ingestion.py
│   ├── model.py
│   ├── train.py
│   └── train_nba.py
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a .streamlit/secrets.toml file in the root folder and insert your free API keys:

```toml
ODDS_API_KEY = "your_odds_api_key_here"
API_FOOTBALL_KEY = "your_api_football_key_here"
```

### 4. Run the Streamlit Application
```bash
streamlit run app/home.py
```

---

## 🧠 Model Training & Retraining

To manually retrain the machine learning models locally:

```bash
# Retrain Master Soccer Model
python src/train.py

# Retrain Master NBA Model
python src/train_nba.py
```

### Automated MLOps
The project incorporates a scheduled GitHub Action workflow defined in .github/workflows/retrain_models.yml. Every 1st of the month at 03:00 AM UTC, a virtual environment automatically boots up, fetches updated multi-league match datasets, retrains the XGBoost estimators, and commits the updated binaries directly back to the main repository branch.

---

## ⚠️ Disclaimer
This project is built strictly for educational, analytical, and portfolio demonstration purposes. It does not constitute financial or sports betting advice. Always practice responsible wagering.