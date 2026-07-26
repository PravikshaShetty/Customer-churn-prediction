# Customer Churn Prediction

🔗 **[Live Demo](https://churn-predictor-praviksha.streamlit.app/)**

Predicts whether a telecom customer is likely to churn, using the IBM/Kaggle
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## What this project does

- Cleans and preprocesses the raw Telco churn dataset (handles missing
  `TotalCharges`, encodes categorical features, scales numeric features)
- Trains and compares three models:
  - Logistic Regression
  - Random Forest (tuned with `GridSearchCV`)
  - XGBoost
- Evaluates models with ROC-AUC, precision, recall, F1, and confusion matrices
- Explores the precision/recall tradeoff across classification thresholds
- Uses SHAP to explain which features drive churn predictions
- Saves trained models (`.pkl`) for reuse in a Streamlit app

## Project structure

```
.
├── Customer_Churn_Prediction.ipynb   # main analysis & modeling notebook
├── app.py                            # Streamlit app for interactive churn scoring
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── logistic_model.pkl                # trained models (used by app.py)
├── rf_model.pkl
├── xgb_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
git clone https://github.com/PravikshaShetty/Customer-churn-prediction.git
cd Customer-churn-prediction
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```

The trained model files (`logistic_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`)
are already included in this repo, so the app works immediately without
running the notebook first. Re-running the notebook will regenerate and
overwrite them if you want to retrain from scratch.

## Results

| Model               | ROC AUC | F1   | Precision | Recall |
|---------------------|---------|------|-----------|--------|
| Logistic Regression | 0.85    | 0.64 | 0.52      | 0.82   |
| Random Forest       | 0.86    | 0.65 | 0.56      | 0.77   |
| XGBoost             | 0.86    | 0.65 | 0.55      | 0.78   |

Recall is prioritized over precision, since for a churn-prevention use case
missing an at-risk customer (false negative) is more costly than flagging a
customer who wasn't actually going to churn (false positive).

## Running the Streamlit app

The notebook saves `logistic_model.pkl`, `rf_model.pkl`, and `xgb_model.pkl`
to the project root. Once those exist, launch the interactive app:

```bash
streamlit run app.py
```

This opens a browser page where you can enter a customer's details (contract
type, tenure, charges, services subscribed, etc.), pick which trained model
to use, and get a churn probability with a clear likely-to-churn /
likely-to-stay verdict.

## Next steps

- Add SHAP-based per-customer explanations to the app UI
