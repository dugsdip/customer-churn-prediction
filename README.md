# Customer Churn Prediction + Streamlit App

Predicts whether a telecom customer will leave, handles the class imbalance
(~26% churn), compares several models, tunes the best one, and serves it
through an interactive Streamlit app.

## Run it
Use a virtual environment so the project's packages stay separate from the rest of your system.

**Windows (PowerShell / CMD)**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Then, with `(.venv)` showing in your terminal:
```bash
pip install -r requirements.txt
python src/train.py        # trains, tunes, saves model/ + plots (about 1-2 min)
streamlit run app.py       # opens the dashboard in your browser
```

## Project structure
```
data/telco_churn.csv     Telco Customer Churn dataset (7,043 customers)
src/train.py             preprocessing, imbalance handling, tuning, evaluation
model/                   saved pipeline, metrics, plots
app.py                   Streamlit dashboard
```

## Approach
1. **Cleaning** - `TotalCharges` is text in the raw file (blank for brand-new customers); converted to numeric.
2. **Preprocessing** - scaling for numeric columns, one-hot encoding for categoricals, all inside a scikit-learn `Pipeline` so training and serving use identical steps.
3. **Class imbalance** - compared `class_weight`, SMOTE (applied inside the pipeline so it only touches training folds), and XGBoost `scale_pos_weight`.
4. **Metrics** - accuracy is misleading at 26% churn, so models are judged on ROC-AUC, recall, precision and F1.
5. **Tuning** - `RandomizedSearchCV` on XGBoost with 5-fold stratified CV.
6. **Threshold** - chosen from out-of-fold predictions on the training set only; the test set is never used for decisions.

## Results (held-out test set, 20%)
Fill in from `model/results.csv` after you run the training script.

## Ideas to extend
- Add SHAP for per-customer explanations
- Try LightGBM / CatBoost
- Add a cost-based threshold (cost of a retention offer vs. lost customer)
- Deploy on Streamlit Community Cloud and add the link here
