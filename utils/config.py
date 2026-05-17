from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / 'Bank_Customer_Churn.csv'
MODEL_DIR = ROOT_DIR / 'models'
ML_MODEL_PATH = MODEL_DIR / 'best_ml_model.pkl'
ANN_MODEL_PATH = MODEL_DIR / 'best_ann_model.keras'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.joblib'
ANN_HISTORY_PATH = MODEL_DIR / 'ann_history.json'
STYLE_PATH = ROOT_DIR / 'assets' / 'style.css'

FEATURE_COLUMNS = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]
NUMERIC_FEATURES = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]
CATEGORICAL_FEATURES = ['Geography']
GENDER_MAP = {'Female': 0, 'Male': 1}
GEOGRAPHY_OPTIONS = ['France', 'Spain', 'Germany']

RISK_BANDS = [
    (0.0, 0.35, 'Low Risk', '#0D6EFD'),
    (0.35, 0.65, 'Medium Risk', '#FFC107'),
    (0.65, 1.01, 'High Risk', '#DC3545')
]

PREDICTION_FIELDS = {
    'Credit Score': {'key': 'CreditScore', 'min': 300, 'max': 850, 'step': 1},
    'Geography': {'key': 'Geography', 'options': GEOGRAPHY_OPTIONS},
    'Gender': {'key': 'Gender', 'options': list(GENDER_MAP.keys())},
    'Age': {'key': 'Age', 'min': 18, 'max': 100, 'step': 1},
    'Tenure': {'key': 'Tenure', 'min': 0, 'max': 10, 'step': 1},
    'Balance': {'key': 'Balance', 'min': 0.0, 'max': 250000.0, 'step': 100.0},
    'Number of Products': {'key': 'NumOfProducts', 'min': 1, 'max': 4, 'step': 1},
    'Has Credit Card': {'key': 'HasCrCard', 'options': [0, 1]},
    'Is Active Member': {'key': 'IsActiveMember', 'options': [0, 1]},
    'Estimated Salary': {'key': 'EstimatedSalary', 'min': 10000.0, 'max': 200000.0, 'step': 500.0}
}
