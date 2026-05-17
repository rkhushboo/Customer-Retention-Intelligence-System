import json
from pathlib import Path
import joblib
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .config import ML_MODEL_PATH, ANN_MODEL_PATH, PREPROCESSOR_PATH, ANN_HISTORY_PATH, RISK_BANDS


def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f'Preprocessor not found at {PREPROCESSOR_PATH}')
    return joblib.load(PREPROCESSOR_PATH)


def load_ml_model():
    if not ML_MODEL_PATH.exists():
        raise FileNotFoundError(f'ML model not found at {ML_MODEL_PATH}')
    return joblib.load(ML_MODEL_PATH)


def load_ann_model():
    if not ANN_MODEL_PATH.exists():
        raise FileNotFoundError(f'ANN model not found at {ANN_MODEL_PATH}')
    return keras.models.load_model(ANN_MODEL_PATH)


def load_ann_history():
    if not ANN_HISTORY_PATH.exists():
        return None
    with open(ANN_HISTORY_PATH, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def predict_probability(model, data_array):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(data_array)[:, 1]
    else:
        proba = model.predict(data_array).ravel()
    return np.clip(proba, 0.0, 1.0)


def get_risk_level(score: float) -> dict:
    for low, high, label, color in RISK_BANDS:
        if low <= score < high:
            return {'label': label, 'color': color, 'score': score}
    return {'label': 'Unknown', 'color': '#6c757d', 'score': score}


def format_prediction_result(name: str, probability: float) -> dict:
    result = {
        'model_name': name,
        'probability': float(probability),
        'prediction': int(probability >= 0.5),
        'risk': get_risk_level(probability)
    }
    return result


def generate_prediction_report(input_data: dict, ml_result: dict, ann_result: dict) -> str:
    lines = [
        'Customer Retention Intelligence System',
        'Prediction Report',
        '=========================',
        '',
        'Input data:',
    ]
    for key, value in input_data.items():
        lines.append(f'  - {key}: {value}')

    lines += [
        '',
        'Machine Learning Model Prediction:',
        f"  - Churn Probability: {ml_result['probability']:.3f}",
        f"  - Predicted Outcome: {'Churn' if ml_result['prediction'] else 'Stay'}",
        f"  - Risk Level: {ml_result['risk']['label']}",
        '',
        'ANN Model Prediction:',
        f"  - Churn Probability: {ann_result['probability']:.3f}",
        f"  - Predicted Outcome: {'Churn' if ann_result['prediction'] else 'Stay'}",
        f"  - Risk Level: {ann_result['risk']['label']}",
        '',
        'Comparison Notes:',
    ]
    if ml_result['prediction'] == ann_result['prediction']:
        lines.append('  - Both models agree on the outcome. The prediction is strong.')
    else:
        lines.append('  - Predictions differ. This indicates model sensitivity to different feature patterns, so monitor customer signals closely.')
    return '\n'.join(lines)


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = predict_probability(model, X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob),
        'confusion_matrix': confusion_matrix(y, y_pred),
    }
    return metrics
