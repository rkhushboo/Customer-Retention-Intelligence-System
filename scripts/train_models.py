import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.data_loader import load_churn_data, build_preprocessor, preprocess_features
from utils.config import ML_MODEL_PATH, ANN_MODEL_PATH, PREPROCESSOR_PATH, ANN_HISTORY_PATH


def train_ml_model(X_train, y_train, X_test, y_test):
    models = {
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=31),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=200, learning_rate=0.05, max_depth=5)
    }
    best_model = None
    best_score = 0.0
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        print(f'{name} accuracy: {score:.4f}')
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def build_ann_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_ann(X_train, y_train, X_val, y_val):
    model = build_ann_model(X_train.shape[1])
    checkpoint = ModelCheckpoint(ANN_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=[checkpoint, early_stop],
        verbose=2
    )
    return model, history.history


def main():
    train_data = load_churn_data()
    y = train_data['Exited']
    X_raw = train_data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])

    preprocessor = build_preprocessor()
    X = preprocess_features(train_data, preprocessor, fit=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f'Saved preprocessor to {PREPROCESSOR_PATH}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print('Resampled training set:', np.bincount(y_train_res))

    best_ml = train_ml_model(X_train_res, y_train_res, X_test, y_test)
    joblib.dump(best_ml, ML_MODEL_PATH)
    print(f'Saved best ML model to {ML_MODEL_PATH}')

    num_val = int(X_train_res.shape[0] * 0.2)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_res, y_train_res, test_size=num_val, random_state=42, stratify=y_train_res)
    ann_model, history = train_ann(X_train_final, y_train_final, X_val, y_val)
    ann_model.save(ANN_MODEL_PATH, save_format='keras')
    with open(ANN_HISTORY_PATH, 'w', encoding='utf-8') as fp:
        json.dump(history, fp)
    print(f'Saved ANN model to {ANN_MODEL_PATH} and history to {ANN_HISTORY_PATH}')

if __name__ == '__main__':
    main()
