import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import DATA_PATH, FEATURE_COLUMNS, GENDER_MAP, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_churn_data() -> pd.DataFrame:
    """Load the bank churn dataset from the root path."""
    df = pd.read_csv(DATA_PATH)
    return df


def feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean feature DataFrame for modeling and prediction."""
    working_df = df.copy()
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
    working_df = working_df.drop(columns=[c for c in drop_cols if c in working_df.columns], errors='ignore')
    working_df = working_df[FEATURE_COLUMNS].copy()
    working_df['Gender'] = working_df['Gender'].map(GENDER_MAP).fillna(0).astype(int)
    return working_df


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor


def preprocess_features(df: pd.DataFrame, preprocessor: ColumnTransformer, fit: bool = False):
    """Transform feature data with the preprocessing pipeline."""
    raw = feature_dataframe(df)
    if fit:
        return preprocessor.fit_transform(raw)
    return preprocessor.transform(raw)


def build_user_input(features: dict) -> pd.DataFrame:
    """Build a data frame from a prediction form input dictionary."""
    payload = {key: value for key, value in features.items()}
    payload_df = pd.DataFrame([payload])[FEATURE_COLUMNS].copy()
    payload_df['Gender'] = payload_df['Gender'].map(GENDER_MAP).fillna(0).astype(int)
    return payload_df
