# app/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CSV = os.path.join(DATA_DIR, 'Variant III.csv')
def get_csv_path():
    if os.path.exists(DEFAULT_CSV):
        return DEFAULT_CSV
    # Fallback: use any CSV in data/
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith('.csv'):
            return os.path.join(DATA_DIR, fname)
    raise FileNotFoundError('No CSV file found in data/.')

def load_raw(path=None, sample_frac=1.0):
    if path is None:
        path = get_csv_path()
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df

def basic_feature_engineering(df):
    df = df.copy()
    # Ensure target is integer
    df['fraud_bool'] = df['fraud_bool'].astype(int)

    # Create some engineered features often useful for fraud:
    # session intensity: session_length_in_minutes * velocity_6h
    if 'session_length_in_minutes' in df.columns and 'velocity_6h' in df.columns:
        df['session_intensity'] = df['session_length_in_minutes'].fillna(0) * df['velocity_6h'].fillna(0)

    # address stability: current vs prev
    if 'current_address_months_count' in df.columns and 'prev_address_months_count' in df.columns:
        df['address_stability'] = df['current_address_months_count'].fillna(0) - df['prev_address_months_count'].fillna(0)

    # flag suspicious small/large amounts
    if 'intended_balcon_amount' in df.columns:
        df['large_amount'] = (df['intended_balcon_amount'] > df['intended_balcon_amount'].quantile(0.99)).astype(int)
        df['small_amount'] = (df['intended_balcon_amount'] < df['intended_balcon_amount'].quantile(0.01)).astype(int)

    # month as cyclic features (if exists)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    return df

def encode_and_scale(df, target_col='fraud_bool'):
    df = df.copy()
    y = df[target_col].astype(int)
    # drop columns that are obviously IDs or text (none obvious here), keep all others
    # identify categorical object columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # We'll label-encode categorical columns (safe for tree models; DL will get numeric)
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    # drop target from features
    X = df.drop(columns=[target_col])

    # Keep numeric columns only (after encoding categorical columns are numeric)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    return pd.DataFrame(X_scaled, columns=numeric_cols), y.reset_index(drop=True), scaler, numeric_cols, encoders

if __name__ == "__main__":
    df = load_raw()
    df = basic_feature_engineering(df)
    X, y, scaler, cols, encs = encode_and_scale(df)
    print("Loaded and preprocessed. Features:", len(cols))
    print("Sample:\n", X.head())
