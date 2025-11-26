
# train_models.py
import os
from preprocess import load_raw, basic_feature_engineering, encode_and_scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "C:\\Users\\palak priya\\OneDrive\\Desktop\\fraud_upi\\data\\Variant III.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train(sample_frac=0.5):
    print("Loading raw data...")
    df = load_raw(DATA_PATH, sample_frac=sample_frac)
    print("Engineering features...")
    df = basic_feature_engineering(df)
    print("Encoding & scaling...")
    X, y, scaler, feature_cols, encoders = encode_and_scale(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Random Forest ----------
    print("Training RandomForest (fast config)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump((rf, scaler, feature_cols, encoders), os.path.join(MODELS_DIR, "fraud_model.pkl"))
    print("Saved RandomForest -> models/fraud_model.pkl")

    # ---------- Tiny Deep Learning model (optional) ----------
    print("Training tiny DL model (3 epochs)...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    model.fit(X_train, y_train, epochs=6, batch_size=128, validation_data=(X_test, y_test), callbacks=[es], verbose=1)
    model.save(os.path.join(MODELS_DIR, "fraud_model.h5"))
    print("Saved DL model -> models/fraud_model.h5")

    # also save a small sample test set for confusion matrix generation
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_data.pkl"))
    print("Training complete.")

if __name__ == "__main__":
    # sample_frac default 0.5 for speed; set to 1.0 to use full data
    train(sample_frac=0.5)
