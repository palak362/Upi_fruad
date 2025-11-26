# app/app.py
import os
from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyo
from sklearn.metrics import confusion_matrix
import pandas as pd
from preprocess import load_raw, basic_feature_engineering, encode_and_scale

app = Flask(__name__, template_folder='templates', static_folder='static')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'models')
MODEL_PKL = os.path.join(MODELS_DIR, "fraud_model.pkl")
MODEL_H5 = os.path.join(MODELS_DIR, "fraud_model.h5")
TEST_PICKLE = os.path.join(MODELS_DIR, "test_data.pkl")

if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError("Run train_models.py first to create models in /models/")

# Load RF artifact (rf, scaler, feature_cols, encoders)
rf, scaler, feature_cols, encoders = joblib.load(MODEL_PKL)
dl_model = None
if os.path.exists(MODEL_H5):
    dl_model = load_model(MODEL_H5)

# prepare df sample for heatmap/confusion visuals
df_full = load_raw(sample_frac=0.3)
df_full = basic_feature_engineering(df_full)

# prepare test data saved earlier (if present)
test_data = None
if os.path.exists(TEST_PICKLE):
    X_test, y_test = joblib.load(TEST_PICKLE)
    test_data = (X_test, y_test)

# in-session probability list
session_probs = []

def preprocess_single(input_data):
    """
    Build a feature-row aligned with training feature_cols.
    We'll try:
      - If input field name matches an encoder -> use encoder
      - Else, try to map by common keys (amount -> intended_balcon_amount etc.)
    """
    # create a zero row for features
    row = {c: 0 for c in feature_cols}
    # map direct names
    for k, v in input_data.items():
        if k in row:
            try:
                row[k] = float(v)
            except:
                # categorical: label encode if seen during training
                if k in encoders:
                    try:
                        row[k] = int(encoders[k].transform([str(v)])[0])
                    except:
                        row[k] = 0
    # fallback heuristics
    if 'intended_balcon_amount' in row and 'amount' in input_data:
        row['intended_balcon_amount'] = float(input_data['amount'])

    # create df and scale
    df_row = pd.DataFrame([row], columns=feature_cols).fillna(0)
    X_scaled = scaler.transform(df_row)
    return X_scaled

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # collect fields from form
    amount = request.form.get("amount", type=float, default=0.0)
    payment_type = request.form.get("payment_type", default="")
    device_os = request.form.get("device_os", default="")
    employment_status = request.form.get("employment_status", default="")

    input_data = {
        "intended_balcon_amount": amount,
        "payment_type": payment_type,
        "device_os": device_os,
        "employment_status": employment_status
    }

    X_single = preprocess_single(input_data)
    rf_prob = rf.predict_proba(X_single)[0][1] if hasattr(rf, "predict_proba") else float(rf.predict(X_single)[0])
    dl_prob = float(dl_model.predict(X_single)[0][0]) if dl_model is not None else rf_prob
    prob = round(((rf_prob + dl_prob) / 2) * 100, 2)
    label = "⚠️ Fraudulent Transaction" if prob > 50 else "✅ Legit Transaction"

    # append to session list
    session_probs.append(prob)

    # Build Plotly line (session)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=list(range(1, len(session_probs)+1)),
        y=session_probs,
        mode='lines+markers',
        line=dict(color='#00d8ff'),
        name='Fraud Probability'
    ))
    fig_line.update_layout(title="Fraud Probability Over Session",
                           xaxis_title="Transaction #",
                           yaxis_title="Probability (%)",
                           plot_bgcolor='#111827', paper_bgcolor='#0f172a',
                           font=dict(color='#e0e0e0'), yaxis=dict(range=[0,100]))
    plot_div = pyo.plot(fig_line, output_type='div', include_plotlyjs=False)

    # Build heatmap: choose two categorical columns if exist: payment_type vs employment_status
    heatmap_div = None
    if 'payment_type' in df_full.columns and 'employment_status' in df_full.columns:
        pivot = df_full.pivot_table(index='payment_type', columns='employment_status', values='fraud_bool', aggfunc='sum').fillna(0)
        fig_hm = px.imshow(pivot, text_auto=True, color_continuous_scale='Reds',
                           labels=dict(x="Employment Status", y="Payment Type", color="Fraud Count"),
                           title="Heatmap: payment_type vs employment_status (fraud counts)")
        fig_hm.update_layout(plot_bgcolor='#111827', paper_bgcolor='#0f172a', font=dict(color='#e0e0e0'))
        heatmap_div = pyo.plot(fig_hm, output_type='div', include_plotlyjs=False)

    # Confusion matrix on saved test set (demo)
    cm_div = None
    if test_data is not None:
        X_test, y_test = test_data
        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        labels = ['Legit', 'Fraud']
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion Matrix (Demo)")
        fig_cm.update_layout(plot_bgcolor='#111827', paper_bgcolor='#0f172a', font=dict(color='#e0e0e0'))
        cm_div = pyo.plot(fig_cm, output_type='div', include_plotlyjs=False)

    return render_template("index.html", result=label, prob=prob, plot_div=plot_div, heatmap_div=heatmap_div, cm_div=cm_div)

if __name__ == "__main__":
    app.run(debug=True, port=5000)


