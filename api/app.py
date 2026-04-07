from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# =========================
# PATH SETUP (VERCEL-SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # one level up from api/

model_path = os.path.join(ROOT_DIR, "Model.pkl")
scaler_path = os.path.join(ROOT_DIR, "standar_scaler.pkl")
template_path = os.path.join(ROOT_DIR, "templates")

app = Flask(__name__, template_folder=template_path)

# =========================
# LOAD MODEL & SCALER
# =========================
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# =========================
# FEATURE ORDER (MUST MATCH TRAINING)
# =========================
feature_columns = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'sim_column_BSNL',
    'sim_column_Idea',
    'sim_column_Reliancejio',
    'Contract_od',
    'TotalCharges_var_trim',
    'MonthlyCharges_boxcox_trim',
    'tenure_scaled_trim'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = []
            sim_value = request.form.get("sim_column")
            sim_features = {
                'sim_column_BSNL': 1 if sim_value == "BSNL" else 0,
                'sim_column_Idea': 1 if sim_value == "Idea" else 0,
                'sim_column_Reliancejio': 1 if sim_value == "Reliancejio" else 0
            }

            for col in feature_columns:
                if col in sim_features:
                    input_data.append(sim_features[col])
                else:
                    value = request.form.get(col)
                    input_data.append(float(value) if value else 0)

            features_array = np.array([input_data])
            features_scaled = scaler.transform(features_array)
            pred = model.predict(features_scaled)[0]

            prediction = "Customer will Churn ❌" if pred == 1 else "Customer will Stay ✅"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

# Vercel entry point
handler = app
