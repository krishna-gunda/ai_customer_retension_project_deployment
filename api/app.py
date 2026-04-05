```python
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# =========================
# PATH SETUP (VERY IMPORTANT FOR VERCEL)
# =========================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "../Model.pkl")
scaler_path = os.path.join(BASE_DIR, "../standar_scaler.pkl")
template_path = os.path.join(BASE_DIR, "../templates")

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

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = []

            # =========================
            # HANDLE SIM COLUMN (ONE-HOT)
            # =========================
            sim_value = request.form.get("sim_column")

            sim_features = {
                'sim_column_BSNL': 1 if sim_value == "BSNL" else 0,
                'sim_column_Idea': 1 if sim_value == "Idea" else 0,
                'sim_column_Reliancejio': 1 if sim_value == "Reliancejio" else 0
            }

            # =========================
            # BUILD INPUT DATA
            # =========================
            for col in feature_columns:
                if col in sim_features:
                    input_data.append(sim_features[col])
                else:
                    value = request.form.get(col)

                    if value is None or value == "":
                        input_data.append(0)
                    else:
                        input_data.append(float(value))

            # =========================
            # DEBUG (VERY IMPORTANT)
            # =========================
            print("Feature count:", len(input_data))
            print("Input data:", input_data)

            # =========================
            # CONVERT TO NUMPY
            # =========================
            features_array = np.array([input_data])

            # =========================
            # SCALE
            # =========================
            features_scaled = scaler.transform(features_array)

            # =========================
            # PREDICT
            # =========================
            pred = model.predict(features_scaled)[0]

            if pred == 1:
                prediction = "Customer will Churn ❌"
            else:
                prediction = "Customer will Stay ✅"

        except Exception as e:
            print("ERROR:", str(e))
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


# =========================
# RUN (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    app.run(debug=True)
```
