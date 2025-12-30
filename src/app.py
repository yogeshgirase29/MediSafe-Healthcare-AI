# app.py — Unified Flask App for AI Healthcare Portal
from flask import Flask, render_template, request, redirect
import os, pickle, joblib, numpy as np, pandas as pd
from datetime import datetime

app = Flask(__name__)

BASE_PATH = "models"

# ==========================================================
#  SECTION 1 — MEDICINE QUALITY MODEL
# ==========================================================
try:
    with open(os.path.join(BASE_PATH, "medicine_quality_model.pkl"), "rb") as f:
        quality_model = pickle.load(f)
    with open(os.path.join(BASE_PATH, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(BASE_PATH, "scaler2.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print("✅ Medicine Quality model loaded.")
except Exception as e:
    print(f"⚠️ Medicine Quality model load error: {e}")
    quality_model = encoders = scaler = None


# ==========================================================
#  SECTION 2 — DISEASE PREDICTION MODEL
# ==========================================================
def normalize_symptom_name(s):
    if s is None:
        return ""
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
    )


try:
    with open(os.path.join(BASE_PATH, "best_model.pkl"), "rb") as f:
        disease_model = pickle.load(f)
    with open(os.path.join(BASE_PATH, "label_encoder.pkl"), "rb") as f:
        disease_encoder = pickle.load(f)
    with open(os.path.join(BASE_PATH, "symptom_columns.pkl"), "rb") as f:
        symptom_cols = pickle.load(f)

    symptom_map = {normalize_symptom_name(c): i for i, c in enumerate(symptom_cols)}
    print("✅ Disease Prediction model loaded.")
except Exception as e:
    print(f"⚠️ Disease Prediction model load error: {e}")
    disease_model = disease_encoder = None
    symptom_cols = []
    symptom_map = {}


# ==========================================================
#  SECTION 3 — MEDICINE RECOMMENDATION MODEL
# ==========================================================
SYSTEMS = ["allopathy", "ayurveda", "homeopathy"]
med_models = {}

for sys in SYSTEMS:
    try:
        med_models[sys] = {
            "model": joblib.load(os.path.join(BASE_PATH, f"model_{sys}.pkl")),
            "preprocessor": joblib.load(os.path.join(BASE_PATH, f"preprocessor_{sys}.pkl")),
            "encoder": joblib.load(os.path.join(BASE_PATH, f"label_encoder_{sys}.pkl")),
            "lookup": joblib.load(os.path.join(BASE_PATH, f"lookup_{sys}.pkl")),
        }
        print(f"✅ Loaded {sys} model successfully.")
    except Exception as e:
        print(f"⚠️ Could not load {sys}: {e}")

# Default dropdown data
medicine_types = ["Allopathy", "Ayurveda", "Homeopathy"]
age_groups = ["Child", "Adult", "Elderly"]
disease_severity = ["Mild", "Moderate", "Severe"]

try:
    allopathy_lookup = med_models["allopathy"]["lookup"]
    disease_names = sorted(allopathy_lookup["Disease_Name"].unique().tolist())
except:
    disease_names = ['Acidity', 'Allergy', 'Anemia', 'Anxiety', 'Arthritis', 'Asthma',
       'Back Pain', 'Cold', 'Common Cold', 'Constipation', 'Cough',
       'Diabetes', 'Fever', 'Flu', 'Headache', 'Hypertension',
       'Indigestion', 'Joint Pain', 'Migraine', 'Obesity',
       'Urinary Tract Infection', 'Weak Immunity']


# ==========================================================
#  SECTION 4 — ROUTES
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")


# ------------------ Medicine Quality ----------------------
@app.route("/quality")
def quality_home():
    return render_template("quality_home.html")


@app.route("/quality_predict", methods=["POST"])
def detect():
    if request.method == 'POST':
        try:
            # Get user inputs
            active_ingredient = request.form['activeIngredient']
            # days_until_expiry = float(request.form['daysUntilExpiry'])
            
            expiry_date_str = request.form['expiryDate']
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            current_date = datetime.today().date()
            days_until_expiry = (expiry_date - current_date).days
            days_until_expiry = max(days_until_expiry, 0)  
            if (expiry_date - current_date).days < 0:
                return render_template('quality_home.html', warning="Warning: This medicine is already expired!")



            temperature = float(request.form['temperature'])
            warning_labels = int(request.form['warningLabels'])  # No encoding required
            dissolution = float(request.form['dissolution'])
            disintegration_time = float(request.form['disintegrationTime'])
            impurity_level = float(request.form['impurityLevel'])
            assay_purity = float(request.form['assayPurity'])

            # Apply Label Encoding (For Categorical Features)
            if 'Active Ingredient' in encoders:
                active_ingredient_encoded = encoders['Active Ingredient'].transform([active_ingredient])[0]
            else:
                active_ingredient_encoded = 0  # Default value if not in training data

            # Separate numerical features
            numerical_features = np.array([[days_until_expiry, temperature, 
                                            dissolution, disintegration_time, 
                                            impurity_level, assay_purity]])

            # Apply Scaling ONLY on Numerical Features
            numerical_features_scaled = scaler.transform(numerical_features)

            # Combine Encoded Categorical + Scaled Numerical Data in Correct Order
            input_data_final = np.hstack((
                [[active_ingredient_encoded]],  # 1️⃣ Active Ingredient (Encoded)
                numerical_features_scaled[:, :2],  # 2️⃣ Days Until Expiry, 3️⃣ Storage Temperature (Scaled)
                [[warning_labels]],  # 4️⃣ Warning Labels Present (As is)
                numerical_features_scaled[:, 2:]  # 5️⃣ Dissolution Rate, 6️⃣ Disintegration Time, 7️⃣ Impurity Level, 8️⃣ Assay Purity (Scaled)
            ))

            # print(input_data_final)
            # Make prediction
            output = quality_model.predict(input_data_final)

            # Convert prediction to readable text
            result = 'Safe' if output[0] == 1 else 'Not Safe'

            return render_template('result_quality.html', output=result)

        except Exception as e:
            return f"Error: {str(e)}"

    return redirect('/')


# ------------------ Disease Prediction --------------------
@app.route("/disease")
def disease_home():
    return render_template("disease_home.html", symptom_cols=symptom_cols)


@app.route("/disease_predict", methods=["POST"])
def disease_predict():
    if disease_model is None:
        return "Model not loaded. Please check server logs."

    try:
        # getlist to collect all symptom dropdowns
        selected_symptoms = request.form.getlist('symptoms')
        # print("Raw selected symptoms from form:", selected_symptoms)

        # Build input vector
        input_vec = np.zeros((1, len(symptom_cols)), dtype=int)

        matched_indices = []
        normalized_inputs = []

        for s in selected_symptoms:
            s_norm = normalize_symptom_name(s)
            normalized_inputs.append(s_norm)
            if s_norm in symptom_map:
                idx = symptom_map[s_norm]
                input_vec[0, idx] = 1
                matched_indices.append(idx)
            else:
                # try fallback: direct exact match (case-insensitive)
                key = str(s).strip().lower()
                if key in symptom_map:
                    idx = symptom_map[key]
                    input_vec[0, idx] = 1
                    matched_indices.append(idx)
                else:
                    # optionally log unseen symptom
                    print("Unmatched symptom (ignored):", s, "-> normalized:", s_norm)

        # If input vector is all zero, return friendly message or continue predict
        if input_vec.sum() == 0:
            # You can either return an error page or still attempt prediction (less meaningful)
            return render_template('result_disease.html', disease="No matching symptoms found. Please select valid symptoms from the list.", confidence=None, symptoms=selected_symptoms)

        # prediction
        pred_label = disease_model.predict(input_vec)[0]
        try:
            pred_disease = disease_encoder.inverse_transform([pred_label])[0]
        except Exception:
            pred_disease = str(pred_label)

        # optionally get probability
        confidence = None
        if hasattr(disease_model, "predict_proba"):
            probs = disease_model.predict_proba(input_vec)[0]
            confidence = round(float(max(probs)) * 100, 2)

        return render_template('result_disease.html', disease=pred_disease, confidence=confidence, symptoms=selected_symptoms)

    except Exception as e:
        print("Predict error:", e)
        return f"Error during prediction: {e}"

# ------------------ Medicine Recommendation ----------------
@app.route("/medicine")
def medicine_home():
    return render_template(
        "medicine_home.html",
        medicine_types=medicine_types,
        disease_names=disease_names,
        age_groups=age_groups,
        severities=disease_severity,
    )


@app.route("/medicine_predict", methods=["POST"])
def medicine_predict():
    try:
        # Get user inputs
        medicine_type = request.form["medicine_type"].lower()
        disease = request.form["disease_name"]
        age_group = request.form["age_group"]
        severity = request.form["severity"]

        # Validate selected type
        if medicine_type not in med_models:
            return render_template("result_medicine.html", prediction="⚠️ Invalid medicine system selected!")

        mdl = med_models[medicine_type]
        model = mdl["model"]
        encoder = mdl["encoder"]
        lookup = mdl["lookup"]

        # Prepare input DataFrame
        input_df = pd.DataFrame(
            [[disease, severity, age_group]],
            columns=["Disease_Name", "Disease_Severity", "Age_Group"]
        )

        # Predict encoded label
        pred_encoded = model.predict(input_df)[0]
        medicine_name = encoder.inverse_transform([int(pred_encoded)])[0]

        # Retrieve recommendation details
        rec = lookup[lookup["Medicine_Name"].str.lower() == medicine_name.lower()]
        if not rec.empty:
            details = {
                "Disease_Name": disease,
                "Medicine_Name": medicine_name,
                "Dosage_Form": rec["Dosage_Form"].values[0],
                "Recommended_Dosage": rec["Recommended_Dosage"].values[0],
                "Treatment_Duration": rec["Treatment_Duration"].values[0],
                "Precautions": rec["Precautions"].values[0],
                "Medicine_Approval_Status": rec["Medicine_Approval_Status"].values[0],
            }
        else:
            details = {
                "Disease_Name": disease,
                "Medicine_Name": medicine_name,
                "Dosage_Form": "N/A",
                "Recommended_Dosage": "N/A",
                "Treatment_Duration": "N/A",
                "Precautions": "N/A",
                "Medicine_Approval_Status": "N/A"
            }

        return render_template("result_medicine.html", details=details, system=medicine_type.capitalize())

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template("result_medicine.html", prediction=f"Error during prediction: {e}")

# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
