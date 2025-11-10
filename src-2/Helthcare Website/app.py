import numpy as np
from flask import Flask, render_template, request, redirect
import pickle
import os
from datetime import datetime


# Load model, encoders, and scaler
model_file = os.path.join(os.path.dirname(__file__), 'medicine_quality_model.pkl')
encoder_file = os.path.join(os.path.dirname(__file__), 'encoders.pkl')
scaler_file = os.path.join(os.path.dirname(__file__), 'scaler2.pkl')

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(encoder_file, 'rb') as f:
    encoders = pickle.load(f)  # Load encoders (for categorical features)

with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)  # Load scaler (for numerical features)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
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
                return render_template('index.html', warning="Warning: This medicine is already expired!")



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
            output = model.predict(input_data_final)

            # Convert prediction to readable text
            result = 'Safe' if output[0] == 1 else 'Not Safe'

            return render_template('output.html', output=result)

        except Exception as e:
            return f"Error: {str(e)}"

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
