from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
# Configure CORS pour accepter les requêtes du frontend Vercel
CORS(app, origins=[
    'http://localhost:3000',  # développement local
    'https://*.vercel.app',   # domaines Vercel
    'https://mycar-frontend.vercel.app'  # ton domaine spécifique
])

# Chargez votre modèle et vos encodeurs
try:
    model = joblib.load('model/car_price_model.joblib')
    encoders = joblib.load('model/encoders.joblib')
except:
    print("Warning: Model files not found. Prediction will return dummy data.")

# Ajoutez une constante pour le taux de conversion (à mettre à jour régulièrement)
USD_TO_TND_RATE = 3.12  # exemple de taux

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extraction des données du frontend
        year = data.get('year')
        make = data.get('make')
        model_name = data.get('model')
        mileage = data.get('mileage')
        condition = data.get('condition', 5)  # Valeur par défaut de 5

        # Validation des données
        if not all([year, make, model_name, mileage]):
            return jsonify({
                'error': 'Missing required fields'
            }), 400

        try:
            # Prédiction du prix
            # Remplacez ce bloc par votre logique de prédiction réelle
            predicted_price = calculate_predicted_price(year, make, model_name, mileage, condition)
            
            # Format de réponse correspondant au frontend
            return jsonify({
                'predicted_price': predicted_price
            })

        except Exception as e:
            return jsonify({
                'error': f'Prediction error: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

def calculate_predicted_price(year, make, model_name, mileage, condition):
    
    try:
        # Si le modèle est chargé, utilisez-le pour la prédiction
        if 'model' in globals():
            # Préparation des features
            features = np.array([[
                year,
                encoders['make_encoder'].transform([make])[0],
                encoders['model_encoder'].transform([model_name])[0],
                mileage,
                condition
            ]])
            
            # Prédiction
            predicted_price = model.predict(features)[0]
            # Conversion en TND
            predicted_price_tnd = predicted_price * USD_TO_TND_RATE
            return float(predicted_price_tnd)
            
        else:
            # Retourne une estimation factice pour les tests
            base_price = 50000
            age_factor = (datetime.now().year - year) * 1000
            mileage_factor = mileage * 0.01
            condition_factor = (10 - condition) * 1000
            
            estimated_price = base_price - age_factor - mileage_factor - condition_factor
            # Conversion en TND
            estimated_price_tnd = estimated_price * USD_TO_TND_RATE
            return max(estimated_price_tnd, 5000 * USD_TO_TND_RATE)  # Prix minimum de 5000

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)