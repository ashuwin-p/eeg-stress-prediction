from flask import Flask, request, jsonify
import pandas as pd
import joblib
from prediction import EEGStressPredictor
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Load model and scaler
model_path = Path(__file__).parent / "best_xgboost_model.pkl"
scaler_path = Path(__file__).parent / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

predictor = EEGStressPredictor(model, scaler)

@app.route('/')
def home():
    return "EEG Stress Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        prediction = predictor.predict(df)

        # Convert to JSON serializable format
        if isinstance(prediction, (np.ndarray, list)):
            prediction = np.array(prediction).tolist()
        elif isinstance(prediction, dict):
            prediction = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                          for k, v in prediction.items()}
        elif isinstance(prediction, (np.float32, np.float64)):
            prediction = float(prediction)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
