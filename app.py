from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("gold_model.pkl")  # Your trained model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['feature1'], data['feature2']]  # Customize input
    prediction = model.predict([features])
    return jsonify({'predicted_price': prediction[0]})

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)