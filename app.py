from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "AI Student Score Prediction API Running--running good"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    hours = float(data["hours"])
    prediction = model.predict([[hours]])
    return jsonify({
        "study_hours": hours,
        "predicted_score": round(prediction[0], 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
