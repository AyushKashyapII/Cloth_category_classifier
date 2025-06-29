from flask import Flask, request, jsonify
from model import predict_attributes
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Fashion Classifier is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    try:
        result = predict_attributes(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
