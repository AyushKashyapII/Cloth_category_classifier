from flask import Flask, request, jsonify
import os
from inference import load_model, predict

app = Flask(__name__)

# Initialize once on startup
df_dir = "DeepFashion"  # this folder should exist
checkpoint = "results/pretrained_model/epoch_340.pth"  # downloaded manually
model, cat_names, attr_names, attr_types = load_model(df_dir, checkpoint)

@app.route("/")
def home():
    return "Fashion Classifier API is running."

@app.route("/predict", methods=["POST"])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['file']
    filepath = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    try:
        result = predict(model, cat_names, attr_names, attr_types, filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
