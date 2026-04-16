
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import tempfile
import os
import subprocess
from extract_features import extract_mfcc

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = load_model("safety_alert_model.h5")
scaler = joblib.load("scaler.pkl")

# Hardcoded class labels (same order as training)
CLASSES = ["angry_confrontation", "gunshot", "safe", "screaming_distress", "siren", "vehicle_horn"]
encoder = LabelEncoder()
encoder.fit(CLASSES)
print(f"Model loaded! Classes: {list(encoder.classes_)}")

ALERT_MESSAGES = {
    "screaming_distress": "Someone may be in distress or danger nearby!",
    "gunshot":            "Possible gunshot detected nearby!",
    "siren":              "Emergency vehicle siren detected nearby!",
    "vehicle_horn":       "Vehicle horn detected - watch for traffic!",
    "angry_confrontation":"Angry confrontation detected nearby!",
    "safe":               "Environment sounds safe."
}

DANGER_CLASSES = {"screaming_distress", "gunshot", "siren", "vehicle_horn", "angry_confrontation"}

ALERT_LEVELS = {
    "screaming_distress": "critical",
    "gunshot":            "critical",
    "siren":              "warning",
    "vehicle_horn":       "warning",
    "angry_confrontation":"warning",
    "safe":               "safe"
}

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        input_path = f.name
        audio_file.save(input_path)

    wav_path = input_path.replace(".webm", ".wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "22050", "-ac", "1", wav_path
        ], capture_output=True, check=True)
    except Exception as e:
        os.unlink(input_path)
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

    try:
        mfcc = extract_mfcc(wav_path)
        mfcc_scaled = scaler.transform(np.expand_dims(mfcc, axis=0))
        prediction = model.predict(mfcc_scaled, verbose=0)[0]

        class_index = np.argmax(prediction)
        sound_label = encoder.inverse_transform([class_index])[0]
        confidence = float(prediction[class_index] * 100)

        for i, label in enumerate(encoder.classes_):
            if label in DANGER_CLASSES and prediction[i] * 100 > 30:
                if sound_label == "safe":
                    sound_label = label
                    confidence = float(prediction[i] * 100)
                    break

        return jsonify({
            "label": sound_label,
            "confidence": round(confidence, 1),
            "message": ALERT_MESSAGES.get(sound_label, "Unknown sound"),
            "alert_level": ALERT_LEVELS.get(sound_label, "safe"),
            "is_danger": sound_label in DANGER_CLASSES
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
