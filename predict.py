
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from extract_features import extract_mfcc
import joblib
import os
import sys
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import time

def load_encoder(csv_path="features.csv"):
    df = pd.read_csv(csv_path)
    encoder = LabelEncoder()
    encoder.fit(df["label"].values)
    return encoder

ALERT_MESSAGES = {
    "screaming_distress": "🚨 ALERT: Someone may be in distress or danger nearby!",
    "gunshot":            "🔫 ALERT: Possible gunshot detected nearby!",
    "siren":              "🚒 ALERT: Emergency vehicle siren detected nearby!",
    "vehicle_horn":       "🚗 ALERT: Vehicle horn detected - watch for traffic!",
    "angry_confrontation":"😠 ALERT: Angry confrontation detected nearby!",
    "safe":               "✅ Environment sounds safe."
}

DANGER_CLASSES = {"screaming_distress", "gunshot", "siren", "vehicle_horn", "angry_confrontation"}

def predict_sound(file_path, model, encoder, scaler):
    mfcc = extract_mfcc(file_path)
    mfcc_scaled = scaler.transform(np.expand_dims(mfcc, axis=0))
    prediction = model.predict(mfcc_scaled, verbose=0)[0]

    class_index = np.argmax(prediction)
    sound_label = encoder.inverse_transform([class_index])[0]
    confidence = prediction[class_index] * 100

    danger_triggered = None
    danger_confidence = 0
    for i, label in enumerate(encoder.classes_):
        if label in DANGER_CLASSES and prediction[i] * 100 > 30:
            if prediction[i] > danger_confidence:
                danger_triggered = label
                danger_confidence = prediction[i]

    if danger_triggered and sound_label == "safe":
        sound_label = danger_triggered
        confidence = danger_confidence * 100
        print(f"  Detected: {sound_label} ({confidence:.1f}% confidence) [low confidence alert]")
    else:
        print(f"  Detected: {sound_label} ({confidence:.1f}% confidence)")

    alert = ALERT_MESSAGES.get(sound_label, "Unknown sound detected.")
    print(f"  {alert}")
    return sound_label, confidence

def listen_realtime(model, encoder, scaler, duration=3, samplerate=22050):
    print("\n🎤 Real-time listening started! Press Ctrl+C to stop.\n")
    while True:
        print(f"Listening for {duration} seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        wav.write(tmp_path, samplerate, audio)
        predict_sound(tmp_path, model, encoder, scaler)
        os.unlink(tmp_path)
        print("-" * 40)

if __name__ == "__main__":
    MODEL_PATH = "safety_alert_model.h5"
    CSV_PATH = "features.csv"
    SCALER_PATH = "scaler.pkl"

    model = load_model(MODEL_PATH)
    encoder = load_encoder(CSV_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model loaded. Classes: {list(encoder.classes_)}")
    print("-" * 50)

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"Error: File not found: {audio_file}")
            sys.exit(1)
        print(f"Analyzing: {audio_file}")
        predict_sound(audio_file, model, encoder, scaler)
    else:
        listen_realtime(model, encoder, scaler)
