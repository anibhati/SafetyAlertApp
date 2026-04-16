
import librosa
import numpy as np
import pandas as pd
import os

def extract_mfcc(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

URBANSOUND_LABEL_MAP = {
    "0": "safe",
    "1": "vehicle_horn",
    "2": "safe",
    "3": "safe",
    "4": "safe",
    "5": "safe",
    "6": "gunshot",
    "7": "safe",
    "8": "siren",
    "9": "safe"
}

EMOTIONS_LABEL_MAP = {
    "angry": "angry_confrontation",
    "disgusted": "safe",
    "fearful": "screaming_distress",
    "happy": "safe",
    "neutral": "safe",
    "sad": "safe",
    "suprised": "safe"
}

def build_urbansound_dataset(data_path):
    features, labels = [], []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        mfcc = extract_mfcc(file_path)
                        raw_label = file.split("-")[1]
                        label = URBANSOUND_LABEL_MAP.get(raw_label, "safe")
                        features.append(mfcc)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error with {file}: {e}")
    return features, labels

def build_screaming_dataset(data_path):
    features, labels = [], []
    label_map = {"Screaming": "screaming_distress", "NotScreaming": "safe"}
    for folder_name, label in label_map.items():
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} not found, skipping.")
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                try:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    print(f"Error with {file}: {e}")
    return features, labels

def build_emotions_dataset(data_path):
    features, labels = [], []
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            raw_label = folder_name.lower()
            label = EMOTIONS_LABEL_MAP.get(raw_label, "safe")
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        mfcc = extract_mfcc(file_path)
                        features.append(mfcc)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error with {file}: {e}")
    return features, labels

if __name__ == "__main__":
    URBANSOUND_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "UrbanSounds")
    SCREAMING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HumanScreamingDataSet")
    EMOTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Emotions")

    print("Extracting UrbanSound8K features...")
    us_features, us_labels = build_urbansound_dataset(URBANSOUND_PATH)
    print(f"  -> {len(us_labels)} samples")

    print("Extracting HumanScreaming features...")
    sc_features, sc_labels = build_screaming_dataset(SCREAMING_PATH)
    print(f"  -> {len(sc_labels)} samples")

    print("Extracting Emotions features...")
    em_features, em_labels = build_emotions_dataset(EMOTIONS_PATH)
    print(f"  -> {len(em_labels)} samples")

    all_features = us_features + sc_features + em_features
    all_labels = us_labels + sc_labels + em_labels

    df = pd.DataFrame(all_features)
    df["label"] = all_labels
    df.to_csv("features.csv", index=False)

    print(f"\nDone! Total samples: {len(all_labels)}")
    print("Label distribution:")
    print(pd.Series(all_labels).value_counts())
