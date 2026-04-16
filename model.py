
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    X, y = load_data("features.csv")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    print(f"Classes: {list(encoder.classes_)}")
    print(f"Total samples: {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    model = build_model(X_train.shape[1], y_categorical.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint("safety_alert_model.h5", save_best_only=True, verbose=1)
    ]

    history = model.fit(X_train, y_train, epochs=150, batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks)

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")
    print("Done! Model saved.")
