
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"

DATA_PATH = BASE_DIR / "data" / "raw" / "students.csv"


FEATURES = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
    "learning_speed",
]


def main():
    scaler_path = MODEL_DIR / "scaler.pkl"
    kmeans_path = MODEL_DIR / "kmeans.pkl"

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)

    data = pd.read_csv(DATA_PATH)

    sample_student = data[FEATURES].iloc[[0]]

    scaled_features = scaler.transform(
        sample_student
    )

    cluster_prediction = kmeans.predict(
        scaled_features
    )

    print("=" * 50)
    print("TEST SAVED MODEL")
    print("=" * 50)

    print("\nData siswa yang diuji:")
    print(sample_student)

    print("\nHasil scaling:")
    print(scaled_features)

    print("\nPrediksi cluster:")
    print(cluster_prediction[0])

    print("\nModel berhasil digunakan kembali.")


if __name__ == "__main__":
    main()