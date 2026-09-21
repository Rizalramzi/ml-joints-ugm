
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
    "learning_speed",
]


def prepare_clustering_features(
    data: pd.DataFrame,
):
    missing_columns = [
        column
        for column in NUMERIC_FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    features = data[NUMERIC_FEATURES].copy()

    if features.isnull().any().any():
        raise ValueError(
            "Fitur clustering mengandung missing value."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "students.csv"

    data = pd.read_csv(file_path)

    scaled_features, scaler = prepare_clustering_features(data)

    print("Fitur yang digunakan:")
    print(NUMERIC_FEATURES)

    print("\nShape fitur sebelum scaling:")
    print(data[NUMERIC_FEATURES].shape)

    print("\nShape setelah scaling:")
    print(scaled_features.shape)

    print("\nRata-rata fitur setelah scaling:")
    print(scaled_features.mean(axis=0))

    print("\nStandar deviasi fitur setelah scaling:")
    print(scaled_features.std(axis=0))


if __name__ == "__main__":
    main()