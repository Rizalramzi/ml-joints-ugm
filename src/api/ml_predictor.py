
from pathlib import Path
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
    "learning_speed",
]


def load_models():
    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    kmeans = joblib.load(
        MODEL_DIR / "kmeans.pkl"
    )

    with open(
        MODEL_DIR / "cluster_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return scaler, kmeans, metadata


def predict_student(student_data: dict):
    scaler, kmeans, metadata = load_models()

    features = pd.DataFrame(
        [[
            student_data[feature]
            for feature in FEATURES
        ]],
        columns=FEATURES,
    )

    scaled_features = scaler.transform(features)

    cluster_id = int(
        kmeans.predict(scaled_features)[0]
    )

    cluster_mapping = metadata[
        "cluster_mapping"
    ]

    cluster_name = cluster_mapping.get(
        str(cluster_id),
        "Unknown",
    )

    average_score = sum(
        student_data[feature]
        for feature in [
            "diagnostic_score",
            "assignment_avg",
            "quiz_avg",
            "uts_score",
        ]
    ) / 4

    mastery_score = average_score / 100

    learning_gap = 1 - mastery_score

    if learning_gap >= 0.40:
        priority = "High"
    elif learning_gap >= 0.25:
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "average_score": round(
            average_score,
            4,
        ),
        "mastery_score": round(
            mastery_score,
            4,
        ),
        "learning_gap": round(
            learning_gap,
            4,
        ),
        "priority": priority,
    }