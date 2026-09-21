
from pathlib import Path

import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "students_clustered.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "student_profiles.csv"


SCORE_COLUMNS = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
]


# ==========================================
# LOAD DATA
# ==========================================

def load_clustered_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


# ==========================================
# HITUNG PROFIL SISWA
# ==========================================

def calculate_student_profile(data):
    missing_columns = [
        column
        for column in SCORE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    data = data.copy()

    # Nilai rata-rata performa akademik
    data["average_score"] = data[SCORE_COLUMNS].mean(
        axis=1
    )

    # Normalisasi skor menjadi 0 sampai 1
    data["mastery_score"] = (
        data["average_score"] / 100
    ).clip(0, 1)

    # Learning gap:
    # semakin tinggi nilainya, semakin besar kebutuhan belajar
    data["learning_gap"] = (
        1 - data["mastery_score"]
    ).clip(0, 1)

    # Kategori kebutuhan belajar
    def determine_learning_priority(gap):
        if gap >= 0.40:
            return "High Priority"
        elif gap >= 0.25:
            return "Medium Priority"
        else:
            return "Low Priority"

    data["learning_priority"] = data[
        "learning_gap"
    ].apply(determine_learning_priority)

    return data


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - STUDENT PROFILE")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_clustered_data()

    profiles = calculate_student_profile(data)

    profiles.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nContoh profil siswa:")
    print(
        profiles[
            [
                "student_id",
                "average_score",
                "mastery_score",
                "learning_gap",
                "learning_priority",
                "cluster_name",
            ]
        ].head(10)
    )

    print("\nDistribusi prioritas belajar:")
    print(
        profiles["learning_priority"]
        .value_counts()
    )

    print(
        f"\nProfil siswa disimpan di:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()