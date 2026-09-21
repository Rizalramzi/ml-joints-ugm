
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
    / "student_profiles.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "learning_recommendations.csv"
)


# ==========================================
# REKOMENDASI BERDASARKAN MINAT
# ==========================================

INTEREST_ACTIVITY_MAP = {
    "Game": (
        "Gunakan latihan berbasis permainan "
        "dan tantangan bertahap."
    ),
    "Musik": (
        "Gunakan penjelasan audio dan contoh "
        "yang dikaitkan dengan pola atau ritme."
    ),
    "Sains": (
        "Gunakan contoh eksperimen sederhana "
        "dan hubungan konsep dengan fenomena."
    ),
    "Teknologi": (
        "Gunakan simulasi, ilustrasi digital, "
        "dan contoh teknologi."
    ),
    "Olahraga": (
        "Gunakan latihan berbasis target "
        "dan aktivitas bertahap."
    ),
}


# ==========================================
# LOGIKA REKOMENDASI
# ==========================================

def determine_recommendation(row):
    priority = row["learning_priority"]
    cluster_name = row["cluster_name"]
    interest = row["interest"]

    activity = INTEREST_ACTIVITY_MAP.get(
        interest,
        "Gunakan latihan dan penjelasan bertahap.",
    )

    if priority == "High Priority":
        learning_strategy = (
            "Mulai dari konsep dasar, "
            "contoh soal, dan latihan dengan bimbingan."
        )

        difficulty_level = "Easy"

    elif priority == "Medium Priority":
        learning_strategy = (
            "Gunakan ringkasan materi, "
            "contoh soal, dan latihan bertahap."
        )

        difficulty_level = "Medium"

    else:
        learning_strategy = (
            "Berikan latihan pengayaan "
            "dan soal dengan tingkat kesulitan lebih tinggi."
        )

        difficulty_level = "Hard"

    if cluster_name == "Fast Learner":
        additional_instruction = (
            "Tambahkan soal pengayaan setelah "
            "konsep utama dikuasai."
        )

    elif cluster_name == "Needs Guidance":
        additional_instruction = (
            "Berikan bantuan langkah demi langkah "
            "dan pengecekan pemahaman."
        )

    else:
        additional_instruction = (
            "Pertahankan latihan rutin "
            "dengan tingkat kesulitan bertahap."
        )

    return pd.Series({
        "recommended_difficulty": difficulty_level,
        "learning_strategy": learning_strategy,
        "interest_based_activity": activity,
        "additional_instruction": additional_instruction,
    })


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - LEARNING RECOMMENDATION")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(INPUT_PATH)

    required_columns = [
        "student_id",
        "learning_priority",
        "cluster_name",
        "interest",
        "learning_gap",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom yang dibutuhkan tidak ditemukan: "
            f"{missing_columns}"
        )

    recommendations = data.apply(
        determine_recommendation,
        axis=1,
    )

    result = pd.concat(
        [data, recommendations],
        axis=1,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nContoh rekomendasi:")

    print(
        result[
            [
                "student_id",
                "cluster_name",
                "learning_priority",
                "recommended_difficulty",
                "learning_strategy",
            ]
        ].head(10)
    )

    print(
        f"\nRekomendasi disimpan di:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()