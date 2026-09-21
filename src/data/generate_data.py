
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_STUDENTS = 50


def generate_student_data(
    n_students: int = N_STUDENTS,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_ids = [
        f"S{i:03d}"
        for i in range(1, n_students + 1)
    ]

    # Nilai sintetis untuk keperluan pengujian prototipe.
    diagnostic_score = rng.normal(65, 15, n_students)
    assignment_avg = rng.normal(70, 13, n_students)
    quiz_avg = rng.normal(68, 15, n_students)
    uts_score = rng.normal(67, 14, n_students)

    # Fitur preferensi sintetis.
    learning_speed = rng.uniform(0.2, 1.0, n_students)
    visual_preference = rng.uniform(0.0, 1.0, n_students)
    audio_preference = rng.uniform(0.0, 1.0, n_students)

    interests = rng.choice(
        [
            "Olahraga",
            "Teknologi",
            "Musik",
            "Game",
            "Sains",
        ],
        size=n_students,
    )

    observation_notes = rng.choice(
        [
            "Mampu bekerja mandiri",
            "Perlu contoh tambahan",
            "Aktif berdiskusi",
            "Perlu bimbingan",
            "Cepat memahami konsep",
        ],
        size=n_students,
    )

    data = pd.DataFrame(
        {
            "student_id": student_ids,
            "diagnostic_score": diagnostic_score,
            "assignment_avg": assignment_avg,
            "quiz_avg": quiz_avg,
            "uts_score": uts_score,
            "learning_speed": learning_speed,
            "visual_preference": visual_preference,
            "audio_preference": audio_preference,
            "interest": interests,
            "observation_note": observation_notes,
        }
    )

    score_columns = [
        "diagnostic_score",
        "assignment_avg",
        "quiz_avg",
        "uts_score",
    ]

    # Batasi nilai ke rentang 0-100.
    data[score_columns] = data[score_columns].clip(0, 100)

    return data


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_student_data()

    output_path = output_dir / "students.csv"
    data.to_csv(output_path, index=False)

    print(f"Dataset berhasil disimpan: {output_path}")
    print(f"Jumlah siswa: {len(data)}")
    print("\nContoh data:")
    print(data.head())

    print("\nInformasi dataset:")
    print(data.info())


if __name__ == "__main__":
    main()