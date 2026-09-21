
from pathlib import Path

import json
import pandas as pd


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "learning_recommendations.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "lesson_plan_prompts.jsonl"
)


# ==========================================
# PROMPT BUILDER
# ==========================================

def build_lesson_plan_prompt(row):
    student_profile = {
        "student_id": row["student_id"],
        "cluster_name": row["cluster_name"],
        "learning_priority": row["learning_priority"],
        "mastery_score": round(
            float(row["mastery_score"]),
            4,
        ),
        "learning_gap": round(
            float(row["learning_gap"]),
            4,
        ),
        "interest": row["interest"],
        "recommended_difficulty": (
            row["recommended_difficulty"]
        ),
        "learning_strategy": row["learning_strategy"],
        "interest_based_activity": (
            row["interest_based_activity"]
        ),
        "additional_instruction": (
            row["additional_instruction"]
        ),
    }

    profile_json = json.dumps(
        student_profile,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
Kamu adalah AI tutor untuk platform EduAdapt AI.

Buat rencana pembelajaran personal berdasarkan
profil siswa berikut:

{profile_json}

Konteks pembelajaran:
- Mata pelajaran: Matematika
- Jenjang: SMP
- Materi: Persamaan Linear Satu Variabel

Ketentuan:
1. Sesuaikan tingkat kesulitan dengan profil siswa.
2. Gunakan bahasa Indonesia yang mudah dipahami.
3. Berikan tujuan pembelajaran yang jelas.
4. Berikan penjelasan materi secara bertahap.
5. Berikan aktivitas pembelajaran yang relevan.
6. Sertakan latihan soal.
7. Jangan menganggap siswa sudah menguasai
   materi yang belum dibuktikan.

Kembalikan jawaban hanya dalam format JSON
dengan struktur berikut:

{{
  "student_id": "string",
  "learning_objective": [
    "string"
  ],
  "lesson_explanation": [
    "string"
  ],
  "learning_activities": [
    "string"
  ],
  "practice_questions": [
    {{
      "question": "string",
      "difficulty": "Easy|Medium|Hard",
      "answer": "string"
    }}
  ],
  "evaluation_method": [
    "string"
  ]
}}
"""

    return prompt.strip()


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - LLM PROMPT BUILDER")
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

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for _, row in data.iterrows():
            prompt = build_lesson_plan_prompt(row)

            record = {
                "student_id": row["student_id"],
                "prompt": prompt,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"\nJumlah prompt: {len(data)}"
    )

    print(
        f"Prompt disimpan di:\n{OUTPUT_PATH}"
    )

    print("\nContoh prompt siswa pertama:\n")
    print(build_lesson_plan_prompt(data.iloc[0]))


if __name__ == "__main__":
    main()