
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


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
    / "lesson_plans.jsonl"
)

ERROR_PATH = (
    OUTPUT_DIR
    / "lesson_plan_errors.jsonl"
)

ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)


# ==========================================
# VALIDASI ENVIRONMENT
# ==========================================

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY belum ditemukan. "
        "Pastikan sudah mengisi file .env."
    )


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=60000
    )
)


# ==========================================
# JSON SCHEMA
# ==========================================

LESSON_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "student_id": {
            "type": "string",
            "description": "ID siswa",
        },
        "learning_objective": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Tujuan pembelajaran",
        },
        "lesson_explanation": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Penjelasan materi bertahap",
        },
        "learning_activities": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Aktivitas pembelajaran",
        },
        "practice_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "Easy",
                            "Medium",
                            "Hard",
                        ],
                    },
                    "answer": {
                        "type": "string",
                    },
                },
                "required": [
                    "question",
                    "difficulty",
                    "answer",
                ],
            },
            "description": "Latihan soal",
        },
        "evaluation_method": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Metode evaluasi",
        },
    },
    "required": [
        "student_id",
        "learning_objective",
        "lesson_explanation",
        "learning_activities",
        "practice_questions",
        "evaluation_method",
    ],
}


# ==========================================
# PROMPT
# ==========================================

def build_prompt(row):
    profile = {
        "student_id": str(row["student_id"]),
        "cluster_name": str(row["cluster_name"]),
        "learning_priority": str(
            row["learning_priority"]
        ),
        "mastery_score": round(
            float(row["mastery_score"]),
            4,
        ),
        "learning_gap": round(
            float(row["learning_gap"]),
            4,
        ),
        "interest": str(row["interest"]),
        "recommended_difficulty": str(
            row["recommended_difficulty"]
        ),
        "learning_strategy": str(
            row["learning_strategy"]
        ),
        "interest_based_activity": str(
            row["interest_based_activity"]
        ),
        "additional_instruction": str(
            row["additional_instruction"]
        ),
    }

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Kamu adalah AI tutor EduAdapt AI.

Buat lesson plan personal untuk siswa SMP.

Profil siswa:
{profile_json}

Konteks pembelajaran:
- Mata pelajaran: Matematika
- Materi: Persamaan Linear Satu Variabel
- Bahasa: Indonesia
- Tingkat: SMP

Instruksi:
1. Sesuaikan materi dengan profil siswa.
2. Gunakan bahasa yang sederhana dan jelas.
3. Jangan mengklaim siswa menguasai materi
   yang belum dibuktikan.
4. Gunakan pendekatan bertahap.
5. Sesuaikan tingkat kesulitan latihan.
6. Sertakan jawaban untuk setiap latihan.
7. Jawaban harus relevan dengan materi PLSV.
8. Jangan memasukkan markdown.
9. Kembalikan hanya JSON sesuai schema.
""".strip()


# ==========================================
# VALIDASI HASIL
# ==========================================

REQUIRED_FIELDS = [
    "student_id",
    "learning_objective",
    "lesson_explanation",
    "learning_activities",
    "practice_questions",
    "evaluation_method",
]


def validate_lesson_plan(result, expected_student_id):
    if not isinstance(result, dict):
        raise ValueError(
            "Response bukan JSON object."
        )

    for field in REQUIRED_FIELDS:
        if field not in result:
            raise ValueError(
                f"Field wajib tidak ditemukan: {field}"
            )

    if result["student_id"] != expected_student_id:
        raise ValueError(
            "student_id pada response tidak sesuai."
        )

    if not isinstance(
        result["practice_questions"],
        list,
    ):
        raise ValueError(
            "practice_questions harus berupa list."
        )

    for question in result["practice_questions"]:
        required_question_fields = [
            "question",
            "difficulty",
            "answer",
        ]

        for field in required_question_fields:
            if field not in question:
                raise ValueError(
                    f"Field soal tidak ditemukan: {field}"
                )

        if question["difficulty"] not in [
            "Easy",
            "Medium",
            "Hard",
        ]:
            raise ValueError(
                "Difficulty tidak valid."
            )

    return True


# ==========================================
# REQUEST GEMINI
# ==========================================

def generate_lesson_plan(prompt):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LESSON_PLAN_SCHEMA
                )
            )

            if response is None:
                raise RuntimeError(
                    "Gemini tidak mengembalikan response."
                )

            if not response.text:
                raise RuntimeError(
                    "Gemini mengembalikan response kosong."
                )

            result = json.loads(response.text)
            return result

        except Exception as e:
            error_message = str(e)

        retryable_errors = (
            "503",
            "UNAVAILABLE",
            "timeout",
            "timed out",
            "DeadlineExceeded"
        )

        if any(
            error in error_message
            for error in retryable_errors
        ):
            if attempt < max_retries - 1:
                delay = 5 * (2 ** attempt)

                print(
                    f"Request gagal sementara. "
                    f"Retry {attempt + 1}/"
                    f"{max_retries - 1} "
                    f"dalam {delay} detik..."
                )

                time.sleep(delay)
            else:
                raise
        else:
            raise RuntimeError(
                f"Gagal memproses Gemini: {error_message}"
            )

# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - GEMINI LESSON PLAN GENERATOR")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(INPUT_PATH)

    print(f"\nModel Gemini: {GEMINI_MODEL}")
    print(f"Jumlah siswa: {len(data)}")

    success_count = 0
    error_count = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file, open(
        ERROR_PATH,
        "w",
        encoding="utf-8",
    ) as error_file:

        for index, row in data.iterrows():
            student_id = str(row["student_id"])

            print(
                f"\n[{index + 1}/{len(data)}] "
                f"Memproses {student_id}..."
            )

            try:
                prompt = build_prompt(row)

                result = generate_lesson_plan(
                    prompt
                )

                validate_lesson_plan(
                    result,
                    student_id,
                )

                record = {
                    "student_id": student_id,
                    "lesson_plan": result,
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                success_count += 1

                print(
                    f"Berhasil: {student_id}"
                )

            except Exception as error:
                error_count += 1

                error_record = {
                    "student_id": student_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }

                error_file.write(
                    json.dumps(
                        error_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                error_file.flush()

                print(
                    f"Gagal: {student_id} - {error}"
                )

            # Jeda antar request
            # untuk mengurangi risiko rate limit
            if index < len(data) - 1:
                time.sleep(1)

    print("\n" + "=" * 60)
    print("PROSES SELESAI")
    print("=" * 60)

    print(f"Berhasil: {success_count}")
    print(f"Gagal: {error_count}")

    print(
        f"\nLesson plan disimpan di:\n{OUTPUT_PATH}"
    )

    print(
        f"\nError disimpan di:\n{ERROR_PATH}"
    )


if __name__ == "__main__":
    main()