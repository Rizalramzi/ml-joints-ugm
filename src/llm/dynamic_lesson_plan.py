
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)


if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY belum ditemukan."
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
            "type": "string"
        },
        "learning_objective": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "lesson_explanation": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "learning_activities": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "practice_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "Easy",
                            "Medium",
                            "Hard"
                        ]
                    },
                    "answer": {
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "difficulty",
                    "answer"
                ]
            }
        },
        "evaluation_method": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "student_id",
        "learning_objective",
        "lesson_explanation",
        "learning_activities",
        "practice_questions",
        "evaluation_method"
    ]
}


# ==========================================
# PROMPT DINAMIS
# ==========================================

def build_dynamic_prompt(
    student_data: dict,
    prediction: dict,
    recommendation: dict
):
    profile = {
        "student_id": student_data["student_id"],
        "diagnostic_score": student_data["diagnostic_score"],
        "assignment_avg": student_data["assignment_avg"],
        "quiz_avg": student_data["quiz_avg"],
        "uts_score": student_data["uts_score"],
        "learning_speed": student_data["learning_speed"],
        "cluster_name": prediction["cluster_name"],
        "mastery_score": prediction["mastery_score"],
        "learning_gap": prediction["learning_gap"],
        "priority": prediction["priority"],
        "focus": recommendation["focus"],
        "weakest_assessment": recommendation[
            "weakest_assessment"
        ],
        "weakest_score": recommendation["weakest_score"],
        "difficulty": recommendation["difficulty"],
        "learning_strategy": recommendation[
            "learning_strategy"
        ],
        "reason": recommendation["reason"]
    }

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2
    )

    return f"""
Kamu adalah AI tutor EduAdapt AI.

Buat lesson plan personal untuk siswa SMP
berdasarkan profil siswa berikut:

{profile_json}

Konteks pembelajaran:
- Mata pelajaran: Matematika
- Materi: Persamaan Linear Satu Variabel
- Tingkat: SMP
- Bahasa: Indonesia

Instruksi:
1. Sesuaikan pembelajaran dengan profil siswa.
2. Prioritaskan bagian yang memiliki nilai paling rendah.
3. Gunakan bahasa Indonesia yang sederhana.
4. Gunakan pembelajaran bertahap.
5. Sesuaikan tingkat kesulitan dengan rekomendasi.
6. Sertakan tujuan pembelajaran.
7. Sertakan penjelasan materi.
8. Sertakan aktivitas pembelajaran.
9. Sertakan latihan soal dan jawaban.
10. Sertakan metode evaluasi.
11. Jangan mengklaim siswa menguasai materi
    yang belum terbukti.
12. Jangan menggunakan markdown.
13. Kembalikan hanya JSON sesuai schema.
""".strip()


# ==========================================
# VALIDASI
# ==========================================

def validate_dynamic_lesson_plan(
    result: dict,
    expected_student_id: str
):
    required_fields = [
        "student_id",
        "learning_objective",
        "lesson_explanation",
        "learning_activities",
        "practice_questions",
        "evaluation_method"
    ]

    if not isinstance(result, dict):
        raise ValueError(
            "Response Gemini bukan JSON object."
        )

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Field wajib tidak ditemukan: {field}"
            )

    if result["student_id"] != expected_student_id:
        raise ValueError(
            "student_id dari Gemini tidak sesuai."
        )

    if not isinstance(
        result["practice_questions"],
        list
    ):
        raise ValueError(
            "practice_questions harus berupa list."
        )

    for question in result["practice_questions"]:
        for field in [
            "question",
            "difficulty",
            "answer"
        ]:
            if field not in question:
                raise ValueError(
                    f"Field soal tidak ditemukan: {field}"
                )

        if question["difficulty"] not in [
            "Easy",
            "Medium",
            "Hard"
        ]:
            raise ValueError(
                "Difficulty soal tidak valid."
            )

    return True


# ==========================================
# REQUEST GEMINI
# ==========================================


def generate_dynamic_lesson_plan(
    student_data: dict,
    prediction: dict,
    recommendation: dict
):
    prompt = build_dynamic_prompt(
        student_data,
        prediction,
        recommendation
    )

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

            if response is None or not response.text:
                raise RuntimeError(
                    "Gemini mengembalikan response kosong."
                )

            result = json.loads(response.text)

            validate_dynamic_lesson_plan(
                result,
                student_data["student_id"]
            )

            return result

        except Exception as error:
            error_message = str(error)

            retryable_errors = (
                "503",
                "504",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
                "timeout",
                "timed out",
                "DeadlineExceeded"
            )

            is_retryable = any(
                error_text in error_message
                for error_text in retryable_errors
            )

            if is_retryable and attempt < max_retries - 1:
                delay = 5 * (2 ** attempt)

                print(
                    f"Request Gemini gagal sementara: "
                    f"{error_message}"
                )

                print(
                    f"Retry {attempt + 1}/"
                    f"{max_retries - 1} "
                    f"dalam {delay} detik..."
                )

                time.sleep(delay)
                continue

            raise RuntimeError(
                f"Gagal membuat lesson plan: "
                f"{error_message}"
            )

