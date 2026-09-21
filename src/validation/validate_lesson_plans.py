
import json
from pathlib import Path


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

LESSON_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "lesson_plans.jsonl"
)

EXPECTED_STUDENTS = 50

REQUIRED_FIELDS = [
    "student_id",
    "learning_objective",
    "lesson_explanation",
    "learning_activities",
    "practice_questions",
    "evaluation_method",
]

REQUIRED_QUESTION_FIELDS = [
    "question",
    "difficulty",
    "answer",
]

VALID_DIFFICULTIES = [
    "Easy",
    "Medium",
    "Hard",
]


# ==========================================
# VALIDASI SATU LESSON PLAN
# ==========================================

def validate_lesson_plan(record):
    errors = []

    if not isinstance(record, dict):
        return ["Record bukan dictionary."]

    if "student_id" not in record:
        errors.append("student_id tidak ditemukan.")

    if "lesson_plan" not in record:
        errors.append("lesson_plan tidak ditemukan.")
        return errors

    lesson_plan = record["lesson_plan"]

    if not isinstance(lesson_plan, dict):
        errors.append("lesson_plan bukan dictionary.")
        return errors

    for field in REQUIRED_FIELDS:
        if field not in lesson_plan:
            errors.append(
                f"Field wajib tidak ditemukan: {field}"
            )

    if (
        "student_id" in lesson_plan
        and "student_id" in record
    ):
        if (
            str(lesson_plan["student_id"])
            != str(record["student_id"])
        ):
            errors.append(
                "student_id pada lesson_plan tidak sesuai."
            )

    for field in [
        "learning_objective",
        "lesson_explanation",
        "learning_activities",
        "practice_questions",
        "evaluation_method",
    ]:
        if field in lesson_plan:
            if not isinstance(lesson_plan[field], list):
                errors.append(
                    f"{field} harus berupa list."
                )
            elif len(lesson_plan[field]) == 0:
                errors.append(
                    f"{field} tidak boleh kosong."
                )

    if "practice_questions" in lesson_plan:
        questions = lesson_plan["practice_questions"]

        if isinstance(questions, list):
            for index, question in enumerate(questions):
                if not isinstance(question, dict):
                    errors.append(
                        f"Soal ke-{index + 1} bukan dictionary."
                    )
                    continue

                for field in REQUIRED_QUESTION_FIELDS:
                    if field not in question:
                        errors.append(
                            f"Soal ke-{index + 1}: "
                            f"field {field} tidak ditemukan."
                        )

                if "difficulty" in question:
                    if (
                        question["difficulty"]
                        not in VALID_DIFFICULTIES
                    ):
                        errors.append(
                            f"Soal ke-{index + 1}: "
                            "difficulty tidak valid."
                        )

                for field in [
                    "question",
                    "answer",
                ]:
                    if field in question:
                        if not isinstance(
                            question[field],
                            str,
                        ):
                            errors.append(
                                f"Soal ke-{index + 1}: "
                                f"{field} harus string."
                            )
                        elif not question[field].strip():
                            errors.append(
                                f"Soal ke-{index + 1}: "
                                f"{field} kosong."
                            )

    return errors


# ==========================================
# MAIN VALIDATION
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - LESSON PLAN VALIDATOR")
    print("=" * 60)

    if not LESSON_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {LESSON_PLAN_PATH}"
        )

    records = []
    invalid_records = []
    student_ids = set()

    with open(
        LESSON_PLAN_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                invalid_records.append({
                    "line": line_number,
                    "errors": [
                        f"JSON tidak valid: {error}"
                    ],
                })
                continue

            student_id = str(
                record.get("student_id", "")
            )

            errors = validate_lesson_plan(record)

            if student_id in student_ids:
                errors.append(
                    "student_id duplikat."
                )

            if student_id:
                student_ids.add(student_id)

            if errors:
                invalid_records.append({
                    "line": line_number,
                    "student_id": student_id,
                    "errors": errors,
                })

            records.append(record)

    total_records = len(records)
    valid_records = total_records - len(invalid_records)

    print(f"\nTotal record: {total_records}")
    print(f"Jumlah student ID unik: {len(student_ids)}")
    print(f"Record valid: {valid_records}")
    print(f"Record tidak valid: {len(invalid_records)}")

    if total_records != EXPECTED_STUDENTS:
        print(
            f"\nPERINGATAN: Jumlah record seharusnya "
            f"{EXPECTED_STUDENTS}."
        )

    if invalid_records:
        print("\nDetail error:")

        for invalid in invalid_records:
            print(
                f"\nBaris: {invalid.get('line')}"
            )
            print(
                f"Student ID: "
                f"{invalid.get('student_id', '-')}"
            )

            for error in invalid["errors"]:
                print(f"- {error}")

    else:
        print("\nSemua lesson plan berhasil divalidasi.")

    if (
        total_records == EXPECTED_STUDENTS
        and len(student_ids) == EXPECTED_STUDENTS
        and not invalid_records
    ):
        print("\nSTATUS: VALID")
    else:
        print("\nSTATUS: PERLU DIPERIKSA")


if __name__ == "__main__":
    main()