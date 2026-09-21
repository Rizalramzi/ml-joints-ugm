
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.llm.gemini_lesson_plan import (
    build_prompt,
    validate_lesson_plan,
    generate_lesson_plan,
    INPUT_PATH,
    OUTPUT_DIR,
    OUTPUT_PATH,
    ERROR_PATH,
)


def load_failed_students():
    """Membaca daftar siswa yang gagal diproses."""

    if not ERROR_PATH.exists():
        print("File error tidak ditemukan.")
        return []

    failed_students = []

    with open(
        ERROR_PATH,
        "r",
        encoding="utf-8",
    ) as error_file:

        for line in error_file:
            if line.strip():
                record = json.loads(line)
                failed_students.append(
                    record["student_id"]
                )

    return failed_students


def load_existing_lesson_plans():
    """Membaca lesson plan yang sudah berhasil."""

    existing_plans = []

    if not OUTPUT_PATH.exists():
        return existing_plans

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as output_file:

        for line in output_file:
            if line.strip():
                existing_plans.append(
                    json.loads(line)
                )

    return existing_plans


def main():
    print("=" * 60)
    print("EDUADAPT AI - RETRY FAILED LESSON PLANS")
    print("=" * 60)

    data = pd.read_csv(INPUT_PATH)

    failed_students = load_failed_students()
    existing_plans = load_existing_lesson_plans()

    existing_ids = {
        record["student_id"]
        for record in existing_plans
    }

    print(
        f"\nJumlah siswa gagal: "
        f"{len(failed_students)}"
    )

    if not failed_students:
        print("Tidak ada siswa yang perlu diulang.")
        return

    success_count = 0
    error_count = 0

    new_plans = []
    new_errors = []

    for index, row in data.iterrows():
        student_id = str(row["student_id"])

        if student_id not in failed_students:
            continue

        if student_id in existing_ids:
            continue

        print(
            f"\nMemproses ulang {student_id}..."
        )

        try:
            prompt = build_prompt(row)

            result = generate_lesson_plan(prompt)

            validate_lesson_plan(
                result,
                student_id,
            )

            record = {
                "student_id": student_id,
                "lesson_plan": result,
            }

            new_plans.append(record)
            success_count += 1

            print(
                f"Berhasil: {student_id}"
            )

        except Exception as error:
            error_record = {
                "student_id": student_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            new_errors.append(error_record)
            error_count += 1

            print(
                f"Gagal: {student_id} - {error}"
            )

        # Jeda agar tidak terlalu cepat
        time.sleep(5)

    # Menambahkan lesson plan baru
    if new_plans:
        with open(
            OUTPUT_PATH,
            "a",
            encoding="utf-8",
        ) as output_file:

            for record in new_plans:
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Mengganti daftar error dengan error yang terbaru
    remaining_errors = []

    for error in new_errors:
        remaining_errors.append(error)

    if ERROR_PATH.exists():
        with open(
            ERROR_PATH,
            "w",
            encoding="utf-8",
        ) as error_file:

            for error in remaining_errors:
                error_file.write(
                    json.dumps(
                        error,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("\n" + "=" * 60)
    print("RETRY SELESAI")
    print("=" * 60)

    print(f"Berhasil: {success_count}")
    print(f"Gagal: {error_count}")


if __name__ == "__main__":
    main()