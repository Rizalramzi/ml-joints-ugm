
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field
from src.api.ml_predictor import predict_student

from src.recommendation.dynamic_recommendation import (
    generate_dynamic_recommendation,
)
# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROFILE_PATH = (
    PROCESSED_DIR / "student_profiles.csv"
)

RECOMMENDATION_PATH = (
    PROCESSED_DIR / "learning_recommendations.csv"
)

LESSON_PLAN_PATH = (
    PROCESSED_DIR / "lesson_plans.jsonl"
)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="EduAdapt AI API",
    description=(
        "Backend API untuk personalisasi "
        "pembelajaran berbasis Machine Learning "
        "dan Gemini AI."
    ),
    version="1.0.0",
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    return pd.read_csv(path)


def load_lesson_plans():
    lesson_plans = {}

    if not LESSON_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {LESSON_PLAN_PATH}"
        )

    with open(
        LESSON_PLAN_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            student_id = str(
                record["student_id"]
            )

            lesson_plans[student_id] = record

    return lesson_plans


def convert_nan_to_none(record):
    return {
        key: (
            None
            if pd.isna(value)
            else value
        )
        for key, value in record.items()
    }

class StudentInput(BaseModel):
    student_id: str = Field(
        min_length=1,
        max_length=50,
    )

    diagnostic_score: float = Field(
        ge=0,
        le=100,
    )

    assignment_avg: float = Field(
        ge=0,
        le=100,
    )

    quiz_avg: float = Field(
        ge=0,
        le=100,
    )

    uts_score: float = Field(
        ge=0,
        le=100,
    )

    learning_speed: float = Field(
        ge=0,
        le=1,
    )

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EduAdapt AI API",
    }


# ==========================================
# STUDENT LIST
# ==========================================

@app.get("/students")
def get_students():
    try:
        profiles = load_csv(PROFILE_PATH)

        return {
            "total": len(profiles),
            "students": [
                convert_nan_to_none(record)
                for record in profiles.to_dict(
                    orient="records"
                )
            ],
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================
# STUDENT PROFILE
# ==========================================

@app.get("/students/{student_id}")
def get_student_profile(student_id: str):
    try:
        profiles = load_csv(PROFILE_PATH)

        student = profiles[
            profiles["student_id"].astype(str)
            == student_id
        ]

        if student.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Siswa {student_id} "
                    "tidak ditemukan."
                ),
            )

        profile = convert_nan_to_none(
            student.iloc[0].to_dict()
        )

        if RECOMMENDATION_PATH.exists():
            recommendations = load_csv(
                RECOMMENDATION_PATH
            )

            recommendation = recommendations[
                recommendations["student_id"].astype(str)
                == student_id
            ]

            if not recommendation.empty:
                profile["recommendation"] = (
                    convert_nan_to_none(
                        recommendation.iloc[0].to_dict()
                    )
                )

        return profile

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================
# LESSON PLAN
# ==========================================

@app.get(
    "/students/{student_id}/lesson-plan"
)
def get_student_lesson_plan(student_id: str):
    try:
        lesson_plans = load_lesson_plans()

        if student_id not in lesson_plans:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Lesson plan untuk "
                    f"{student_id} tidak ditemukan."
                ),
            )

        return lesson_plans[student_id]

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==========================================
# COMBINED STUDENT DATA
# ==========================================

@app.get(
    "/students/{student_id}/dashboard"
)
def get_student_dashboard(student_id: str):
    profile_response = get_student_profile(
        student_id
    )

    lesson_plan_response = (
        get_student_lesson_plan(student_id)
    )

    return {
        "student_id": student_id,
        "profile": profile_response,
        "lesson_plan": lesson_plan_response,
    }


@app.post("/students/analyze")
def analyze_student(student: StudentInput):
    student_data = student.model_dump()

    prediction = predict_student(
        student_data
    )

    recommendation = (
        generate_dynamic_recommendation(
            student_data,
            prediction,
        )
    )

    return {
        "student_id": student.student_id,
        "prediction": prediction,
        "recommendation": recommendation,
    }