def generate_dynamic_recommendation(
    student_data: dict,
    prediction: dict,
):
    diagnostic_score = student_data["diagnostic_score"]
    assignment_avg = student_data["assignment_avg"]
    quiz_avg = student_data["quiz_avg"]
    uts_score = student_data["uts_score"]

    scores = {
        "Diagnostic": diagnostic_score,
        "Assignment": assignment_avg,
        "Quiz": quiz_avg,
        "UTS": uts_score,
    }

    weakest_area = min(
        scores,
        key=scores.get,
    )

    weakest_score = scores[weakest_area]

    if weakest_score < 60:
        difficulty = "Dasar"
        strategy = "Pembelajaran konsep dan latihan bertahap"
    elif weakest_score < 75:
        difficulty = "Menengah"
        strategy = "Latihan terarah dan penguatan konsep"
    else:
        difficulty = "Lanjutan"
        strategy = "Latihan penerapan dan soal tantangan"

    if prediction["priority"] == "High":
        focus = "Penguatan konsep dasar"
    elif prediction["priority"] == "Medium":
        focus = "Penguatan materi yang masih lemah"
    else:
        focus = "Pengembangan kemampuan lanjutan"

    return {
        "focus": focus,
        "weakest_assessment": weakest_area,
        "weakest_score": round(
            weakest_score,
            2,
        ),
        "difficulty": difficulty,
        "learning_strategy": strategy,
        "reason": (
            f"Nilai terendah terdapat pada "
            f"{weakest_area} dengan nilai "
            f"{weakest_score:.2f}."
        ),
    }