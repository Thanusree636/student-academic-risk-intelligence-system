# main.py

# ==============================
# Import Required Libraries
# ==============================

from fastapi import FastAPI
from pydantic import BaseModel,Field
import pandas as pd
import numpy as np
import uvicorn


# ==============================
# Create FastAPI Application
# ==============================

app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


# ==============================
# Load and Prepare Dataset
# ==============================

def load_data():
    """
    Load Maths.csv and apply the same feature engineering
    used in analysis.py.
    """

    # Load dataset from the data folder
    data = pd.read_csv("data/Maths.csv")

    # ------------------------------
    # Feature Engineering
    # ------------------------------

    # Result: Pass if G3 > 0, otherwise Dropout
    data["Result"] = np.where(data["G3"] > 0, "Pass", "Dropout")

    # Percentage based on final grade
    data["Percentage"] = (data["G3"] / 20) * 100

    # Average alcohol consumption
    data["avg_alcohol"] = (
        data["Dalc"] + data["Walc"]
    ) / 2

    # Average parent education level
    data["parent_edu_avg"] = (
        data["Medu"] + data["Fedu"]
    ) / 2

    # Grade trend from G1 to G2 to G3
    data["grade_trend"] = (
        data["G3"] - data["G1"]
    )

    # Total support received by the student
    data["total_support"] = (
        data["famsup"].map({"yes": 1, "no": 0}) +
        data["schoolsup"].map({"yes": 1, "no": 0}) +
        data["paid"].map({"yes": 1, "no": 0})
    )

    # Risk score based on academic and attendance factors
    data["risk_score"] = (
        (20 - data["G3"]) +
        data["absences"]
    )

    # Average of first and second period grades
    data["g1_g2_avg"] = (
        data["G1"] + data["G2"]
    ) / 2

    # Return prepared DataFrame
    return data


# ==============================
# Load Data at Startup
# ==============================

df = load_data()


# ============================================================
# Endpoint 1: GET /summary
# ============================================================

@app.get("/summary")
def get_summary():
    """
    Return overall academic summary.

    Calculations for class average and pass rate
    consider non-dropout students only.
    """

    # Total number of students
    total_students = len(df)

    # Separate non-dropout students
    non_dropout = df[df["G3"] > 0]

    # Calculate class average G3
    class_average_g3 = round(
        float(non_dropout["G3"].mean()), 2
    )

    # Calculate pass rate among all students
    pass_rate_percent = round(
        float((len(non_dropout) / total_students) * 100), 2
    )

    # Count students at risk: G3 between 1 and 9
    at_risk_count = int(
        ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()
    )

    # Count dropouts: G3 = 0
    dropout_count = int(
        (df["G3"] == 0).sum()
    )

    return {
        "total_students": int(total_students),
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate_percent,
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count
    }


# ============================================================
# Endpoint 2: GET /at-risk
# ============================================================

@app.get("/at-risk")
def get_at_risk_students():
    """
    Return students whose G3 score is between 1 and 9.

    Students are sorted by G3 in ascending order,
    so the lowest-performing students appear first.
    """

    # Filter students with G3 between 1 and 9
    at_risk = df[
        (df["G3"] >= 1) &
        (df["G3"] <= 9)
    ].copy()

    # Sort by G3 ascending
    at_risk = at_risk.sort_values(
        by="G3",
        ascending=True
    )

    # Return only the requested fields
    return at_risk[
        ["student_index", "G1", "G2", "G3", "absences"]
    ].to_dict(orient="records")


# ============================================================
# Endpoint 3: GET /top-students
# ============================================================

@app.get("/top-students")
def get_top_students():
    """
    Return the top 5 students based on G3.

    Dropout students (G3 = 0) are excluded.
    Students are sorted by G3 in descending order.
    """

    # Exclude dropout students
    non_dropout = df[df["G3"] > 0].copy()

    # Sort by G3 from highest to lowest
    top_students = non_dropout.sort_values(
        by="G3",
        ascending=False
    ).head(5)

    # Return only the requested fields
    return top_students[
        ["student_index", "G1", "G2", "G3"]
    ].to_dict(orient="records")

# ==============================
# Pydantic Model for Student Input
# ==============================

class StudentInput(BaseModel):
    """
    Input model for predicting a student's academic result.
    """

    # First period grade: must be between 0 and 20
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="G1 must be between 0 and 20"
    )

    # Second period grade: must be between 0 and 20
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="G2 must be between 0 and 20"
    )

    # Weekly study time: must be between 1 and 4
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Study time must be between 1 and 4"
    )

    # Number of absences: must be between 0 and 100
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Absences must be between 0 and 100"
    )

    # Number of previous failures: must be between 0 and 4
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Failures must be between 0 and 4"
    )


# ============================================================
# Endpoint: POST /predict-result
# ============================================================

@app.post("/predict-result")
def predict_result(student: StudentInput):
    """
    Estimate the student's G3 score and predict their result.
    """

    # --------------------------------
    # Calculate Estimated G3
    # --------------------------------

    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp estimated G3 between 0 and 20
    estimated_g3 = np.clip(estimated_g3, 0, 20)

    # Round estimated G3 to 2 decimal places
    estimated_g3 = round(float(estimated_g3), 2)

    # --------------------------------
    # Determine Prediction
    # --------------------------------

    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # --------------------------------
    # Determine Confidence
    # --------------------------------

    # High confidence when both grades are above 12
    # OR both grades are below 8
    if (
        (student.G1 > 12 and student.G2 > 12)
        or
        (student.G1 < 8 and student.G2 < 8)
    ):
        confidence = "High"
    else:
        confidence = "Medium"

    # --------------------------------
    # Return Prediction Result
    # --------------------------------

    return {
        "estimated_g3": estimated_g3,
        "prediction": prediction,
        "confidence": confidence
    }
# ============================================================
# Root Endpoint: GET /
# ============================================================

@app.get("/")
def root():
    """
    Return basic information about the API.
    """

    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# ============================================================
# Uvicorn Runner
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )