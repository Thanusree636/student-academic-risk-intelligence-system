# app.py

# ==============================
# Import Required Libraries
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px


# ==============================
# Streamlit Page Configuration
# ==============================

st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓"
)


# ==============================
# Load and Prepare Dataset
# ==============================

# Load Maths.csv from the data folder
df = pd.read_csv("data/Maths.csv")


# ==============================
# Feature Engineering
# ==============================

# Result: Pass if G3 > 0, otherwise Dropout
df["Result"] = df["G3"].apply(
    lambda x: "Pass" if x > 0 else "Dropout"
)

# Percentage based on final grade
df["Percentage"] = (df["G3"] / 20) * 100

# Average alcohol consumption
df["avg_alcohol"] = (
    df["Dalc"] + df["Walc"]
) / 2

# Average parent education level
df["parent_edu_avg"] = (
    df["Medu"] + df["Fedu"]
) / 2

# Grade trend from G1 to G3
df["grade_trend"] = df["G3"] - df["G1"]

# Total support received by the student
df["total_support"] = (
    df["famsup"].map({"yes": 1, "no": 0})
    + df["schoolsup"].map({"yes": 1, "no": 0})
    + df["paid"].map({"yes": 1, "no": 0})
)

# Risk score based on academic performance and absences
df["risk_score"] = (
    (20 - df["G3"]) + df["absences"]
)

# Average of G1 and G2
df["g1_g2_avg"] = (
    df["G1"] + df["G2"]
) / 2


# ==============================
# Main Dashboard Title
# ==============================

st.title("🎓 Student Academic Risk Intelligence System")


# ==============================
# Calculate KPI Values
# ==============================

# Total number of students
total_students = len(df)

# Non-dropout students
non_dropout = df[df["G3"] > 0]

# Class average G3 excluding dropouts
class_average_g3 = round(
    non_dropout["G3"].mean(), 2
)

# Pass count
pass_count = len(
    non_dropout[non_dropout["G3"] >= 10]
)

# Pass rate percentage
pass_rate = round(
    (pass_count / len(non_dropout)) * 100, 1
)

# At-risk students: G3 between 1 and 9
at_risk_count = len(
    df[(df["G3"] >= 1) & (df["G3"] <= 9)]
)


# ==============================
# KPI Cards - One Row
# ==============================

col1, col2, col3, col4 = st.columns(4)

# Card 1: Total Students
with col1:
    st.metric(
        label="Total Students",
        value=total_students
    )

# Card 2: Class Average G3
with col2:
    st.metric(
        label="Class Average G3",
        value=class_average_g3
    )

# Card 3: Pass Rate
with col3:
    st.metric(
        label="Pass Rate %",
        value=f"{pass_rate}%"
    )

# Card 4: At-Risk Count
with col4:
    st.metric(
        label="At-Risk Count",
        value=at_risk_count
    )

# ============================================================
# Performance Charts
# ============================================================

st.subheader("📊 Performance Charts")

# Create two columns for side-by-side charts
chart_col1, chart_col2 = st.columns(2)


# ------------------------------
# Left Chart: Study Time vs G3
# ------------------------------

with chart_col1:

    # Scatter plot showing relationship between study time and G3
    fig_scatter = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        },
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade"
    )

    # Display scatter plot
    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# ------------------------------
# Right Chart: Internet vs G3
# ------------------------------

with chart_col2:

    # Calculate average G3 grouped by internet access
    internet_avg = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
    )

    # Bar chart showing average G3 by internet access
    fig_bar = px.bar(
        internet_avg,
        x="internet",
        y="G3",
        title="Average G3 by Internet Access"
    )

    # Display bar chart
    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ============================================================
# Student Analysis Table
# ============================================================

st.subheader("🚨 Student Analysis Table")


# ------------------------------
# Result Filter Dropdown
# ------------------------------

result_filter = st.selectbox(
    "Filter by Result",
    ["All", "Pass", "Fail", "Dropout"]
)


# ------------------------------
# Filter DataFrame
# ------------------------------

if result_filter == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["Result"] == result_filter
    ].copy()


# ------------------------------
# Display Filtered Student Table
# ------------------------------

# Select only the requested columns
table_columns = [
    "G1",
    "G2",
    "G3",
    "Result",
    "Percentage",
    "absences",
    "studytime",
    "failures",
    "risk_score"
]

# Display the filtered DataFrame
st.dataframe(
    filtered_df[table_columns],
    use_container_width=True
)


# ============================================================
# At-Risk Students
# ============================================================

st.subheader("⚠️ At-Risk Students")


# ------------------------------
# Filter At-Risk Students
# ------------------------------

# At-risk students have G3 between 1 and 9
at_risk_df = df[
    (df["G3"] >= 1) &
    (df["G3"] <= 9)
].copy()


# Sort by G3 ascending
# Lowest G3 appears first
at_risk_df = at_risk_df.sort_values(
    by="G3",
    ascending=True
)


# ------------------------------
# Display At-Risk Count
# ------------------------------

st.write(
    f"Total at-risk students: {len(at_risk_df)}"
)


# ------------------------------
# Display At-Risk Table
# ------------------------------

at_risk_columns = [
    "G1",
    "G2",
    "G3",
    "absences",
    "studytime",
    "failures"
]

st.dataframe(
    at_risk_df[at_risk_columns],
    use_container_width=True
)