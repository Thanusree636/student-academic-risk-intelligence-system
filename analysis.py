import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# ============================================================
# PROMPT 1 - Load Data and Feature Engineering
# ============================================================

def load_and_prepare_data(filepath):
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Create Result column based on final grade G3
    # G3 = 0     -> Dropout
    # G3 = 1-9   -> Fail
    # G3 = 10-20 -> Pass
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout" if x == 0
        else "Fail" if x <= 9
        else "Pass"
    )

    # Convert G3 into percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average parent education
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values in school support, family support and paid classes
    support_columns = ["schoolsup", "famsup", "paid"]
    df["total_support"] = (
        df[support_columns] == "yes"
    ).sum(axis=1)

    # Calculate risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the complete prepared DataFrame
    return df


# ============================================================
# PROMPT 2 - NumPy Analysis
# ============================================================

def calculate_statistics(df):
    # Exclude dropout students from academic calculations
    non_dropout = df[df["G3"] != 0]

    # Calculate class average G3 excluding dropouts
    class_avg_g3 = np.mean(non_dropout["G3"])

    # Count students who passed
    pass_count = np.sum(non_dropout["G3"] >= 10)

    # Calculate pass rate among non-dropout students
    pass_rate = (pass_count / len(non_dropout)) * 100

    # Count dropout students
    dropout_count = np.sum(df["G3"] == 0)

    # Count at-risk students
    # At-risk means G3 is between 1 and 9
    at_risk_count = np.sum(
        (df["G3"] >= 1) & (df["G3"] <= 9)
    )

    # Calculate correlation matrix for G1, G2 and G3
    # Dropout students are excluded
    correlation_matrix = np.corrcoef(
        non_dropout[["G1", "G2", "G3"]].values.T
    )

    # Return all statistics as a dictionary
    return {
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": dropout_count,
        "at_risk_count": at_risk_count,
        "correlation_matrix": correlation_matrix
    }


# ============================================================
# PROMPT 3 - Matplotlib Static Charts
# ============================================================

def generate_static_charts(df):
    # Create output folder if it does not exist
    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------
    # Chart 1 - Average G3 by Study Time
    # --------------------------------------------------------

    # Calculate average G3 for each studytime level
    studytime_avg = df.groupby("studytime")["G3"].mean()

    # Create bar chart
    plt.figure(figsize=(8, 5))

    plt.bar(
        studytime_avg.index,
        studytime_avg.values
    )

    # Add title and labels
    plt.title("Average G3 by Study Time")
    plt.xlabel(
        "Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)"
    )
    plt.ylabel("Average G3")

    # Make sure all studytime levels appear on X-axis
    plt.xticks([1, 2, 3, 4])

    # Save the chart
    plt.savefig(
        "output/avg_g3_by_studytime.png"
    )

    # Close the chart
    plt.close()

    # --------------------------------------------------------
    # Chart 2 - Student Result Distribution
    # --------------------------------------------------------

    # Count Pass, Fail and Dropout students
    result_counts = df["Result"].value_counts()

    # Create pie chart
    plt.figure(figsize=(7, 7))

    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%"
    )

    # Add title
    plt.title("Student Result Distribution")

    # Save the chart
    plt.savefig(
        "output/pass_fail_dropout_pie.png"
    )

    # Close the chart
    plt.close()


# ============================================================
# PROMPT 4 - Plotly Interactive Charts
# ============================================================

def generate_interactive_charts(df):

    # --------------------------------------------------------
    # Chart 1 - Study Time vs Final Grade
    # --------------------------------------------------------

    # Create interactive scatter plot
    fig1 = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade (G3)",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        }
    )

    # Display scatter plot
    fig1.show()

    # --------------------------------------------------------
    # Chart 2 - Average G3 by Internet Access
    # --------------------------------------------------------

    # Calculate average G3 for each internet group
    internet_avg = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
    )

    # Create interactive bar chart
    fig2 = px.bar(
        internet_avg,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display bar chart
    fig2.show()


# ============================================================
# PROMPT 5 - Summary Table
# ============================================================

def print_summary(stats):
    # Calculate total students from the statistics
    # Note: total_students is not stored in stats by Prompt 2,
    # so it will be provided separately in the main block.
    print("=" * 50)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"Total Students       : {stats['total_students']}")
    print(
        f"Class Average G3     : "
        f"{stats['class_avg_g3']:.2f}"
    )
    print(
        f"Pass Rate            : "
        f"{stats['pass_rate']:.2f}%"
    )
    print(
        f"At-Risk Count        : "
        f"{stats['at_risk_count']}"
    )
    print(
        f"Dropout Count        : "
        f"{stats['dropout_count']}"
    )

    print("=" * 50)


# ============================================================
# MAIN BLOCK - PROMPT 5
# ============================================================

if __name__ == "__main__":

    # Load and prepare the dataset
    df = load_and_prepare_data("data/Maths.csv")

    # Calculate statistics
    stats = calculate_statistics(df)

    # Add total students for the summary
    stats["total_students"] = len(df)

    # Generate static Matplotlib charts
    generate_static_charts(df)

    # Generate interactive Plotly charts
    generate_interactive_charts(df)

    # Print analysis summary
    print_summary(stats)

    # Display completion message
    print("Analysis complete. Charts saved to output/ folder")