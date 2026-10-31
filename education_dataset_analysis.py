import os
import re
import json
import zipfile
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# STEP 1: Load all files from the education dataset ZIP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, "track4_education_dataset_files.zip")

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"Cannot find {ZIP_PATH}. Keep this Python file in the same folder "
        "as track4_education_dataset_files.zip, or update ZIP_PATH."
    )

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    school_master = pd.read_csv(z.open("track4_school_master.csv"))
    attendance = pd.read_csv(z.open("track4_student_attendance.csv"))
    test_scores = pd.DataFrame(json.load(z.open("track4_test_scores.json")))
    infrastructure = pd.read_csv(z.open("track4_school_infrastructure.csv"))
    with z.open("track4_mid_day_meal_procurement.xlsx") as f:
        mdm = pd.read_excel(BytesIO(f.read()), sheet_name="mdm_procurement")

print("STEP 1: DATASETS LOADED")
for name, df in {
    "School Master": school_master,
    "Attendance": attendance,
    "Test Scores": test_scores,
    "Infrastructure": infrastructure,
    "Mid-Day Meal Procurement": mdm
}.items():
    print(f"{name}: {df.shape}")

# STEP 2: Inspect the data
print("\nSTEP 2: DATASET OVERVIEW")
for name, df in {
    "School Master": school_master,
    "Attendance": attendance,
    "Test Scores": test_scores,
    "Infrastructure": infrastructure,
    "Mid-Day Meal Procurement": mdm
}.items():
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("Total missing values:", int(df.isnull().sum().sum()))

# STEP 3: Standardize school IDs and common text values
def clean_school_id(value):
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return np.nan
    return "SCH" + digits.zfill(4)

def clean_text(series):
    return series.astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA}
    )

for df in [school_master, attendance, test_scores, infrastructure, mdm]:
    if "school_id" in df.columns:
        df["school_id"] = df["school_id"].apply(clean_school_id)

for df in [school_master, attendance, test_scores, infrastructure, mdm]:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col != "school_id":
            df[col] = clean_text(df[col])

# Standardize school master text categories
for col in ["district", "block", "school_type", "medium"]:
    if col in school_master.columns:
        school_master[col] = school_master[col].str.strip().str.title()

print("\nSTEP 3: SCHOOL IDS STANDARDIZED")
print("Unique schools in master data:", school_master["school_id"].nunique())
print("Duplicate school IDs in master:", int(school_master["school_id"].duplicated().sum()))

# Keep one master record per school ID if duplicates exist
school_master = school_master.drop_duplicates(subset="school_id", keep="first")

# STEP 4: Clean attendance data and identify anomalies
attendance["date"] = pd.to_datetime(attendance["date"], errors="coerce", format="mixed")
attendance["total_students"] = pd.to_numeric(attendance["total_students"], errors="coerce")
attendance["present_students"] = pd.to_numeric(attendance["present_students"], errors="coerce")

attendance["impossible_attendance"] = (
    attendance["present_students"] > attendance["total_students"]
)
attendance["attendance_rate"] = np.where(
    attendance["total_students"] > 0,
    attendance["present_students"] / attendance["total_students"] * 100,
    np.nan
)

attendance["is_sunday"] = attendance["date"].dt.dayofweek.eq(6)
attendance["full_attendance"] = (
    attendance["total_students"].gt(0)
    & attendance["present_students"].eq(attendance["total_students"])
)
attendance["proxy_attendance_flag"] = (
    attendance["is_sunday"] & attendance["full_attendance"]
)

print("\nSTEP 4: ATTENDANCE QUALITY CHECK")
print("Invalid/unparsed dates:", int(attendance["date"].isna().sum()))
print("Impossible records (present > total):", int(attendance["impossible_attendance"].sum()))
print("100% attendance on Sundays (proxy flag):", int(attendance["proxy_attendance_flag"].sum()))
print("Average recorded attendance rate (%):",
      round(attendance.loc[~attendance["impossible_attendance"], "attendance_rate"].mean(), 2))

# Do not use impossible attendance records for attendance-rate analysis
attendance_valid = attendance[
    attendance["date"].notna()
    & attendance["total_students"].gt(0)
    & attendance["present_students"].ge(0)
    & ~attendance["impossible_attendance"]
].copy()

# STEP 5: Clean infrastructure boolean fields
def normalize_boolean(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    true_values = {"yes", "y", "1", "true", "hai", "haan", "functional", "working", "available"}
    false_values = {"no", "n", "0", "false", "nahi", "kharab", "not functional", "non-functional"}
    if value in true_values:
        return 1
    if value in false_values:
        return 0
    return np.nan

infra_cols = [
    "has_electricity", "has_drinking_water", "has_functional_toilet",
    "has_boundary_wall", "has_playground"
]
for col in infra_cols:
    if col in infrastructure.columns:
        infrastructure[col] = infrastructure[col].apply(normalize_boolean)

infrastructure["date"] = pd.to_datetime(infrastructure["date"], errors="coerce", format="mixed")
infra_school = infrastructure.groupby("school_id", dropna=False)[infra_cols].mean().reset_index()
infra_school = infra_school.rename(columns={c: c + "_rate" for c in infra_cols})

print("\nSTEP 5: INFRASTRUCTURE CLEANING")
print(infrastructure[infra_cols].mean().mul(100).round(2).rename("Available (%)"))

# STEP 6: Clean mid-day meal procurement data
def normalize_grain(value):
    if pd.isna(value):
        return np.nan
    v = re.sub(r"[^a-z]", "", str(value).lower())
    if any(k in v for k in ["rice", "chawal", "chaval"]):
        return "Rice"
    if any(k in v for k in ["wheat", "gehun", "gehu"]):
        return "Wheat"
    if any(k in v for k in ["dal", "pulse", "lentil"]):
        return "Pulses"
    if any(k in v for k in ["oil", "tel"]):
        return "Oil"
    return str(value).strip().title()

def parse_quantity_kg(quantity, unit):
    if pd.isna(quantity):
        return np.nan
    raw = str(quantity).strip().lower().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan
    amount = float(match.group())
    unit_text = ("" if pd.isna(unit) else str(unit)).strip().lower()
    combined = raw + " " + unit_text
    if any(x in combined for x in ["gram", " g "]):
        return amount / 1000
    if any(x in combined for x in ["bag", "sack", "bori"]):
        return amount * 50
    # Default to kg when unit is blank or is kg/kgs/kilogram.
    if not unit_text or any(x in unit_text for x in ["kg", "kilo"]):
        return amount
    return amount

def parse_cost(value):
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9.\-]", "", str(value).replace(",", ""))
    try:
        return float(cleaned) if cleaned not in {"", ".", "-", "-."} else np.nan
    except ValueError:
        return np.nan

mdm["grain_type_clean"] = mdm["grain_type"].apply(normalize_grain)
mdm["quantity_kg"] = [
    parse_quantity_kg(q, u) for q, u in zip(mdm["quantity"], mdm["unit"])
]
mdm["total_cost_clean"] = mdm["total_cost"].apply(parse_cost)
mdm["date"] = pd.to_datetime(mdm["date"], errors="coerce", format="mixed")
mdm["vendor_name_clean"] = (
    mdm["vendor_name"].astype("string").str.lower().str.strip()
    .str.replace(r"\b(pvt|private|limited|ltd|llp)\b", "", regex=True)
    .str.replace(r"[^a-z0-9 ]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True).str.strip()
)
mdm["payment_status_clean"] = mdm["payment_status"].astype("string").str.lower().str.strip()

print("\nSTEP 6: MID-DAY MEAL CLEANING")
print("Missing/invalid quantity (kg):", int(mdm["quantity_kg"].isna().sum()))
print("Missing/invalid cost:", int(mdm["total_cost_clean"].isna().sum()))
print("Total cleaned procurement cost:", round(mdm["total_cost_clean"].sum(skipna=True), 2))
print("Total recorded quantity (kg):", round(mdm["quantity_kg"].sum(skipna=True), 2))

# STEP 7: Standardize test scores into percentages
def score_to_percentage(row):
    score = row.get("avg_score", np.nan)
    scale = str(row.get("grading_scale", "")).strip().lower()
    max_marks = pd.to_numeric(row.get("max_marks", np.nan), errors="coerce")

    if pd.isna(score):
        return np.nan
    s = str(score).strip().upper()

    # Percentage values
    if "%" in s or "percent" in scale or scale == "pct":
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else np.nan

    # Raw marks such as 21.6/50
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        obtained, total = float(m.group(1)), float(m.group(2))
        return obtained / total * 100 if total > 0 else np.nan

    # Letter-grade mapping based on dataset task notes
    letter_map = {
        "A+": 95, "A": 85, "A-": 80, "B+": 78, "B": 75,
        "B-": 70, "C+": 68, "C": 65, "C-": 60,
        "D+": 58, "D": 55, "D-": 50, "E": 40, "F": 30
    }
    if s in letter_map:
        return float(letter_map[s])

    # Numeric scores with CGPA scale
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    numeric = float(m.group())
    if "cgpa" in scale:
        return numeric * 9.5 if numeric <= 10 else np.nan
    if pd.notna(max_marks) and max_marks > 0:
        return numeric / max_marks * 100
    return numeric if 0 <= numeric <= 100 else np.nan

test_scores["score_percentage"] = test_scores.apply(score_to_percentage, axis=1)
test_scores["date"] = pd.to_datetime(test_scores["date"], errors="coerce", format="mixed")
test_scores["subject_clean"] = (
    test_scores["subject"].astype("string").str.lower().str.strip()
    .replace({
        "math": "Mathematics", "maths": "Mathematics", "ganit": "Mathematics",
        "sci": "Science", "english": "English", "punjabi": "Punjabi"
    })
    .str.title()
)
test_scores["subject_clean"] = test_scores["subject_clean"].replace(
    {"Math": "Mathematics", "Maths": "Mathematics", "Ganit": "Mathematics"}
)

print("\nSTEP 7: TEST SCORE CLEANING")
print("Scores converted to percentages:", int(test_scores["score_percentage"].notna().sum()))
print("Scores needing review / missing:", int(test_scores["score_percentage"].isna().sum()))
print(test_scores[["grading_scale", "avg_score", "score_percentage"]].head(10))

# STEP 8: Visualize the education dataset
plt.figure(figsize=(7, 4))
sns.histplot(attendance_valid["attendance_rate"].clip(0, 100), bins=20, kde=True, color="steelblue")
plt.title("Distribution of Student Attendance Rates")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

monthly_attendance = (
    attendance_valid.assign(month=attendance_valid["date"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)["attendance_rate"].mean()
)
plt.figure(figsize=(10, 4))
sns.lineplot(data=monthly_attendance, x="month", y="attendance_rate", marker="o", color="steelblue")
plt.title("Monthly Average Student Attendance")
plt.xlabel("Month")
plt.ylabel("Average Attendance Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

district_attendance = attendance_valid.merge(
    school_master[["school_id", "district"]], on="school_id", how="left"
)
district_attendance = district_attendance.groupby("district", as_index=False)["attendance_rate"].mean()
district_attendance = district_attendance.dropna().sort_values("attendance_rate", ascending=False).head(15)
plt.figure(figsize=(10, 5))
sns.barplot(data=district_attendance, x="attendance_rate", y="district", color="steelblue")
plt.title("Average Attendance Rate by District (Top 15)")
plt.xlabel("Average Attendance Rate (%)")
plt.ylabel("District")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
score_plot = test_scores.dropna(subset=["score_percentage"])
if not score_plot.empty:
    sns.histplot(data=score_plot, x="score_percentage", bins=15, kde=True, color="steelblue")
    plt.title("Distribution of Standardized Test Scores")
    plt.xlabel("Score (%)")
    plt.ylabel("Number of Assessments")
    plt.tight_layout()
    plt.show()
else:
    print("No valid standardized test scores were available to plot.")

plt.figure(figsize=(8, 4))
infra_percent = infrastructure[infra_cols].mean().mul(100).dropna().sort_values()
if not infra_percent.empty:
    sns.barplot(x=infra_percent.values, y=infra_percent.index, color="steelblue")
    plt.title("School Infrastructure Availability")
    plt.xlabel("Records marked available (%)")
    plt.ylabel("Infrastructure Facility")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 4))
grain_summary = mdm.groupby("grain_type_clean", dropna=False)["quantity_kg"].sum().sort_values(ascending=False).head(10)
if not grain_summary.empty:
    sns.barplot(x=grain_summary.values, y=grain_summary.index, color="steelblue")
    plt.title("Total Mid-Day Meal Procurement by Grain")
    plt.xlabel("Quantity (kg)")
    plt.ylabel("Grain Type")
    plt.tight_layout()
    plt.show()

# STEP 9: Create school-level modelling dataset
# Target: High attendance if the school's mean valid recorded attendance is at or above the median across schools.
# Attendance counts/rates are used only to create the target, not as model features.
school_attendance = attendance_valid.groupby("school_id", as_index=False).agg(
    mean_attendance_rate=("attendance_rate", "mean"),
    attendance_records=("attendance_rate", "count")
)
attendance_threshold = school_attendance["mean_attendance_rate"].median()
school_attendance["Attendance_Level"] = np.where(
    school_attendance["mean_attendance_rate"] >= attendance_threshold, "High", "Low"
)

school_data = (
    school_master.merge(school_attendance, on="school_id", how="inner")
    .merge(infra_school, on="school_id", how="left")
)

# Exclude identifiers and target-generating attendance measures from predictors
feature_columns = [
    c for c in [
        "district", "block", "total_enrolled_students", "school_type", "medium",
        "has_electricity_rate", "has_drinking_water_rate",
        "has_functional_toilet_rate", "has_boundary_wall_rate", "has_playground_rate"
    ] if c in school_data.columns
]
X = school_data[feature_columns].copy()
# Convert pandas nullable strings to regular Python objects for sklearn imputers.
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
y = school_data["Attendance_Level"].copy()

print("\nSTEP 9: FEATURES AND TARGET")
print("School-level dataset shape:", school_data.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature columns:", X.columns.tolist())
print(f"Attendance threshold (school-level median): {attendance_threshold:.2f}%")
print("Attendance-level distribution:")
print(y.value_counts(dropna=False))

# If the target has only one class, classification cannot be trained.
if y.nunique() < 2:
    print("\nML MODELS NOT TRAINED: Attendance_Level has only one class.")
    print("Review the attendance threshold or the underlying attendance data.")
else:
    # STEP 10: Train-test split and preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    print("\nSTEP 10: TRAIN-TEST SPLIT")
    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Target Shape:", y_train.shape)
    print("Testing Target Shape:", y_test.shape)

    # STEP 11: Train five machine learning algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42)
    }

    trained_models = {}
    results = []

    for name, model in models.items():
        # GaussianNB requires dense data after one-hot encoding.
        if name == "Naive Bayes":
            model_preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), categorical_features)
            ])
        else:
            model_preprocessor = preprocessor

        pipeline = Pipeline(steps=[
            ("preprocessor", model_preprocessor),
            ("classifier", model)
        ])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print(name, "trained successfully!")

    print("\nAll five machine learning models have been trained.")

    # STEP 12: Evaluate the models
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="High", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="High", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="High", zero_division=0)

        results.append({
            "Algorithm": name,
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Recall (%)": round(recall * 100, 2),
            "F1 Score (%)": round(f1 * 100, 2)
        })

        print("\n" + "=" * 50)
        print("Algorithm:", name)
        print("=" * 50)
        print("\nConfusion Matrix (labels: Low, High):")
        print(confusion_matrix(y_test, y_pred, labels=["Low", "High"]))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False)
    print("\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 4))
    results_long = results_df.melt(
        id_vars="Algorithm",
        value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=results_long, x="Algorithm", y="Score", hue="Metric", palette="deep")
    plt.title("Comparison of Machine Learning Models")
    plt.ylabel("Score (%)")
    plt.xlabel("Algorithm")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

# STEP 13: Save cleaned outputs for further analysis/dashboard use
attendance.to_csv(os.path.join(SCRIPT_DIR, "cleaned_attendance_with_flags.csv"), index=False)
school_master.to_csv(os.path.join(SCRIPT_DIR, "cleaned_school_master.csv"), index=False)
mdm.to_csv(os.path.join(SCRIPT_DIR, "cleaned_mid_day_meal_procurement.csv"), index=False)
test_scores.to_csv(os.path.join(SCRIPT_DIR, "cleaned_test_scores.csv"), index=False)
infrastructure.to_csv(os.path.join(SCRIPT_DIR, "cleaned_school_infrastructure.csv"), index=False)
print("\nSTEP 13: Cleaned datasets saved as CSV files.")
