# Education Dataset Analysis

This project cleans and explores synthetic school attendance, test score,
infrastructure, and Mid-Day Meal procurement data. It also trains and compares
five machine-learning classifiers for predicting whether a school's mean valid
attendance is at or above the median across schools.

## What the script does

Run `education_dataset_analysis.py` to:

- Standardize school IDs, text categories, dates, and infrastructure values.
- Flag attendance records with impossible counts and full attendance on Sundays.
- Normalize procurement grain names, quantities, costs, and vendor names.
- Convert mixed test-score formats to percentages.
- Display data summaries and charts.
- Train and evaluate Logistic Regression, K-Nearest Neighbors, Gaussian Naive
  Bayes, Decision Tree, and Support Vector Machine models.
- Save cleaned data as CSV files beside the script.

The dataset is synthetic and intended for educational/datathon use.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- The dataset ZIP file described below

## Dataset setup

Place `track4_education_dataset_files.zip` in the same folder as
`education_dataset_analysis.py`. The ZIP must contain these files at its root:

- `track4_school_master.csv`
- `track4_student_attendance.csv`
- `track4_test_scores.json`
- `track4_school_infrastructure.csv`
- `track4_mid_day_meal_procurement.xlsx`

If you have the extracted `track4_education_dataset_files` folder rather than
the ZIP, open PowerShell in the project folder and create the expected archive:

```powershell
Compress-Archive -Path .\track4_education_dataset_files\* -DestinationPath .\track4_education_dataset_files.zip
```

## Install and run (Windows PowerShell)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\education_dataset_analysis.py
```

The script locates the ZIP relative to its own file, so it can be launched from
another working directory as long as the ZIP remains beside the script. It
displays charts while running and writes these cleaned CSV files beside the
script:

- `cleaned_attendance_with_flags.csv`
- `cleaned_school_master.csv`
- `cleaned_mid_day_meal_procurement.csv`
- `cleaned_test_scores.csv`
- `cleaned_school_infrastructure.csv`

## Notes

- The attendance target is derived from the median school-level mean valid
  attendance rate. Attendance counts and rates are not used as model features.
- Model metrics depend on the input data and the fixed train/test split.
- The project currently loads data from the ZIP archive; the extracted folder
  alone is not read directly.
