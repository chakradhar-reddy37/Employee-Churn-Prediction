"""
Employee Churn Prediction

IBM HR Analytics-style employee attrition classifier demonstrating:
- Data preprocessing
- Categorical encoding
- Feature engineering
- Class-imbalance handling
- Random Forest classification
- Threshold tuning
- Accuracy/recall evaluation
- Matplotlib/Seaborn result visualization

Expected input:
    WA_Fn-UseC_-HR-Employee-Attrition.csv

Install:
    pip install pandas numpy scikit-learn matplotlib seaborn

Usage:
    python "Employee Churn Prediction.py" WA_Fn-UseC_-HR-Employee-Attrition.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Attrition"


def load_data(path):
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    return df


def engineer_features(df):
    result = df.copy()

    # Convert target to binary.
    result[TARGET] = result[TARGET].map({"Yes": 1, "No": 0})

    # Useful interaction/derived features.
    if {"MonthlyIncome", "JobLevel"}.issubset(result.columns):
        result["IncomePerJobLevel"] = (
            result["MonthlyIncome"] / result["JobLevel"].replace(0, 1)
        )

    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(result.columns):
        result["CompanyTenureRatio"] = (
            result["YearsAtCompany"]
            / result["TotalWorkingYears"].replace(0, 1)
        )

    if {"YearsInCurrentRole", "YearsAtCompany"}.issubset(result.columns):
        result["RoleTenureRatio"] = (
            result["YearsInCurrentRole"]
            / result["YearsAtCompany"].replace(0, 1)
        )

    # Remove identifiers that do not provide useful predictive signal.
    for column in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
        if column in result.columns:
            result = result.drop(columns=column)

    return result


def build_pipeline(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def tune_threshold(y_true, probabilities, minimum_recall=0.70):
    """
    Select the highest precision threshold that still satisfies
    the requested minimum recall, when possible.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    valid = [
        (p, r, t)
        for p, r, t in zip(
            precision[:-1],
            recall[:-1],
            thresholds,
        )
        if r >= minimum_recall
    ]

    if not valid:
        return 0.50

    return max(valid, key=lambda item: item[0])[2]


def plot_confusion_matrix(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Employee Attrition Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python "
            '"Employee Churn Prediction.py" dataset.csv'
        )
        return

    df = load_data(sys.argv[1])
    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    threshold = tune_threshold(
        y_test,
        probabilities,
        minimum_recall=0.70,
    )

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f"Optimized threshold: {threshold:.3f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall:   {recall:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Stayed", "Left"],
        )
    )

    plot_confusion_matrix(y_test, predictions)


if __name__ == "__main__":
    main()
