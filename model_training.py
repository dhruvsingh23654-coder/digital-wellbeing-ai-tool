from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_and_save(X, y, feature_cols, model_filename, features_filename, label):
    print(f"\n{'='*50}")
    print(f"Training: {label}")
    print(f"Rows: {len(X)} | Features: {feature_cols}")
    print(f"Class distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = make_pipeline()
    model.fit(X_train, y_train)

    print(f"\nEvaluation ({label}):")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    joblib.dump(model, MODEL_DIR / model_filename)
    (MODEL_DIR / features_filename).write_text(json.dumps(feature_cols, indent=2))
    print(f"Saved: {model_filename}, {features_filename}")

def train_sleep_stress():
    df = pd.read_csv(RAW_DIR / "sleep_mobile_stress_dataset_15000.csv")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["physical_activity_hours"] = df["physical_activity_minutes"] / 60

    df["late_night_ratio"] = (df["phone_usage_before_sleep_minutes"] / 120).clip(0, 1)

    feature_cols = [
        "daily_screen_time_hours",
        "sleep_duration_hours",
        "physical_activity_hours",
        "notifications_received_per_day",
        "late_night_ratio",
        "mental_fatigue_score",
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    median_stress = df["stress_level"].median()
    y = (df["stress_level"] > median_stress).astype(int)

    train_and_save(
        X, y,
        feature_cols=feature_cols,
        model_filename="sleep_stress_model.pkl",
        features_filename="sleep_stress_features.json",
        label="Sleep & Stress Model"
    )

if __name__ == "__main__":
    train_sleep_stress()
    print("\nModel trained and saved.")
