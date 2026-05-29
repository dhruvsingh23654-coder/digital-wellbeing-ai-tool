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

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "ml" / "datasets" / "raw"
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_all_csvs():
    frames = []
    for file in RAW_DIR.glob("*.csv"):
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    return pd.concat(frames, ignore_index=True, sort=False)

def normalize(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def build_target(df):
    if "target" in df.columns:
        return pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

    if "addiction_risk" in df.columns:
        return pd.to_numeric(df["addiction_risk"], errors="coerce").fillna(0).astype(int)

    screen = pd.to_numeric(df.get("screen_time", df.get("daily_screen_time", 0)), errors="coerce").fillna(0)
    social = pd.to_numeric(df.get("social_media_usage", df.get("social_media_hours", 0)), errors="coerce").fillna(0)
    return ((screen + social) > (screen + social).median()).astype(int)

def build_features(df):
    df = df.copy()
    mapping = {
        "daily_screen_time": "screen_time",
        "screen_time_hours": "screen_time",
        "social_media_hours": "social_media_usage",
        "sleep_duration": "sleep_hours",
        "study_hours": "study_hours",
        "exercise_time": "exercise_time",
        "notifications": "notifications",
        "app_unlocks": "app_unlocks",
        "late_night_usage": "late_night_usage",
        "mood_level": "mood_level",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    for col in [
        "screen_time", "social_media_usage", "sleep_hours", "study_hours",
        "exercise_time", "notifications", "app_unlocks", "late_night_usage", "mood_level"
    ]:
        if col not in df.columns:
            df[col] = 0

    df["usage_ratio"] = df["social_media_usage"] / (df["screen_time"] + 1e-5)
    return df

def main():
    df = normalize(load_all_csvs())
    df = build_features(df)
    y = build_target(df)

    feature_cols = [
        "screen_time", "social_media_usage", "sleep_hours", "study_hours",
        "exercise_time", "notifications", "app_unlocks", "late_night_usage",
        "mood_level", "usage_ratio"
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_DIR / "classifier_model.pkl")
    (MODEL_DIR / "features.json").write_text(json.dumps(feature_cols, indent=2))

    print("Training complete.")
    print(f"Saved: {MODEL_DIR / 'classifier_model.pkl'}")

if __name__ == "__main__":
    main()