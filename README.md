# MindSync AI

A web application that analyzes daily lifestyle and smartphone usage patterns to predict digital wellbeing risk using machine learning.

## What it does

Users enter their daily habits — screen time, sleep, study hours, social media usage, and more — and the app predicts whether they fall into a **Low Risk** or **High Risk** digital wellbeing category, along with personalized recommendations.

## Tech Stack

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Frontend:** Bootstrap 5, HTML, Jinja2

## Project Structure

```
MindSync-AI/
├── app.py                  # Flask app, routes, HTML templates, prediction logic
├── model_training.py       # End-to-end training pipeline
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── raw/                # Source CSVs (do not modify)
│   │   ├── mental_health_and_technology_usage_2024.csv
│   │   ├── sleep_mobile_stress_dataset_15000.csv
│   │   └── user_behavior_dataset.csv
│   └── processed/          # Auto-generated during training
│
├── trained_models/
│   ├── classifier_model.pkl  # Trained sklearn pipeline
│   └── features.json         # Feature order used during training
│
└── tests/
    └── test_app.py
```

## Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the model**
```bash
python model_training.py
```
This reads all CSVs from `datasets/raw/`, trains a Logistic Regression pipeline, and saves the model and feature schema to `trained_models/`.

**3. Run the app**
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

**4. Run tests**
```bash
python -m pytest tests/test_app.py -v
```

## How it works

### Training (`model_training.py`)
- Loads and merges all raw CSVs
- Normalizes column names across datasets
- Builds a consistent feature set, filling missing columns with 0
- Derives `usage_ratio` = `social_media_usage / screen_time`
- Trains a pipeline: `SimpleImputer → StandardScaler → LogisticRegression`
- Saves `classifier_model.pkl` and `features.json`

### Inference (`app.py`)
- User submits daily usage values via the web form
- App validates that total hour-based fields don't exceed 24
- Builds a DataFrame aligned to `features.json` column order
- Runs prediction → returns Low Risk or High Risk
- Generates rule-based personalized recommendations

## Input Fields

| Field | Unit |
|---|---|
| Daily Screen Time | hours/day |
| Social Media Usage | hours/day |
| Sleep Hours | hours/day |
| Study / Productivity Hours | hours/day |
| Exercise Time | hours/day |
| Notifications | count/day |
| App Unlocks | count/day |
| Late Night Usage | ratio 0–1 |
| Mood Level | 1–10 scale |
