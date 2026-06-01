# MindSync AI

An academic web application that analyzes daily lifestyle and smartphone usage patterns to predict sleep and stress risk using machine learning.

## What it does

Users enter their daily habits — screen time, sleep, study hours, social media usage, and more — and the app predicts whether they fall into a **Low Risk** or **High Risk** category for sleep and stress, along with personalized recommendations.

## Tech Stack

- **Backend:** Flask
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Frontend:** Bootstrap 5, Jinja2
- **Server:** Gunicorn (production)

## Project Structure

```
MindSync-AI/
├── app.py                  # Flask app, routes, templates, prediction logic
├── model_training.py       # Training pipeline
├── requirements.txt
├── README.md
│
├── datasets/
│   └── raw/
│       ├── mental_health_and_technology_usage_2024.csv
│       └── sleep_mobile_stress_dataset_15000.csv
│
├── trained_models/
│   ├── sleep_stress_model.pkl       # Trained sklearn pipeline (94% accuracy)
│   └── sleep_stress_features.json   # Feature schema used at inference
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
Trains the sleep/stress model and saves artifacts to `trained_models/`.

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

### Sleep & Stress Risk (ML model)
A logistic regression pipeline (`SimpleImputer → StandardScaler → LogisticRegression`) trained on `sleep_mobile_stress_dataset_15000.csv` (15,000 records, 94% accuracy).

| App field | Model feature |
|---|---|
| screen_time | daily_screen_time_hours |
| sleep_hours | sleep_duration_hours |
| exercise_time | physical_activity_hours |
| notifications | notifications_received_per_day |
| late_night_usage | late_night_ratio |
| mood_level (inverted) | mental_fatigue_score |

### Training (`model_training.py`)
- Loads and preprocesses `sleep_mobile_stress_dataset_15000.csv`
- Normalizes units (exercise minutes → hours, late night minutes → 0–1 ratio)
- Inverts mood level to derive mental fatigue proxy
- Trains with `class_weight="balanced"` to handle class imbalance
- Saves model artifact and feature schema to `trained_models/`

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

## License

MIT License