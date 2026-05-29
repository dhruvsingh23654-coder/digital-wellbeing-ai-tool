from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"
FEATURES_FILE = MODEL_DIR / "features.json"
MODEL_FILE = MODEL_DIR / "classifier_model.pkl"

app = Flask(__name__)

FIELD_CONFIG = [
    {"name": "screen_time", "label": "Daily Screen Time", "unit": "hours/day", "help": "Total phone usage in a day.", "step": "0.1", "min": "0", "max": "24", "placeholder": "6.5"},
    {"name": "social_media_usage", "label": "Social Media Usage", "unit": "hours/day", "help": "Time spent on social apps.", "step": "0.1", "min": "0", "max": "24", "placeholder": "2.0"},
    {"name": "sleep_hours", "label": "Sleep Hours", "unit": "hours/day", "help": "Average sleep duration.", "step": "0.1", "min": "0", "max": "24", "placeholder": "7.5"},
    {"name": "study_hours", "label": "Study / Productivity Hours", "unit": "hours/day", "help": "Time spent on study or work.", "step": "0.1", "min": "0", "max": "24", "placeholder": "4.0"},
    {"name": "exercise_time", "label": "Exercise Time", "unit": "hours/day", "help": "Physical activity per day.", "step": "0.1", "min": "0", "max": "24", "placeholder": "0.5"},
    {"name": "notifications", "label": "Notifications", "unit": "count/day", "help": "Approximate alerts received.", "step": "1", "min": "0", "max": "500", "placeholder": "60"},
    {"name": "app_unlocks", "label": "App Unlocks", "unit": "count/day", "help": "Phone unlock frequency.", "step": "1", "min": "0", "max": "500", "placeholder": "80"},
    {"name": "late_night_usage", "label": "Late Night Usage", "unit": "ratio 0-1", "help": "Usage after 11 PM.", "step": "0.1", "min": "0", "max": "1", "placeholder": "0.2"},
    {"name": "mood_level", "label": "Mood Level", "unit": "1-10 scale", "help": "Self-reported mood score.", "step": "0.1", "min": "1", "max": "10", "placeholder": "6.0"},
]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run training first.")
    return joblib.load(MODEL_FILE)


def load_features():
    if FEATURES_FILE.exists():
        return json.loads(FEATURES_FILE.read_text())
    return [f["name"] for f in FIELD_CONFIG] + ["usage_ratio"]


BASE_HTML_START = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MindSync</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background:#f6f8fb; color:#2b2f36; font-size:0.9rem; }
    .navbar { background:#22313f !important; }
    .brand { font-weight:600; letter-spacing:0.2px; }
    .card { border:1px solid #e6eaf0; border-radius:12px; }
    .page-title { font-size:1.35rem; font-weight:600; margin-bottom:0.25rem; }
    .subtle { color:#6c757d; font-size:0.88rem; }
    .form-label { font-size:0.85rem; font-weight:600; margin-bottom:0.25rem; }
    .form-text { font-size:0.78rem; color:#6c757d; }
    .form-control { font-size:0.88rem; padding:0.6rem 0.75rem; }
    .badge-soft { background:#f0f6f2; color:#1f6f5a; border:1px solid #d9efe6; font-weight:500; }
    .result-box { background:#fbfcfe; border:1px solid #e6eaf0; border-radius:10px; padding:0.9rem; }
    .rec-list { margin:0; padding-left:1.1rem; color:#2b2f36; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark mb-4">
  <div class="container">
    <a class="navbar-brand brand" href="/">MindSync</a>
    <div class="navbar-nav ms-auto"><a class="nav-link" href="/predict">Predict</a></div>
  </div>
</nav>
"""


HOME_HTML = BASE_HTML_START + """
<div class="container pb-5">
  <div class="row justify-content-center">
    <div class="col-lg-9">
      <div class="card shadow-sm">
        <div class="card-body p-4 p-md-5">
          <span class="badge badge-soft mb-3">Digital Wellbeing</span>
          <h1 class="page-title">MindSync</h1>
          <p class="subtle mb-4">Academic tool for simple lifestyle and phone-usage guidance.</p>
          <div class="row g-3 mb-4">
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Wellbeing</div><div class="subtle">Estimated from inputs.</div></div></div>
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Risk</div><div class="subtle">Balanced vs elevated usage.</div></div></div>
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Productivity</div><div class="subtle">Basic productivity estimate.</div></div></div>
          </div>
          <a class="btn btn-primary" href="/predict">Start Analysis</a>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""


FORM_HTML = BASE_HTML_START + """
<div class="container pb-5">
  <div class="row justify-content-center">
    <div class="col-lg-10">
      <div class="card shadow-sm">
        <div class="card-body p-4 p-md-5">
          <div class="mb-4">
            <span class="badge badge-soft mb-2">Input Form</span>
            <h2 class="page-title">Predict Wellbeing</h2>
            <p class="subtle mb-0">Enter daily values to get basic wellbeing and risk guidance.</p>
          </div>

          {% if error %}<div class="alert alert-danger py-2">{{ error }}</div>{% endif %}

          <form method="post" class="row g-3">
            {% for field in fields %}
            <div class="col-md-6">
              <label class="form-label">{{ field.label }} <span class="text-muted fw-normal">({{ field.unit }})</span></label>
              <input class="form-control" type="number" name="{{ field.name }}" step="{{ field.step }}" min="{{ field.min }}" max="{{ field.max }}" placeholder="{{ field.placeholder }}" value="{{ request.form.get(field.name, '') }}" required>
              <div class="form-text">{{ field.help }}</div>
            </div>
            {% endfor %}
            <div class="col-12 pt-2">
              <button class="btn btn-success px-4" type="submit">Predict</button>
              <a class="btn btn-outline-secondary ms-2" href="/predict">Reset</a>
            </div>
          </form>

          {% if result %}
          <hr class="my-3">
          <div class="row g-3">
            <div class="col-md-4"><div class="result-box"><div class="subtle mb-1">Predicted Class</div><div class="fs-6 fw-semibold">{{ result }}</div></div></div>
            <div class="col-md-8"><div class="result-box"><div class="subtle mb-1">Label meaning</div><div>Low Risk = balanced usage pattern.<br>High Risk = elevated digital wellbeing concern.</div></div></div>
          </div>

          {% if recommendations %}
          <div class="mt-3">
            <h6 class="mb-2">Personalized recommendations</h6>
            <ul class="rec-list">
              {% for r in recommendations %}<li>{{ r }}</li>{% endfor %}
            </ul>
          </div>
          {% endif %}

          {% endif %}
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""


def compute_recommendations(data: dict, predicted_label: str) -> list:
    recs = []
    st = data.get("screen_time", 0)
    social = data.get("social_media_usage", 0)
    sleep = data.get("sleep_hours", 0)
    study = data.get("study_hours", 0)
    exercise = data.get("exercise_time", 0)
    late = data.get("late_night_usage", 0)
    notif = data.get("notifications", 0)
    unlocks = data.get("app_unlocks", 0)
    mood = data.get("mood_level", 5)

    if st >= 10:
        recs.append("Limit non-essential screen time; set app timers for 30–60 minutes.")
    elif st >= 7:
        recs.append("Introduce short screen-free breaks (20–30 minutes) during the day.")
    else:
        recs.append("Screen time is reasonable; maintain this balance.")

    if late > 0.3:
        recs.append("Avoid screens after 11 PM; enable Do Not Disturb and night mode.")
    if social > study and study > 0:
        recs.append("Use a Pomodoro timer and block social apps during study sessions.")
    elif social > study and study == 0:
        recs.append("Schedule dedicated study/work blocks and limit social app usage.")

    if sleep < 6:
        recs.append("Aim for 7–9 hours of sleep; keep a consistent bedtime.")
    if exercise < 0.5:
        recs.append("Add short daily exercise (20–30 minutes).")
    else:
        recs.append("Good exercise habits — keep them up.")

    if notif > 120:
        recs.append("Turn off non-essential notifications to reduce distractions.")
    if unlocks > 120:
        recs.append("Reduce phone checks by consolidating notifications and using widgets.")

    if mood <= 3:
        recs.append("Low mood noted — consider short mindfulness breaks or professional support.")
    elif mood >= 8:
        recs.append("Mood looks positive — continue current self-care practices.")

    if predicted_label == "High Risk":
        recs.append("Consider a short digital detox and review daily routines.")
    else:
        recs.append("Monitor weekly and make small adjustments if needed.")

    return recs[:6]


@app.route("/")
def home():
    return HOME_HTML


@app.route("/predict", methods=["GET", "POST"])
def predict():
    error = None
    result = None
    recommendations = []
    fields = FIELD_CONFIG
    feature_names = load_features()

    if request.method == "POST":
        try:
            data = {f["name"]: float(request.form.get(f["name"], 0) or 0) for f in fields}
            data["usage_ratio"] = data.get("social_media_usage", 0) / (data.get("screen_time", 1) + 1e-5)
            df = pd.DataFrame([data])
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            model = load_model()
            pred = int(model.predict(df[feature_names])[0])
            result = "High Risk" if pred == 1 else "Low Risk"
            recommendations = compute_recommendations(data, result)
        except FileNotFoundError as e:
            error = str(e)
        except Exception:
            error = "Prediction failed. Ensure models are trained and feature list is correct."

    return render_template_string(FORM_HTML, fields=fields, result=result, error=error, request=request, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)