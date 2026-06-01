from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"
SLEEP_MODEL = MODEL_DIR / "sleep_stress_model.pkl"
SLEEP_FEATURES = MODEL_DIR / "sleep_stress_features.json"

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

HOUR_FIELDS = ["screen_time", "social_media_usage", "sleep_hours", "study_hours", "exercise_time"]

def load_model():
    if not SLEEP_MODEL.exists():
        raise FileNotFoundError("Model not found. Run model_training.py first.")
    return joblib.load(SLEEP_MODEL)

def load_features():
    if not SLEEP_FEATURES.exists():
        raise FileNotFoundError("features.json not found. Run model_training.py first.")
    return json.loads(SLEEP_FEATURES.read_text())

def predict_stress_risk(data: dict) -> str:
    feature_names = load_features()
    model_input = {
        "daily_screen_time_hours": data.get("screen_time", 0),
        "sleep_duration_hours": data.get("sleep_hours", 0),
        "physical_activity_hours": data.get("exercise_time", 0),
        "notifications_received_per_day": data.get("notifications", 0),
        "late_night_ratio": data.get("late_night_usage", 0),
        "mental_fatigue_score": 10 - data.get("mood_level", 5),
    }
    df = pd.DataFrame([model_input])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    model = load_model()
    pred  = int(model.predict(df[feature_names])[0])
    return "High Risk" if pred == 1 else "Low Risk"

def compute_recommendations(data: dict, result: str) -> list:
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

    if late > 0.5:
        recs.append("Avoid screens after 11 PM; enable Do Not Disturb and night mode.")
    elif late > 0.3:
        recs.append("Try to wind down from screens at least 30 minutes before bed.")

    if social > study and study > 0:
        recs.append("Use a Pomodoro timer and block social apps during study sessions.")
    elif social > study and study == 0:
        recs.append("Schedule dedicated study/work blocks and limit social app usage.")

    if sleep < 6:
        recs.append("Aim for 7–9 hours of sleep; keep a consistent bedtime.")
    elif sleep < 7:
        recs.append("Try to get at least 7 hours of sleep for optimal recovery.")

    if exercise < 0.5:
        recs.append("Add short daily exercise (20–30 minutes) to improve mood and sleep.")
    else:
        recs.append("Good exercise habits — keep them up.")

    if notif > 120:
        recs.append("Turn off non-essential notifications to reduce distractions.")
    if unlocks > 120:
        recs.append("Reduce phone checks by consolidating notifications and using widgets.")

    if mood <= 3:
        recs.append("Low mood noted — consider short mindfulness breaks or speaking with someone you trust.")
    elif mood <= 5:
        recs.append("Mood is moderate — small habits like journaling or a short walk can help.")
    elif mood >= 8:
        recs.append("Mood looks positive — continue current self-care practices.")

    if result == "High Risk":
        recs.append("Consider a short digital detox and review your daily screen habits.")
    else:
        recs.append("Overall indicators look balanced — monitor weekly and adjust as needed.")

    return recs[:6]

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
    .risk-high { color:#dc3545; font-weight:600; }
    .risk-low { color:#198754; font-weight:600; }
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
          <p class="subtle mb-4">Academic tool for lifestyle and phone-usage wellbeing guidance.</p>
          <div class="row g-3 mb-4">
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Stress Risk</div><div class="subtle">ML prediction from your usage patterns.</div></div></div>
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Sleep Impact</div><div class="subtle">Based on sleep and screen habits.</div></div></div>
            <div class="col-md-4"><div class="result-box"><div class="fw-semibold mb-1">Recommendations</div><div class="subtle">Personalised to your inputs.</div></div></div>
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
            <p class="subtle mb-0">Enter your daily values to get sleep and stress risk guidance.</p>
          </div>

          {% if error %}<div class="alert alert-danger py-2">{{ error }}</div>{% endif %}

          <form method="post" class="row g-3" id="predict-form">
            {% for field in fields %}
            <div class="col-md-6">
              <label class="form-label">{{ field.label }} <span class="text-muted fw-normal">({{ field.unit }})</span></label>
              <input class="form-control" type="number" name="{{ field.name }}"
                     step="{{ field.step }}" min="{{ field.min }}" max="{{ field.max }}"
                     placeholder="{{ field.placeholder }}"
                     value="{{ request.form.get(field.name, '') }}" required>
              <div class="form-text">{{ field.help }}</div>
            </div>
            {% endfor %}
            <div class="col-12 pt-2">
              <button class="btn btn-success px-4" type="submit">Predict</button>
              <a class="btn btn-outline-secondary ms-2" href="/predict">Reset</a>
            </div>
          </form>

          {% if result %}
          <hr class="my-4">
          <h5 class="mb-3">Your Results</h5>
          <div class="row g-3 mb-3">
            <div class="col-md-4">
              <div class="result-box">
                <div class="subtle mb-1">Sleep & Stress Risk</div>
                <div class="fs-6 {{ 'risk-high' if result == 'High Risk' else 'risk-low' }}">{{ result }}</div>
              </div>
            </div>
            <div class="col-md-8">
              <div class="result-box">
                <div class="subtle mb-1">What this means</div>
                <div>
                  {% if result == 'High Risk' %}
                    Your sleep and usage patterns suggest elevated stress. Review the recommendations below.
                  {% else %}
                    Your sleep and usage patterns look balanced. Keep up the good habits.
                  {% endif %}
                </div>
              </div>
            </div>
          </div>

          {% if recommendations %}
          <div class="mt-3">
            <h6 class="mb-2">Personalized Recommendations</h6>
            <ul class="rec-list">
              {% for r in recommendations %}<li class="mb-1">{{ r }}</li>{% endfor %}
            </ul>
          </div>
          {% endif %}
          {% endif %}

        </div>
      </div>
    </div>
  </div>
</div>

<script>
document.getElementById('predict-form').addEventListener('submit', function (e) {
    const hourFields = ['screen_time', 'social_media_usage', 'sleep_hours', 'study_hours', 'exercise_time'];
    const total = hourFields.reduce((sum, name) => {
        return sum + (parseFloat(document.querySelector('[name="' + name + '"]').value) || 0);
    }, 0);
    if (total > 24) {
        e.preventDefault();
        alert('Total hours exceed 24! You entered ' + total.toFixed(1) + ' hrs. Please adjust.');
    }
});
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return HOME_HTML

@app.route("/predict", methods=["GET", "POST"])
def predict():
    error = None
    result = None
    recommendations = []

    if request.method == "POST":
        try:
            data = {f["name"]: float(request.form.get(f["name"], 0) or 0) for f in FIELD_CONFIG}

            total_hours = sum(data[f] for f in HOUR_FIELDS)
            if total_hours > 24:
                error = f"Total hours exceed 24 (you entered {total_hours:.1f} hrs). Please adjust."
                return render_template_string(FORM_HTML, fields=FIELD_CONFIG, result=None, error=error, request=request, recommendations=[])

            result = predict_stress_risk(data)
            recommendations = compute_recommendations(data, result)

        except FileNotFoundError as e:
            error = str(e)
        except Exception:
            error = "Prediction failed. Ensure the model is trained and feature list is correct."

    return render_template_string(FORM_HTML, fields=FIELD_CONFIG, result=result, error=error, request=request, recommendations=recommendations)
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
