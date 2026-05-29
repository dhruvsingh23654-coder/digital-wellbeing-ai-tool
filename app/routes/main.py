from flask import Blueprint, render_template, request
from app.services.predictor import generate_predictions

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    user_data = request.form.to_dict()
    predictions = generate_predictions(user_data)
    return render_template('results.html', predictions=predictions)

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')