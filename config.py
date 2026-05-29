import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_default_secret_key'
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ['true', '1', 't']
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'ml/datasets/raw'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit upload size to 16 MB
    ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}  # Allowed file types for uploads

    # Machine Learning Model Configuration
    MODEL_PATH = 'ml/models/'  # Path to save trained models
    PREDICTION_THRESHOLD = 0.5  # Threshold for classification predictions

    # Logging Configuration
    LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', 'INFO')  # Set logging level
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'  # Log format

    # Other configurations can be added as needed