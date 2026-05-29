from flask import Blueprint

# Initialize the routes blueprint
routes_bp = Blueprint('routes', __name__)

from .main import *  # Import all routes from main.py