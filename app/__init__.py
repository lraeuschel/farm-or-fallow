from flask import Flask
import json
import os



def create_app():
    """"Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = "secret_key"

    # Load plant data from JSON file
    global plants_data
    with open(os.path.join(os.path.dirname(__file__), "data", "plants.json")) as f:
        plants_data = json.load(f)

    # Load season and game settings from JSON file
    global season_info
    season_info = json.load(open(os.path.join(os.path.dirname(__file__), "data", "game_settings.json")))

    # Load mitigation information from JSON file
    global mitigation_info
    mitigation_info = json.load(open(os.path.join(os.path.dirname(__file__), "data", "mitigations.json")))

    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    return app
