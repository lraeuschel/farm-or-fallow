"""
main.py – Flask blueprint for the farming-game web app.
Handles session setup, game flow (welcome → winter → summer → results),
and calls game-logic helpers for calculations and data fetching.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from app.game_logic import (
    calculate_winter_step, calculate_summer_step,
    calculate_final_yield, load_historical_weather
)
from app import plants_data, season_info, mitigation_info
from app.get_all_data import fetch_data_and_plot_all, use_existing_data_and_plot

# Blueprint registration
main = Blueprint('main', __name__)

# Fixed list of seasons played in sequence
SEASONS = ["2022-23", "2023-24"]


def ensure_session():
    """
    Initialize required session keys if they don't exist yet.
    Provides default values for tokens, plant choices, costs, etc.
    """
    if "season_index" not in session:
        session["season_index"] = 0
    if "total_profit" not in session:
        session["total_profit"] = 0
    if "winter_plants" not in session:
        session["winter_plants"] = ["fallow"] * 6
    if "summer_plants" not in session:
        session["summer_plants"] = ["fallow"] * 6
    if "tokens" not in session:
        session["tokens"] = 200
    if "cost_last_winter" not in session:
        session["cost_last_winter"] = 0
    if "cost_last_summer" not in session:
        session["cost_last_summer"] = 0
    if "mitigations" not in session:
        session["mitigations"] = [[] for _ in range(6)]
    if "cost_mitigations" not in session:
        session["cost_mitigations"] = 0
    if "season_history" not in session:
        session["season_history"] = {}
    if "is_telavi" not in session:
        session["is_telavi"] = False


@main.route('/')
def index():
    """Clear any existing session and redirect to the welcome page."""
    session.clear()
    ensure_session()
    return redirect(url_for('main.welcome'))


@main.route('/welcome')
def welcome():
    """Initial landing page with start instructions."""
    ensure_session()
    return render_template("welcome.html")


@main.route("/start_game", methods=["POST"])
def start_game():
    """
    Start a new game with user-provided coordinates.
    Fetches weather data and initializes the session.
    """
    data = request.get_json()
    if not data or "lat" not in data or "lng" not in data:
        return jsonify(success=False), 400

    # Round coordinates to one decimal place
    data["lat"] = round(data["lat"], 1)
    data["lng"] = round(data["lng"], 1)

    # Save farm location in the session
    session["farm_location"] = {"lat": data["lat"], "lng": data["lng"]}

    # Download & process weather data, create plots
    fetch_data_and_plot_all(data["lng"], data["lat"])

    ensure_session()
    return jsonify(success=True, redirect=url_for("main.winter"))


@main.route('/start_game_telavi', methods=['POST'])
def start_game_telavi():
    """
    Alternative start: use pre-existing Telavi data instead of user location.
    """
    use_existing_data_and_plot()
    ensure_session()
    session["is_telavi"] = True
    return jsonify(success=True, redirect=url_for("main.winter"))


@main.route('/restart')
def restart():
    """Reset the game and return to the welcome page."""
    session.clear()
    return redirect(url_for('main.welcome'))


@main.route('/winter', methods=['GET', 'POST'])
def winter():
    """
    Winter season selection page.
    GET: show plant options.
    POST: save chosen winter plants and deduct costs.
    """
    ensure_session()

    if request.method == 'POST':
        # Collect user choices; pad to six fields
        selected_winter = request.form.getlist("plants")
        if len(selected_winter) != 6:
            selected_winter += ["fallow"] * (6 - len(selected_winter))

        plants = plants_data
        previous_fields = session.get("summer_plants", ["fallow"] * 6)

        # Compute winter planting cost
        cost_winter = calculate_winter_step(selected_winter, previous_fields, plants)

        # Update session with new choices and token balance
        session.update({
            "winter_plants": selected_winter,
            "tokens": session["tokens"] - cost_winter,
            "cost_last_winter": cost_winter
        })

        return redirect(url_for('main.summer'))

    # Pre-fill with grapes if they were planted last summer
    plants = plants_data
    season = SEASONS[session["season_index"]]
    last_summer_plants = session.get("summer_plants", ["fallow"] * 6)
    winter_plants = ["grapes" if p == "grapes" else "fallow" for p in last_summer_plants]

    return render_template(
        "winter.html",
        plants=plants,
        season=season,
        tokens=session["tokens"],
        winter_plants=winter_plants,
        last_summer_plants=last_summer_plants,
        is_telavi=session["is_telavi"]
    )


@main.route('/summer', methods=['GET', 'POST'])
def summer():
    """
    Summer season selection page.
    GET: show plant and mitigation options.
    POST: save choices, apply costs, and move to results.
    """
    ensure_session()

    if request.method == 'POST':
        selected_summer = request.form.getlist("plants")
        selected_mitigations = [request.form.getlist(f"mitigations_{i}") for i in range(6)]

        if len(selected_summer) != 6:
            selected_summer += ["fallow"] * (6 - len(selected_summer))

        plants = plants_data
        previous_fields = session.get("winter_plants")

        # Calculate costs for summer planting and mitigations
        cost_summer_plants, cost_mitigations = calculate_summer_step(
            selected_summer, previous_fields, plants, selected_mitigations
        )

        session.update({
            "summer_plants": selected_summer,
            "tokens": session["tokens"] - cost_summer_plants - cost_mitigations,
            "cost_last_summer": cost_summer_plants,
            "mitigations": selected_mitigations,
            "cost_mitigations": cost_mitigations
        })

        return redirect(url_for('main.results'))

    plants = plants_data
    mitigation = mitigation_info
    season = SEASONS[session["season_index"]]
    summer_plants = session.get("summer_plants")

    return render_template(
        "summer.html",
        plants=plants,
        season=season,
        winter_plants=session["winter_plants"],
        summer_plants=summer_plants,
        tokens=session["tokens"],
        mitigation_info=mitigation,
        is_telavi=session["is_telavi"]
    )


@main.route('/results')
def results():
    """
    End-of-season results page.
    Calculates yield, updates total profit, and advances the game.
    Shows final summary if all seasons are complete.
    """
    ensure_session()
    season_index = session.get("season_index", 0)

    # If all seasons are done, show the game-over summary
    if season_index >= len(SEASONS):
        return render_template(
            "game_over.html",
            total_profit=session["total_profit"],
            season_history=session["season_history"]
        )

    current_year = SEASONS[season_index]
    weather = load_historical_weather(current_year)
    plants = plants_data

    # Calculate yield and profit for the current season
    yield_this_season = calculate_final_yield(session["summer_plants"], plants, weather)
    tokens = session["tokens"] + yield_this_season
    profit_this_season = yield_this_season - (
        session["cost_last_winter"] + session["cost_last_summer"] + session["cost_mitigations"]
    )

    total_costs = (
        session["cost_last_winter"] +
        session["cost_last_summer"] +
        session["cost_mitigations"]
    )

    # Store results for later display
    session["season_history"][current_year] = {
        "weather": weather,
        "cost_winter": session["cost_last_winter"],
        "cost_summer": session["cost_last_summer"],
        "cost_mitigations": session["cost_mitigations"],
        "total_costs": total_costs,
        "yield": yield_this_season,
        "profit": profit_this_season,
        "winter_plants": session["winter_plants"],
        "summer_plants": session["summer_plants"],
        "mitigations": session["mitigations"]
    }

    # Update cumulative values and advance to the next season
    session.update({
        "tokens": tokens,
        "total_profit": session["total_profit"] + profit_this_season,
        "season_index": session["season_index"] + 1
    })

    # If that was the last season, compute overall totals
    if session["season_index"] >= len(SEASONS):
        total_yield = sum(s.get("yield", 0) for s in session["season_history"].values())
        total_costs_all = sum(
            s.get("cost_winter", 0) + s.get("cost_summer", 0) + s.get("cost_mitigations", 0)
            for s in session["season_history"].values()
        )
        total_profit_sum = sum(s.get("profit", 0) for s in session["season_history"].values())

        return render_template(
            "game_over.html",
            total_profit=session.get("total_profit", 0),
            season_history=session["season_history"],
            tokens=tokens,
            total_yield=total_yield,
            total_costs=total_costs_all,
            total_profit_sum=total_profit_sum
        )

    # Otherwise, show per-season results page
    return render_template(
        "results.html",
        year=current_year,
        weather=weather,
        cost_winter=session["cost_last_winter"],
        cost_summer=session["cost_last_summer"],
        mitigations=session["mitigations"],
        cost_mitigations=session["cost_mitigations"],
        yield_this_season=yield_this_season,
        profit_this_season=profit_this_season,
        tokens=tokens,
        total_profit=session["total_profit"],
        winter_plants=session["winter_plants"],
        summer_plants=session["summer_plants"]
    )
