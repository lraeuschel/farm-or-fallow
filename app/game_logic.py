from app import season_info, mitigation_info

def load_historical_weather(year_range):
    """Load historical weather data for a given year range."""
    data = season_info
    season = data[year_range]

    winter_events = []
    summer_events = []

    if season["winter"]["wet"]:
        winter_events.append("w_wet")
    if season["winter"]["cold"]:
        winter_events.append("w_cold")

    if season["summer"]["wet"]:
        summer_events.append("s_wet")
    if season["summer"]["drought"]:
        summer_events.append("s_drought")
    if season["summer"]["hail"]:
        summer_events.append("s_hail")

    # print(f"Winter events: {winter_events}")
    # print(f"Summer events: {summer_events}")

    return {
        "winter": winter_events,
        "summer": summer_events,
        "season": year_range
    }


def calculate_winter_step(winter_plants, previous_plants, plants_data):
    """Calculate the cost for the winter planting step."""
    cost = 0

    for i, name in enumerate(winter_plants):
        if name == "fallow":
            continue
        plant = plants_data[name]

        if name == "grapes":
            if previous_plants[i] == "grapes":
                cost += plant["annual_cost"]
            else:
                cost += plant["cost"]
        else:
            cost += plant["cost"]   

    return int(cost)

def calculate_summer_step(summer_plants, winter_fields, plants_data, selected_mitigations):
    """Calculate the cost for the summer planting step, including mitigations."""
    cost = 0

    for i, name in enumerate(summer_plants):
        if name != "maize":
            continue
        plant = plants_data[name]

        if winter_fields[i] in ["wheat", "fallow"]:
            cost += plant["cost"]

    cost_mitigations = 0
    for i, mitigations in enumerate(selected_mitigations):
        for mitigation in mitigations:
            cost_mitigations += mitigation_info[mitigation]["costs"]

    return int(cost), int(cost_mitigations)

def calculate_final_yield(final_plants, plants_data, weather):
    """Calculate the final yield based on planted crops and weather conditions."""
    total_yield = 0

    for i, name in enumerate(final_plants):
        if name == "fallow":
            continue

        plant = plants_data[name]
        factor = 1.0

        for season_events in weather.values():
            for event in season_events:
                factor *= plant["impacts"].get(event, 1.0)

        total_yield += plant["yield"] * factor

    return int(total_yield)
