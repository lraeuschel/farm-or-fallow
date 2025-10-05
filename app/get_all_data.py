import os
import requests
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import shutil


# ======================================================
# === Global thresholds and constants for game logic ===
# ======================================================
wet_precip_threshold_winter = 600 # mm precipitation considered "wet" in winter
cold_temp_threshold_winter = 0 # °C mean min temp considered "cold" in winter
wet_precip_threshold_summer = 550 # mm precipitation considered "wet" in summer
dry_precip_threshold = 350 # mm precipitation considered "dry"
streamflow_threshold = 100 # mm / 7 days for high streamflow
hail_days_threshold = 1 # days with hail considered "significant"
hail_codes = [27, 87, 88, 89, 90, 93, 94, 96, 99] # WMO weather code for hail

# ======================================================
# === Utility functions ================================
# ======================================================
def convert_bools(obj):
    """
    Recursively convert all numpy.bool_ objects to native Python bool.
    Ensures JSON serialization without numpy-specific types.
    """
    if isinstance(obj, dict):
        return {k: convert_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools(x) for x in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# ======================================================
# === Data fetching and storage ========================
# ======================================================
def fetch_and_store_all_weather_data(lon, lat):
    """
    Download observed, forecast, and 40-year climate data from Open-Meteo APIs, save them locally,
    and generate a game_settings.json with seasonal weather flags.
    """
    # Create and reuse a local data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- Observed daily weather (2022–2024) ---
    start_obs = "2021-11-01"
    end_obs = "2024-10-31"
    url_archive = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_obs,
        "end_date": end_obs,
        "daily": ["temperature_2m_max","temperature_2m_min","precipitation_sum","weathercode"],
        "timezone": "Europe/Berlin"
    }
    r = requests.get(url_archive, params=params, timeout=180)
    r.raise_for_status()
    data = r.json()
    df_obs = pd.DataFrame(data["daily"])
    df_obs["time"] = pd.to_datetime(df_obs["time"])

    # Save observed daily data as JSON (lists, not numpy arrays)
    obs_path = os.path.join(data_dir, "weather_data.json")
    df_obs_copy = df_obs.copy()
    df_obs_copy["time"] = df_obs_copy["time"].dt.strftime("%Y-%m-%d")
    with open(obs_path, "w") as f:
        json.dump(df_obs_copy.to_dict(orient="list"), f, indent=4)
    print("Saved observed weather data:", obs_path)

    # --- Historical forecast (hourly wind + daily temps/precip) ----
    url_forecast = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params_fc = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_obs,
        "end_date": end_obs,
        "daily": ["temperature_2m_max","temperature_2m_min","precipitation_sum"],
        "hourly": ["windspeed_10m","winddirection_10m"],
        "timezone": "Europe/Berlin"
    }
    r_fc = requests.get(url_forecast, params=params_fc, timeout=180)
    r_fc.raise_for_status()
    data_fc = r_fc.json()
    fc_path = os.path.join(data_dir, "forecast_data.json")
    with open(fc_path, "w") as f:
        json.dump(data_fc, f, indent=4)
    print("Saved historical forecast data:", fc_path)

    # --- 40-year climate archive (1982–2022) ---
    start_clim = "1982-01-01"
    end_clim = "2022-12-31"
    params_40y = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_clim,
        "end_date": end_clim,
        "daily": ["temperature_2m_max","temperature_2m_mean","temperature_2m_min","precipitation_sum"],
        "timezone": "Europe/Berlin"
    }
    r_40y = requests.get(url_archive, params=params_40y, timeout=300)
    r_40y.raise_for_status()
    data_40y = r_40y.json()
    clim_path = os.path.join(data_dir, "temperature_precip_40y.json")
    with open(clim_path, "w") as f:
        json.dump(data_40y, f, indent=4)
    print("Saved 40 years climate data:", clim_path)

    # --- Build game settings from observed data ---
    seasons = {
        "2022-23": {"winter": ("2022-11-01","2023-05-31"), "summer": ("2023-04-01","2023-10-31")},
        "2023-24": {"winter": ("2023-11-01","2024-05-31"), "summer": ("2024-04-01","2024-10-31")}
    }
    game_weather = {}
    for year_label, periods in seasons.items():
        game_weather[year_label] = {}

        # Winter conditions
        start, end = periods["winter"]
        df_winter = df_obs[(df_obs["time"] >= start) & (df_obs["time"] <= end)]
        game_weather[year_label]["winter"] = {
            "wet": bool(df_winter["precipitation_sum"].sum() > wet_precip_threshold_winter),
            "cold": bool(df_winter["temperature_2m_min"].mean() < cold_temp_threshold_winter)
        }

        # Summer conditions
        start, end = periods["summer"]
        df_summer = df_obs[(df_obs["time"] >= start) & (df_obs["time"] <= end)]
        hail_days = (df_summer["weathercode"].isin(hail_codes)).sum()
        game_weather[year_label]["summer"] = {
            "wet": bool(df_summer["precipitation_sum"].sum() > wet_precip_threshold_summer),
            "drought": bool(df_summer["precipitation_sum"].sum() < dry_precip_threshold),
            "streamflow": bool((df_summer["precipitation_sum"].rolling(7).sum() > streamflow_threshold).any()),
            "hail": bool(hail_days >= hail_days_threshold)
        }

    # Save game settings as JSON
    game_json_path = os.path.join(data_dir, "game_settings.json")
    with open(game_json_path, "w") as f:
        json.dump(convert_bools(game_weather), f, indent=4)
    print("Generated game settings:", game_json_path)

# ======================================================
# === Plotting functions ===============================
# ======================================================
def plot_max_temp_tercile_probabilities():
    """
    Calculate monthly max-temperature terciles from 40-year data and plot the probability of each tercile in observed seasons.
    Creates a colored table for each period.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    obs_path = os.path.join(data_dir, "weather_data.json")
    clim_path = os.path.join(data_dir, "temperature_precip_40y.json")
    plot_dir = os.path.join(current_dir, "static", "plots", "temperature_probabilities")
    os.makedirs(plot_dir, exist_ok=True)

    # Load observed daily data
    with open(obs_path) as f:
        df = pd.DataFrame(json.load(f))
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month

    # Load 40-year climate data
    with open(clim_path) as f:
        df_40y = pd.DataFrame(json.load(f)["daily"])
    df_40y["time"] = pd.to_datetime(df_40y["time"])
    df_40y["month"] = df_40y["time"].dt.month

    # Compute monthly tercile thresholds (33% and 66%)
    monthly_terciles = {}
    for m in range(1,13):
        temps = df_40y.loc[df_40y["month"]==m,"temperature_2m_max"]
        t1, t2 = temps.quantile(1/3), temps.quantile(2/3)
        monthly_terciles[m] = (t1, t2)

    # Define seasons to plot
    periods = [
        ("winter_2022-23","2022-11-01","2023-05-31"),
        ("summer_2022-23","2023-04-01","2023-10-31"),
        ("winter_2023-24","2023-11-01","2024-05-31"),
        ("summer_2023-24","2024-04-01","2024-10-31")
    ]

    # For each period, count days in each tercile and output table
    for period_name,start_period,end_period in periods:
        start_dt, end_dt = pd.to_datetime(start_period), pd.to_datetime(end_period)
        df_period = df[(df["time"]>=start_dt) & (df["time"]<=end_dt)].copy()
        df_period["month"] = df_period["time"].dt.month

        # Count days by month and tercile
        counts = {}
        for _, row in df_period.iterrows():
            m = row["month"]
            temp = row["temperature_2m_max"]
            t1,t2 = monthly_terciles[m]
            terc = 1 if temp<t1 else 2 if temp<t2 else 3
            counts.setdefault(m,{1:0,2:0,3:0})
            counts[m][terc]+=1

        # Convert to percentage
        prob_df = pd.DataFrame.from_dict(counts, orient="index")
        prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)*100
        prob_df = prob_df.T.loc[[3,2,1]] # Order Rows: Upper, Middle, Low

        # Month labels for table header
        month_labels = pd.to_datetime(df_period['time'].dt.to_period('M').unique().astype(str)).strftime('%b %Y')
        prob_df.columns = month_labels
        prob_df_int = prob_df.round(0).astype(int)

        # Adjust rounding so each column sums to 100
        for col in prob_df_int.columns:
            diff = 100 - prob_df_int[col].sum()
            if diff !=0:
                max_row = prob_df_int[col].idxmax()
                prob_df_int.loc[max_row,col]+=diff

        # Table plot
        row_colors={3:"lightcoral",2:"lightgreen",1:"lightblue"}
        colors = [[row_colors[row]]*len(prob_df_int.columns) for row in prob_df_int.index]
        fig,ax = plt.subplots(figsize=(12,4))
        ax.axis("off")
        table = ax.table(cellText=prob_df_int.values,
                         rowLabels=["Upper (3)","Middle (2)","Low (1)"],
                         colLabels=prob_df_int.columns,
                         cellColours=colors,
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2,1.2)
        filename = os.path.join(plot_dir,f"temperature_probabilities_{period_name}.png")
        plt.savefig(filename,bbox_inches="tight")
        plt.close(fig)
    print("Saved temperature probabilities plots.")


def plot_observed_max_temp():
    """
    Line plot of monthly average maximum temperatures.
    Background shows 40-year terciles,
    properly limited at the bottom (monthly minimum) and top (monthly maximum).
    """
    import os, json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(current_dir, "data")
    obs_path   = os.path.join(data_dir, "weather_data.json")
    clim_path  = os.path.join(data_dir, "temperature_precip_40y.json")
    plot_dir   = os.path.join(current_dir, "static", "plots", "observed_max_temperature")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Load observed data
    with open(obs_path) as f:
        df_obs = pd.DataFrame(json.load(f))
    df_obs["time"] = pd.to_datetime(df_obs["time"])

    # --- Load 40-year climate data and calculate monthly statistics
    with open(clim_path) as f:
        df_40 = pd.DataFrame(json.load(f)["daily"])
    df_40["time"]  = pd.to_datetime(df_40["time"])
    df_40["month"] = df_40["time"].dt.month

    # For each month: minimum, 33% tercile, 66% tercile, maximum
    monthly_stats = {}
    for m in range(1, 13):
        temps = df_40.loc[df_40["month"] == m, "temperature_2m_max"]
        monthly_stats[m] = {
            "tmin": temps.min(),
            "t1":   temps.quantile(1/3),
            "t2":   temps.quantile(2/3),
            "tmax": temps.max()
        }

    # Define periods
    periods = [
        ("winter_2022-23", "2021-11-01", "2022-10-31"),
        ("summer_2022-23", "2022-04-01", "2023-03-31"),
        ("winter_2023-24", "2022-11-01", "2023-10-31"),
        ("summer_2023-24", "2023-04-01", "2024-03-31"),
    ]

    for name, start, end in periods:
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        df_p = df_obs[(df_obs["time"] >= start_dt) & (df_obs["time"] <= end_dt)].copy()
        df_p["year_month"] = df_p["time"].dt.to_period("M")

        # Calculate monthly averages of observations
        monthly_avg = (
            df_p.groupby("year_month")["temperature_2m_max"]
                .mean().round(1)
        )
        months = pd.period_range(start=start_dt, end=end_dt, freq="M")
        monthly_avg = monthly_avg.reindex(months, fill_value=np.nan)

        # Arrays for all boundaries
        clim_min, lower, upper, clim_max = [], [], [], []
        for m in months:
            s = monthly_stats[m.month]
            clim_min.append(s["tmin"])
            lower.append(s["t1"])
            upper.append(s["t2"])
            clim_max.append(s["tmax"])

        clim_min = np.array(clim_min)
        lower    = np.array(lower)
        upper    = np.array(upper)
        clim_max = np.array(clim_max)

        # --- Plot
        x = np.arange(len(months))
        fig, ax = plt.subplots(figsize=(12, 4))

        # Fill bands between the respective boundaries
        ax.fill_between(x, clim_min, lower,    color="lightblue",  alpha=0.3,
                        label="Lower tercile (≤33 %)")
        ax.fill_between(x, lower,    upper,    color="lightgreen", alpha=0.3,
                        label="Middle tercile (33–66 %)")
        ax.fill_between(x, upper,    clim_max, color="lightcoral", alpha=0.3,
                        label="Upper tercile (≥66 %)")

        # Line for observed monthly averages
        ax.plot(x, monthly_avg.values, marker='o', color='orange',
                linewidth=2, label="Observed monthly average")

        # Axes and labels
        ax.set_ylabel("Ø Max Temp (°C)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.strftime("%b-%y") for m in months], rotation=45)

        # Y-axis: add some buffer below/above historical extremes
        ymin = min(np.nanmin(clim_min), np.nanmin(monthly_avg.values)) - 2
        ymax = max(np.nanmax(clim_max), np.nanmax(monthly_avg.values)) + 2
        ax.set_ylim(ymin, ymax)

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

        plt.tight_layout()
        out_path = os.path.join(
            plot_dir,
            f"observed_max_temperature_{name}.png"
        )
        plt.savefig(out_path)
        plt.close(fig)

    print("Saved max temperature line plots.")

def plot_windrose_forecast():
    """
    Create wind-rose diagrams for 3-month forecast periods
    based on historical forecast wind speed/direction.
    """
    periods = [
        ("winter_2022-23", "2022-12-01", "2023-02-28"),
        ("summer_2022-23", "2023-06-01", "2023-08-31"),
        ("winter_2023-24", "2023-12-01", "2024-02-29"),
        ("summer_2023-24", "2024-06-01", "2024-08-31"),
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(current_dir, "static", "plots", "windrose_forecast")
    os.makedirs(plot_dir, exist_ok=True)

    # Load forecast data (hourly)
    data_dir = os.path.join(current_dir, "data")
    forecast_path = os.path.join(data_dir, "forecast_data.json")
    with open(forecast_path) as f:
        data_fc = json.load(f)

    df = pd.DataFrame(data_fc["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # Plot wind rose for each month in each 3-month period
    for period_name, start_period, end_period in periods:
        start_dt = pd.to_datetime(start_period)
        end_dt = pd.to_datetime(end_period)
        months = pd.period_range(start=start_dt, end=end_dt, freq="M")

        fig, axes = plt.subplots(
            1, len(months),
            figsize=(6 * len(months), 6),
            subplot_kw={'projection': 'windrose'}
        )
        if len(months) == 1:
            axes = [axes]

        for ax, month in zip(axes, months):
            df_month = df[(df["year"] == month.year) & (df["month"] == month.month)]
            ws = df_month["windspeed_10m"]
            wd = df_month["winddirection_10m"]
            ax.bar(wd, ws, opening=0.8, edgecolor="white", bins=[0,2,4,6,8,10,15,20])
            ax.set_legend()
            ax.set_title(month.strftime("%b %Y"), fontsize=12, pad=20)


        plt.tight_layout()
        filename = os.path.join(plot_dir, f"windrose_forecast_{period_name}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    print("Saved windrose plots.")

def plot_temp_40y():
    """
    Plot 40-year monthly climatology of daily max/mean/min temperature
    with 5–95% and 25–75% percentile shading.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    clim_path = os.path.join(data_dir, "temperature_precip_40y.json")
    plot_dir = os.path.join(current_dir, "static", "plots", "climatology_temp_40y")
    os.makedirs(plot_dir, exist_ok=True)

    # Load daily climate data
    with open(clim_path) as f:
        df_40y = pd.DataFrame(json.load(f)["daily"])
    df_40y["time"] = pd.to_datetime(df_40y["time"])
    df_40y.set_index("time", inplace=True)

    # Aggregate to monthly statistics
    monthly = df_40y.groupby(df_40y.index.month).agg(
        max_mean=("temperature_2m_max", "mean"),
        max_p5=("temperature_2m_max", lambda x: x.quantile(0.05)),
        max_p25=("temperature_2m_max", lambda x: x.quantile(0.25)),
        max_p75=("temperature_2m_max", lambda x: x.quantile(0.75)),
        max_p95=("temperature_2m_max", lambda x: x.quantile(0.95)),

        mean_mean=("temperature_2m_mean", "mean"),
        mean_p5=("temperature_2m_mean", lambda x: x.quantile(0.05)),
        mean_p25=("temperature_2m_mean", lambda x: x.quantile(0.25)),
        mean_p75=("temperature_2m_mean", lambda x: x.quantile(0.75)),
        mean_p95=("temperature_2m_mean", lambda x: x.quantile(0.95)),

        min_mean=("temperature_2m_min", "mean"),
        min_p5=("temperature_2m_min", lambda x: x.quantile(0.05)),
        min_p25=("temperature_2m_min", lambda x: x.quantile(0.25)),
        min_p75=("temperature_2m_min", lambda x: x.quantile(0.75)),
        min_p95=("temperature_2m_min", lambda x: x.quantile(0.95))
    )

    months = range(1, 13)

    # Plots with shading
    fig, ax = plt.subplots(figsize=(12,6))

    # Max Temp
    ax.plot(months, monthly["max_mean"], color="darkred", label="daily max (mean)")
    ax.fill_between(months, monthly["max_p25"], monthly["max_p75"], color="red", alpha=0.3, label="daily max 25-75%")
    ax.fill_between(months, monthly["max_p5"], monthly["max_p95"], color="red", alpha=0.15, label="daily max 5-95%")

    # Mean Temp
    ax.plot(months, monthly["mean_mean"], color="darkorange", label="daily mean (mean)")
    ax.fill_between(months, monthly["mean_p25"], monthly["mean_p75"], color="orange", alpha=0.3, label="daily mean 25-75%")
    ax.fill_between(months, monthly["mean_p5"], monthly["mean_p95"], color="orange", alpha=0.15, label="daily mean 5-95%")

    # Min Temp
    ax.plot(months, monthly["min_mean"], color="teal", label="daily min (mean)")
    ax.fill_between(months, monthly["min_p25"], monthly["min_p75"], color="cyan", alpha=0.3, label="daily min 25-75%")
    ax.fill_between(months, monthly["min_p5"], monthly["min_p95"], color="cyan", alpha=0.15, label="daily min 5-95%")

    ax.set_xticks(months)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_ylabel("Monthly temperature (°C)")
    # ax.set_title(f"Climatology of daily temperatures ({start_year}–{end_year})")
    ax.legend(loc="upper right")

    plt.tight_layout()

    fname = f"climatology_temp_40y.png"
    fpath = os.path.join(plot_dir, fname)

    plt.savefig(fpath, dpi=300)
    plt.close(fig)
    print("Saved 40-year temperature plot.")

def plot_precip_40y():
    """
    Plot 40-year monthly precipitation climatology
    with 5–95% and 25–75% percentile shading.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    clim_path = os.path.join(data_dir, "temperature_precip_40y.json")
    plot_dir = os.path.join(current_dir, "static", "plots", "climatology_precip_40y")
    os.makedirs(plot_dir, exist_ok=True)

    # Load daily climate data
    with open(clim_path) as f:
        df_40y = pd.DataFrame(json.load(f)["daily"])
    df_40y["time"] = pd.to_datetime(df_40y["time"])
    df_40y.set_index("time", inplace=True)

    # Aggregate to monthly sums
    monthly = df_40y.groupby([df_40y.index.year, df_40y.index.month]).sum()

    start_year = 1982
    end_year = 2022

    # Limit to full years
    monthly = monthly.loc[start_year:end_year]

    # Calculate monthly statistics
    clim_stats = monthly.groupby(level=1).agg(
        mean=("precipitation_sum", "mean"),
        p25=("precipitation_sum", lambda x: x.quantile(0.25)),
        p75=("precipitation_sum", lambda x: x.quantile(0.75)),
        p5=("precipitation_sum", lambda x: x.quantile(0.05)),
        p95=("precipitation_sum", lambda x: x.quantile(0.95))
    )

    # Plot with shading
    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(months, clim_stats["mean"], color="blue", label="Mean precip")
    ax.fill_between(months, clim_stats["p25"], clim_stats["p75"], color="blue", alpha=0.3, label="25–75%")
    ax.fill_between(months, clim_stats["p5"], clim_stats["p95"], color="blue", alpha=0.15, label="5–95%")
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_title(f"Climatology of monthly precipitation ({start_year}-{end_year})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "climatology_precip_40y.png"), dpi=300)
    plt.close(fig)

    print("Saved 40-year precipitation plot.")


def plot_precip_periods():
    """
    Compare observed monthly precipitation to 40-year climatological means
    for each defined 12-month period.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    obs_path = os.path.join(data_dir, "weather_data.json")
    clim_path = os.path.join(data_dir, "temperature_precip_40y.json")
    plot_dir = os.path.join(current_dir, "static", "plots", "precip_obs_and_clim")
    os.makedirs(plot_dir, exist_ok=True)

    # Load observed and climatology datasets
    with open(obs_path) as f:
        df_obs = pd.DataFrame(json.load(f))
    df_obs["time"] = pd.to_datetime(df_obs["time"])
    df_obs.set_index("time", inplace=True)

    with open(clim_path) as f:
        df_40y = pd.DataFrame(json.load(f)["daily"])
    df_40y["time"] = pd.to_datetime(df_40y["time"])
    df_40y.set_index("time", inplace=True)

    # Mean monthly precipitation over 40 years
    monthly = df_40y.groupby([df_40y.index.year, df_40y.index.month]).sum()
    clim_stats = monthly.groupby(level=1).agg(
        mean=("precipitation_sum", "mean"),
        p25=("precipitation_sum", lambda x: x.quantile(0.25)),
        p75=("precipitation_sum", lambda x: x.quantile(0.75)),
        p5=("precipitation_sum", lambda x: x.quantile(0.05)),
        p95=("precipitation_sum", lambda x: x.quantile(0.95)),
    )

    # Periods to compare
    periods = [
        ("winter_2022-23", "2021-11-01"),
        ("summer_2022-23", "2022-04-01"),
        ("winter_2023-24", "2022-11-01"),
        ("summer_2023-24", "2023-04-01"),
    ]

    for period_name, start_period in periods:
        start_dt = pd.to_datetime(start_period)
        end_dt = start_dt + pd.DateOffset(years=1) - pd.Timedelta(days=1)

        # Observed monthly sums in the period
        df_period = df_obs.loc[start_dt:end_dt].copy()
        df_period["month"] = df_period.index.month
        df_period["year"] = df_period.index.year
        obs = df_period.groupby(["year", "month"])["precipitation_sum"].sum()

        # Create a range of months for the 12-month period
        months_range = pd.date_range(start=start_dt, end=end_dt, freq="MS")
        obs_vals, clim_means, labels = [], [], []
        for d in months_range:
            y, m = d.year, d.month
            obs_vals.append(obs.get((y, m), 0))
            clim_means.append(clim_stats.loc[m, "mean"])
            labels.append(d.strftime("%b %Y"))

        # Plot observed vs climatological precipitation
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(labels, obs_vals, color="dodgerblue", label="observed precipitation")
        ax.plot(labels, clim_means, color="green", marker="o", label="climatological mean")
        ax.set_ylabel("Monthly precipitation (mm)")
        # ax.set_title(f"Observed vs Climatology Precipitation — {period_name}")
        plt.xticks(rotation=45, ha="right")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"precip_obs_and_clim_{period_name}.png"), dpi=300)
        plt.close(fig)

    print("Saved precipitation obs vs climatology plots.")

def plot_monthly_precipitation():
    """
    Plot forecasted monthly precipitation totals for each 12-month period
    using historical forecast data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    obs_path = os.path.join(data_dir, "forecast_data.json")
    plot_dir = os.path.join(current_dir, "static", "plots", "precipitation_forecast")
    os.makedirs(plot_dir, exist_ok=True)

    # Load forecast data
    with open(obs_path) as f:
        data_fc = json.load(f)
    df = pd.DataFrame(data_fc["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.to_period("M")

    # Define periods to compare
    periods = [
        ("winter_2022-23", "2022-11-01", "2023-05-31"),
        ("summer_2022-23", "2023-04-01", "2023-10-31"),
        ("winter_2023-24", "2023-11-01", "2024-05-31"),
        ("summer_2023-24", "2024-04-01", "2024-10-31"),
    ]
    
    for period_name, start_period, end_period in periods:
        start_dt = pd.to_datetime(start_period)
        end_dt = pd.to_datetime(end_period)
        df_period = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].copy()

        # Aggregate to monthly sums
        monthly_precip = df_period.groupby("month")["precipitation_sum"].sum()
        month_labels = monthly_precip.index.strftime("%b %Y")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(month_labels, monthly_precip.values, color="skyblue", width=0.2)
        # ax.set_title(f"Monthly Precipitation — {period_name}")
        ax.set_ylabel("Total Precipitation (mm)")
        ax.set_xlabel("Month")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = os.path.join(plot_dir, f"precipitation_forecast_{period_name}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    print("Saved monthly precipitation forecast plots.")


def fetch_data_and_plot_all(lon, lat):
    fetch_and_store_all_weather_data(lon, lat)
    plot_max_temp_tercile_probabilities()
    plot_observed_max_temp()
    plot_windrose_forecast()
    plot_temp_40y()
    plot_precip_40y()
    plot_precip_periods()
    plot_monthly_precipitation()

def use_existing_data_and_plot():
    """
    Kopiert vorbereitete Daten aus telavi_data nach data
    und erzeugt danach alle Plots neu.
    """
    src_data_dir_name = "data/telavi_data"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, src_data_dir_name)
    dest_dir = os.path.join(current_dir, "data")

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Quelle {src_dir} existiert nicht")

    os.makedirs(dest_dir, exist_ok=True)

    copied_files = []
    for f in os.listdir(src_dir):
        src_file = os.path.join(src_dir, f)
        dest_file = os.path.join(dest_dir, f)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)
            copied_files.append(dest_file)

    print(f"{len(copied_files)} Dateien nach {dest_dir} kopiert")

    # Plots neu erstellen
    plot_max_temp_tercile_probabilities()
    plot_observed_max_temp()
    plot_windrose_forecast()
    plot_temp_40y()
    plot_precip_40y()
    plot_precip_periods()

    print("Alle Plots basierend auf bestehenden Daten erstellt.")
    
# -------------------------------
# Beispielaufrufe
# -------------------------------

# fetch_and_store_all_weather_data(3.5, 47.5)
# plot_max_temp_tercile_probabilities()
# plot_observed_max_temp()
# plot_windrose_forecast()
# plot_temp_40y()
# plot_precip_40y()
# plot_precip_periods()
# plot_monthly_precipitation()
# fetch_data_and_plot_all(3.5, 47.5)
# use_existing_data_and_plot()
