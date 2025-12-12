import sys
from pathlib import Path

import requests
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# --- Cities ---
CITIES = {
    "munich": {"name": "Munich", "lat": 48.1374, "lon": 11.5755},
    "beijing": {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
    "ashgabat": {"name": "Ashgabat", "lat": 37.9601, "lon": 58.3261},
}


def fetch_air_quality(lat: float, lon: float, timezone: str = "auto") -> pd.DataFrame:
    """
    Fetch hourly air-quality data from Open-Meteo Air Quality API.
    Returns a DataFrame with a 'time' column and pollutant metrics.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "hourly": ",".join(
            [
                "european_aqi",
                "pm2_5",
                "pm10",
                "nitrogen_dioxide",
                "ozone",
            ]
        ),
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def append_to_csv(df: pd.DataFrame, city_key: str) -> Path:
    """
    Append fetched data to ./data/<city_key>.csv.
    Deduplicates by timestamp.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{city_key}.csv"

    if path.exists():
        old = pd.read_csv(path)
        old["time"] = pd.to_datetime(old["time"])
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time")
    else:
        combined = df.sort_values("time")

    combined.to_csv(path, index=False)
    return path


def make_lag_features(df: pd.DataFrame, target_col: str, lags: int = 6) -> pd.DataFrame:
    """
    Create lag features for a time series: last N hours of the target.
    """
    out = df[["time", target_col]].copy()
    for i in range(1, lags + 1):
        out[f"{target_col}_lag_{i}"] = out[target_col].shift(i)

    out = out.dropna().reset_index(drop=True)
    return out


def train_and_forecast(csv_path: Path, target_col: str = "european_aqi") -> dict:
    """
    Train a simple baseline model (Ridge regression) on lag features
    and forecast the next AQI value.
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).sort_values("time")

    feat = make_lag_features(df, target_col=target_col, lags=6)

    # Train/test split by time (last 20% for test)
    split = int(len(feat) * 0.8)
    train = feat.iloc[:split]
    test = feat.iloc[split:]

    X_train = train.drop(columns=["time", target_col])
    y_train = train[target_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    mae = None
    if len(test) > 0:
        X_test = test.drop(columns=["time", target_col])
        y_test = test[target_col]
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))

    # Forecast using the latest available lag row
    latest_row = feat.iloc[-1:].drop(columns=["time", target_col])
    next_pred = float(model.predict(latest_row)[0])

    return {
        "rows_total": int(len(df)),
        "rows_used_for_ml": int(len(feat)),
        "mae": mae,
        "forecast_next_aqi": next_pred,
        "last_timestamp": str(df["time"].max()),
        "last_aqi": float(df[target_col].iloc[-1]),
    }


# --- Lazy UX layer ---
def aqi_badge(aqi: float) -> str:
    """
    Feedback
    """
    if aqi < 50:
        return "Air is clean. You can breathe without thinking about it."
    if aqi < 100:
        return "Air is okay-ish. Probably fine, but don't overdo it."
    if aqi < 150:
        return "Air is kinda bad. Maybe skip that extra walk."
    return "Air is bad-bad. Today is a procrastination day."


def aqi_bar(aqi: float, max_aqi: float = 200) -> str:
    """
    Lazy visual because numbers alone are annoying.
    """
    width = 20
    aqi = max(0.0, min(float(aqi), float(max_aqi)))
    filled = int((aqi / max_aqi) * width)
    return f"AQI bar: [{'#' * filled}{'.' * (width - filled)}] {aqi:.0f}"


def main():
    print("Air Quality ML Monitor (lazy edition)")
    print("------------------------------------------")
    print("Cities:", ", ".join(CITIES.keys()))

    if len(sys.argv) >= 2:
        city_key = sys.argv[1].strip().lower()
    else:
        city_key = input("City: ").strip().lower()

    if city_key not in CITIES:
        print("Unknown city. Try:", ", ".join(CITIES.keys()))
        sys.exit(1)

    city = CITIES[city_key]
    print(f"\nFetching air quality for {city['name']}...")
    df = fetch_air_quality(city["lat"], city["lon"])

    csv_path = append_to_csv(df, city_key)
    print(f"Dataset updated: {csv_path} (yaay, you now own data)")

    print("\nTraining tiny ML baseline and predicting the next AQI...")
    result = train_and_forecast(csv_path)

    print("\nSummary (actual useful stuff)")
    print("-----------------------------")
    print(f"Latest timestamp: {result['last_timestamp']}")
    print(f"Latest AQI:       {result['last_aqi']:.1f}")
    print(f"Forecast next:    {result['forecast_next_aqi']:.1f}")

    if result["mae"] is not None:
        print(f"Model MAE:        {result['mae']:.2f}  (lower is better, but we’re not writing a thesis here)")
    else:
        print("Model MAE:        not enough history yet (run it again later)")

    # Simple alerting rule
    if result["forecast_next_aqi"] >= 100:
        print("ALERT: Forecast AQI is high. Maybe don’t speedrun.")
    else:
        print("\nForecast AQI looks okay. Congrats, the sky isn’t mad.")

    print("\nI hate dashboards")
    print("--------------------------------------")
    print(aqi_bar(result["forecast_next_aqi"]))
    print(aqi_badge(result["forecast_next_aqi"]))

    if result["forecast_next_aqi"] >= 100:
        print("Suggestion: maybe don't pollute today.")
    else:
        print("Suggestion: looks fine. Go outside.")

    print("\nTip: run this daily to build history. Future you will appreciate it.")


if __name__ == "__main__":
    main()
