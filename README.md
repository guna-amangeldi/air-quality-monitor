# Air Quality ML Monitor (lazy edition) 🌍

A small Python tool that fetches air-quality data for a city, stores it as a growing dataset, and trains a simple ML baseline to forecast the next AQI value.

It's basically a tiny internal tool: API -> dataset -> baseline model -> readable output.

## What it does
- Downloads hourly air quality data (European AQI + PM2.5 + PM10 + NO₂ + O₃)
- Stores/updates a local dataset in `data/<city>.csv`
- Trains a simple Ridge regression model using lag features (last 6 hours)
- Predicts the next AQI value and prints a small summary + casual “developer UX”

## Requirements
- Python 3.10+
- Dependencies in `requirements.txt`

Install:
```bash
pip3 install -r requirements.txt
