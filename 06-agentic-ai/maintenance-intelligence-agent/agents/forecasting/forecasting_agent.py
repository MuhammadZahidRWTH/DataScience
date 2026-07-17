# agents/forecasting/forecasting_agent.py
# Forecasting Agent — third agent in the MaintenanceGPT pipeline
#
# Responsibilities:
#   1. Load cleaned sensor data from ingestion agent
#   2. Train Prophet models per machine/sensor to forecast future readings
#   3. Predict probability of failure within FORECAST_HORIZON_HOURS
#   4. Use Ollama LLM to summarize forecast risk in plain language
#
# Design rationale:
#   - Time-series forecasting is core to predictive maintenance research
#   - Combines statistical forecasting (Prophet) with LLM summarization
#     for operator-facing risk communication

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import TypedDict
from prophet import Prophet
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import sys
import warnings

warnings.filterwarnings("ignore")  # Prophet is noisy with cmdstanpy logs

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    DATA_PROCESSED,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    FORECAST_HORIZON_HOURS,
    FAILURE_PROBABILITY_THRESHOLD,
)


# ── Agent State ────────────────────────────────────────────
class ForecastingState(TypedDict):
    sensor_df: pd.DataFrame
    forecasts: list[dict]
    risk_summary: str
    summary: str
    errors: list[str]


# ── Load Cleaned Sensor Data ───────────────────────────────
def load_sensor_data(state: ForecastingState) -> ForecastingState:
    """Load cleaned sensor data from ingestion agent output."""
    errors = []

    try:
        sensor_df = pd.read_csv(
            DATA_PROCESSED / "sensor_clean.csv",
            parse_dates=["timestamp"]
        )
        print(f"  [Forecasting] Loaded cleaned sensor data: {len(sensor_df)} rows")
    except Exception as e:
        sensor_df = pd.DataFrame()
        errors.append(f"Sensor data load failed: {e}")

    return {**state, "sensor_df": sensor_df, "errors": errors}


# ── Forecast per Machine/Sensor ────────────────────────────
def forecast_sensor_trends(state: ForecastingState) -> ForecastingState:
    """
    Train Prophet models per (machine, sensor) combination.
    Forecast next FORECAST_HORIZON_HOURS and flag if predicted
    values exceed danger thresholds derived from historical data.

    Why Prophet:
    - Handles seasonality (shift patterns) and trend automatically
    - Robust to missing data — common in real sensor streams
    - Provides uncertainty intervals, useful for risk communication
    """
    df = state["sensor_df"].copy()

    if df.empty:
        return {**state, "forecasts": []}

    sensors = ["temperature_c", "vibration_mm_s", "pressure_bar"]
    machines = df["machine_id"].unique()

    # danger thresholds: mean + 3*std per machine/sensor (historical)
    forecasts = []

    print(f"  [Forecasting] Training Prophet models for {len(machines)} machines x {len(sensors)} sensors...")

    for machine in machines:
        mdf = df[df["machine_id"] == machine].copy()
        mdf = mdf[["timestamp", *sensors]].dropna()

        if len(mdf) < 20:
            continue

        for sensor in sensors:
            try:
                # prepare Prophet input format
                prophet_df = mdf[["timestamp", sensor]].rename(
                    columns={"timestamp": "ds", sensor: "y"}
                )

                # historical danger threshold
                mean_val = prophet_df["y"].mean()
                std_val = prophet_df["y"].std()
                danger_threshold = mean_val + 2 * std_val

                # train Prophet — daily seasonality off (hourly data),
                # weekly seasonality captures shift patterns
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    interval_width=0.90,
                )
                model.fit(prophet_df)

                # forecast next FORECAST_HORIZON_HOURS
                future = model.make_future_dataframe(
                   periods=FORECAST_HORIZON_HOURS, freq="h"
)
                forecast = model.predict(future)

                # only look at the forecasted (future) portion
                future_forecast = forecast.tail(FORECAST_HORIZON_HOURS)

                # probability of exceeding danger threshold
                # approximated via upper confidence bound crossing
                max_predicted = future_forecast["yhat_upper"].max()
                mean_predicted = future_forecast["yhat"].mean()

                exceeds_threshold = (future_forecast["yhat_upper"] > danger_threshold).sum()
                failure_probability = round(
                    exceeds_threshold / FORECAST_HORIZON_HOURS, 3
                )

                # find earliest hour where danger threshold is crossed
                crossing = future_forecast[future_forecast["yhat_upper"] > danger_threshold]
                hours_to_failure = None
                if not crossing.empty:
                    first_cross_time = crossing.iloc[0]["ds"]
                    last_historical_time = prophet_df["ds"].max()
                    hours_to_failure = round(
                        (first_cross_time - last_historical_time).total_seconds() / 3600, 1
                    )

                forecasts.append({
                    "machine_id": machine,
                    "sensor": sensor,
                    "current_mean": round(mean_val, 3),
                    "danger_threshold": round(danger_threshold, 3),
                    "forecast_mean_48h": round(mean_predicted, 3),
                    "forecast_max_48h": round(max_predicted, 3),
                    "failure_probability": failure_probability,
                    "hours_to_threshold_breach": hours_to_failure,
                    "at_risk": failure_probability >= FAILURE_PROBABILITY_THRESHOLD or hours_to_failure is not None,
                })

            except Exception as e:
                forecasts.append({
                    "machine_id": machine,
                    "sensor": sensor,
                    "error": str(e),
                })

    at_risk_count = sum(1 for f in forecasts if f.get("at_risk"))
    print(f"  [Forecasting] {len(forecasts)} forecasts generated, {at_risk_count} machines/sensors at risk")

    return {**state, "forecasts": forecasts}


# ── LLM Risk Summary ────────────────────────────────────────
def summarize_risk_with_llm(state: ForecastingState) -> ForecastingState:
    """
    Use Ollama to translate forecast numbers into operator-facing
    risk summary. Focuses on at-risk machines only.
    """
    forecasts = state["forecasts"]
    at_risk = [f for f in forecasts if f.get("at_risk")]

    if not at_risk:
        return {
            **state,
            "risk_summary": "No machines forecasted to exceed danger thresholds in the next 48 hours."
        }

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    # build a compact summary of at-risk items for the prompt
    risk_lines = []
    for f in at_risk[:8]:  # cap to keep prompt manageable
        hrs = f.get("hours_to_threshold_breach")
        hrs_str = f"{hrs}h" if hrs is not None else "unknown timeframe"
        risk_lines.append(
            f"- {f['machine_id']} | {f['sensor']}: "
            f"forecast probability {f['failure_probability']*100:.0f}% "
            f"of exceeding safe threshold within {hrs_str}"
        )

    prompt = f"""You are a maintenance planning assistant at a German manufacturing plant.

A 48-hour forecasting model flagged the following risks:

{chr(10).join(risk_lines)}

Write a short operator briefing (3-4 sentences) prioritizing which
machines need attention first and why. Be direct and actionable."""

    print(f"  [Forecasting] Generating LLM risk summary for {len(at_risk)} at-risk items...")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        risk_summary = response.content.strip()
    except Exception as e:
        risk_summary = f"Risk summary unavailable: {e}"

    return {**state, "risk_summary": risk_summary}


# ── Save Forecasts ─────────────────────────────────────────
def save_forecasts(state: ForecastingState) -> ForecastingState:
    """Save forecast results for downstream agents."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    output = {
        "forecasts": state["forecasts"],
        "risk_summary": state["risk_summary"],
    }

    with open(DATA_PROCESSED / "forecasts.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  [Forecasting] Saved {len(state['forecasts'])} forecasts to data/processed/")
    return state


# ── Summary ────────────────────────────────────────────────
def generate_summary(state: ForecastingState) -> ForecastingState:
    forecasts = state["forecasts"]
    at_risk = [f for f in forecasts if f.get("at_risk")]

    summary = (
        f"Forecasting complete. "
        f"{len(forecasts)} machine/sensor combinations analyzed. "
        f"{len(at_risk)} flagged at risk within {FORECAST_HORIZON_HOURS}h."
    )
    print(f"  [Forecasting] {summary}")
    return {**state, "summary": summary}


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MaintenanceGPT — Forecasting Agent ===\n")

    state: ForecastingState = {
        "sensor_df": pd.DataFrame(),
        "forecasts": [],
        "risk_summary": "",
        "summary": "",
        "errors": [],
    }

    state = load_sensor_data(state)
    state = forecast_sensor_trends(state)
    state = summarize_risk_with_llm(state)
    state = save_forecasts(state)
    state = generate_summary(state)

    print("\n=== Forecasting Agent Complete ===")
    print(f"Summary: {state['summary']}")
    print(f"\nRisk Briefing:\n{state['risk_summary']}")
    if state["errors"]:
        print(f"\nErrors: {state['errors']}")
