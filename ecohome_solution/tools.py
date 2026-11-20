"""
Tools for the EcoHome Energy Advisor agent.

Each function decorated with ``@tool`` is offered to the model. The model reads
the function name, its docstring and its argument types, decides when to call
it, and receives the returned dictionary as a tool message. So the docstrings
below are written for the model as much as for people: they say what each
argument means, which values are valid, and what comes back.

Design decisions that apply to every tool:

* **Paths are resolved from this file, not from the current folder.** The
  starter opened ``data/energy_data.db`` relative to wherever Python was
  started; imported from another folder, SQLite could not open (or silently
  created) a different file. ``BASE_DIR`` removes that dependency.
* **Errors are returned, not raised.** A tool that raises ends the agent's
  run; a tool that returns ``{"error": ...}`` lets the model correct its
  arguments and try again.
* **Results are summarised before they reach the model.** The starter's usage
  query returned every hourly row: about 82,000 characters for one week and
  354,000 for 30 days, which is most of a model's context window. The query
  tools now return totals, a per-day table and a per-device breakdown, and
  include raw rows only on request and only up to a cap.
* **Simulated data is deterministic.** Weather and prices are simulated (the
  rubric allows mocks). They are generated from a seed derived from the
  location and the date with ``hashlib``, so the same question on the same
  day gets the same numbers, and the evaluation can check the agent's claims
  against the data it was shown. Python's built-in ``hash()`` is randomised
  per process and would not give that guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.energy import DatabaseManager

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "energy_data.db"
DOCS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
MANIFEST_FILE = VECTORSTORE_DIR / "documents_manifest.json"

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"   # one constant shared by notebook 02 and the tool
COLLECTION_NAME = "ecohome_energy_tips"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

BASE_PRICE_PER_KWH = 0.15   # USD; the seeded usage data prices off-peak at 0.10 and peak at 0.15
# Solar array size implied by the seeded data: 5 kWh in an hour at 800 W/m2 of irradiance.
KWH_PER_IRRADIANCE_UNIT = 5.0 / 800.0

DEVICE_ALIASES = {
    "ev": "EV", "electric vehicle": "EV", "electric car": "EV", "car": "EV", "tesla": "EV",
    "hvac": "HVAC", "ac": "HVAC", "air conditioning": "HVAC", "heating": "HVAC", "thermostat": "HVAC",
    "appliance": "appliance", "appliances": "appliance", "dishwasher": "appliance",
    "washing machine": "appliance", "washer": "appliance", "dryer": "appliance",
}

db_manager = DatabaseManager(str(DB_PATH))


# --------------------------------------------------------------------------- helpers

def _seeded_rng(*parts: str) -> random.Random:
    """A random generator that gives the same sequence for the same inputs, in every process."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _parse_date(value: Optional[str], name: str) -> datetime:
    """Parse YYYY-MM-DD, raising ValueError with a message the model can act on."""
    if value is None:
        raise ValueError(f"{name} is required in YYYY-MM-DD format")
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{name} must be a date in YYYY-MM-DD format, got {value!r}") from None


def _normalise_device_type(device_type: Optional[str]) -> Optional[str]:
    """Map 'ev', 'Electric car', 'appliances' and similar to the stored values EV, HVAC, appliance."""
    if device_type is None or not str(device_type).strip():
        return None
    key = str(device_type).strip().lower()
    return DEVICE_ALIASES.get(key, str(device_type).strip())


def _wrap_at(hours):
    """The position where a list of hours given in order runs past midnight ([22, 23, 0, 1] -> 2), or None.

    Only a list in clock order that steps back to an earlier hour counts: [0, 1, 2, 22, 23] is one day's hours."""
    hs = list(hours or [])
    for i in range(1, len(hs)):
        if hs[i] < hs[i - 1]:
            return i if all(hs[j] < hs[j + 1] for j in range(len(hs) - 1) if j != i - 1) else None
    return None


def hour_window(hours) -> str:
    """Hours of the day written as clock ranges whose end is exclusive: [16, 17, 18, 19, 20] -> "16:00-21:00".

    Every range this module prints follows one convention: a range ends at the start of the first hour it does
    NOT include, so the peak hours 16-20 are "16:00-21:00" and the off-peak hours 22-23 and 0-5 are "22:00-06:00".
    Writing the last included hour as the end ("16:00-20:00") silently drops the 20:00 hour; the advisor's data
    checks read ranges the same way, so giving the model ready-made strings removes that off-by-one.
    """
    hs = sorted({int(h) % 24 for h in hours or []})
    if not hs:
        return ""
    runs, start, prev = [], hs[0], hs[0]
    for h in hs[1:]:
        if h != prev + 1:
            runs.append((start, prev)); start = h
        prev = h
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 23:   # 22, 23, 0..5 is one window across midnight
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return ", ".join(f"{a:02d}:00-{(b + 1) % 24:02d}:00" for a, b in runs)


def _daylight_curve(hour: int) -> float:
    """0 at night, rising to 1 at 12:00; matches the seeded solar data (generation from 06:00 to 18:00)."""
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, 1 - abs(hour - 12) / 6)


# --------------------------------------------------------------------------- weather

SOLAR_HOURS = [h for h in range(24) if _daylight_curve(h) > 0]   # 07:00-18:00: the hours the panels can produce
CORE_SOLAR_HOURS = [h for h in range(24) if _daylight_curve(h) >= 0.5]   # 09:00-16:00: half of noon's output or more


def _solar_by_hour(location: str, date: str) -> Dict[int, float]:
    """The forecast's estimated solar kWh for each hour of one date, as get_weather_forecast reports it for that date.

    The simulated forecast is seeded by location and date, so one day computed alone equals that day inside a 7-day
    forecast. With live weather on, the live forecast is used when it covers the date.
    """
    day = _parse_date(date, "date")
    if os.getenv("ECOHOME_LIVE_WEATHER") == "1":
        live = get_weather_forecast.invoke({"location": location, "days": 7})
        rows = {h["hour"]: h["estimated_solar_kwh"] for h in live.get("hourly", []) if h.get("date") == day.strftime("%Y-%m-%d")}
        if rows:
            return rows
    return {h["hour"]: h["estimated_solar_kwh"] for h in _simulated_forecast(location, 1, day)["hourly"]}


CONDITIONS = {
    # condition: (probability weight, irradiance multiplier, humidity range)
    "sunny": (0.45, 1.0, (30, 50)),
    "partly_cloudy": (0.30, 0.6, (45, 65)),
    "cloudy": (0.18, 0.3, (60, 80)),
    "rainy": (0.07, 0.1, (75, 95)),
}


def _simulated_forecast(location: str, days: int, start: datetime) -> Dict[str, Any]:
    """Deterministic forecast: per-day condition weights multiplied into base temperature and irradiance curves."""
    hourly: List[Dict[str, Any]] = []
    daily: List[Dict[str, Any]] = []
    for d in range(days):
        day = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        rng = _seeded_rng("weather", location.lower().strip(), day)
        condition = rng.choices(list(CONDITIONS), weights=[c[0] for c in CONDITIONS.values()])[0]
        _, sun_mult, (h_lo, h_hi) = CONDITIONS[condition]
        base_temp = rng.uniform(8, 24)      # November-ish daily mean, varied by day
        swing = rng.uniform(4, 9)
        day_kwh = 0.0
        for hour in range(24):
            temp = base_temp + swing * math.sin((hour - 9) / 24 * 2 * math.pi)
            irradiance = round(1000 * _daylight_curve(hour) * sun_mult * rng.uniform(0.9, 1.1), 1)
            est_kwh = round(irradiance * KWH_PER_IRRADIANCE_UNIT, 2)
            day_kwh += est_kwh
            hourly.append({
                "date": day, "hour": hour,
                "temperature_c": round(temp, 1),
                "condition": condition,
                "solar_irradiance": irradiance,
                "estimated_solar_kwh": est_kwh,
                "humidity": rng.randint(h_lo, h_hi),
                "wind_speed": round(rng.uniform(1, 8), 1),
            })
        best = [h["hour"] for h in hourly[-24:] if h["estimated_solar_kwh"] >= 0.6 * max(1e-9, max(x["estimated_solar_kwh"] for x in hourly[-24:]))]
        daily.append({"date": day, "day_of_week": (start + timedelta(days=d)).strftime("%A"), "condition": condition,
                      "estimated_solar_kwh": round(day_kwh, 2), "best_solar_hours": best, "best_solar_window": hour_window(best),
                      # a live answer added the window's hours itself and wrote 15.24 kWh for Saturday's 17.24
                      "solar_kwh_in_best_window": round(sum(h["estimated_solar_kwh"] for h in hourly[-24:] if h["hour"] in best), 2)})
    now_h = hourly[min(datetime.now().hour, len(hourly) - 1)]
    return {
        "location": location, "forecast_days": days, "source": "simulated",
        "current": {k: now_h[k] for k in ("temperature_c", "condition", "humidity", "wind_speed")},
        "daily": daily, "hourly": hourly,
    }


def _open_meteo_forecast(location: str, days: int) -> Dict[str, Any]:
    """Live forecast from Open-Meteo (free, no key). Used only when ECOHOME_LIVE_WEATHER=1."""
    q = urllib.parse.urlencode({"name": location.split(",")[0].strip(), "count": 1})
    with urllib.request.urlopen(f"https://geocoding-api.open-meteo.com/v1/search?{q}", timeout=10) as r:
        hits = json.load(r).get("results") or []
    if not hits:
        raise ValueError(f"location not found: {location!r}")
    lat, lon = hits[0]["latitude"], hits[0]["longitude"]
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "forecast_days": days, "timezone": "auto",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,shortwave_radiation",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
    })
    with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=10) as r:
        data = json.load(r)
    cond = lambda cc: "sunny" if cc < 25 else "partly_cloudy" if cc < 60 else "cloudy"
    h = data["hourly"]
    hourly, daily = [], {}
    for i, ts in enumerate(h["time"]):
        day, hour = ts[:10], int(ts[11:13])
        irr = float(h["shortwave_radiation"][i] or 0)
        est = round(irr * KWH_PER_IRRADIANCE_UNIT, 2)
        hourly.append({"date": day, "hour": hour, "temperature_c": h["temperature_2m"][i],
                       "condition": cond(h["cloud_cover"][i]), "solar_irradiance": irr,
                       "estimated_solar_kwh": est, "humidity": h["relative_humidity_2m"][i],
                       "wind_speed": h["wind_speed_10m"][i]})
        daily.setdefault(day, 0.0)
        daily[day] += est
    c = data["current"]
    return {
        "location": f"{hits[0]['name']}, {hits[0].get('admin1', '')}".strip(", "),
        "forecast_days": days, "source": "open-meteo",
        "current": {"temperature_c": c["temperature_2m"], "condition": cond(c["cloud_cover"]),
                    "humidity": c["relative_humidity_2m"], "wind_speed": c["wind_speed_10m"]},
        "daily": [{"date": d, "estimated_solar_kwh": round(v, 2)} for d, v in daily.items()],
        "hourly": hourly,
    }


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get the hourly weather forecast for a location, starting today, including solar irradiance
    and the solar generation it implies for this home's panels.

    The first day is TODAY, so days=7 covers today and the next six days; a date that is not in
    "daily" has no forecast. Quote only the days listed.

    Args:
        location (str): City and region, e.g. "San Francisco, CA".
        days (int): Number of days to forecast, 1 to 7. Use 2 when the question is about tomorrow.

    Returns:
        Dict[str, Any]: location, forecast_days, source ("simulated" or "open-meteo"),
        current conditions, a daily summary (date, day_of_week, condition, estimated_solar_kwh,
        best_solar_hours, solar_kwh_in_best_window (the forecast's solar in those hours: quote it rather than adding
        hours yourself), and best_solar_window: those hours as a clock range, e.g. "10:00-15:00"
        for hours 10-14; quote that string, the range ends where the last hour ends),
        and hourly entries with date, hour (0-23), temperature_c, condition, solar_irradiance (W/m2),
        estimated_solar_kwh, humidity (%) and wind_speed (m/s).
    """
    if not isinstance(days, int) or not 1 <= days <= 7:
        return {"error": f"days must be an integer from 1 to 7, got {days!r}"}
    if not location or not str(location).strip():
        return {"error": "location is required, e.g. 'San Francisco, CA'"}
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if os.getenv("ECOHOME_LIVE_WEATHER") == "1":
        try:
            return _open_meteo_forecast(location, days)
        except Exception as e:  # network down, unknown city: fall back, and say so
            forecast = _simulated_forecast(location, days, start)
            forecast["note"] = f"live weather unavailable ({type(e).__name__}); simulated forecast returned"
            return forecast
    return _simulated_forecast(location, days, start)


# --------------------------------------------------------------------------- prices

# Time-of-use schedule: (period, multiplier on the base price, demand charge USD/kW)
def _period(hour: int, weekend: bool) -> tuple:
    if hour < 6 or hour >= 22:
        return ("off_peak", 0.70, 0.0)
    if 16 <= hour < 21 and not weekend:
        return ("peak", 1.60, 0.05)
    return ("mid_peak", 1.00, 0.02)


@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get time-of-use electricity prices for one day, hour by hour.

    Args:
        date (str): Date in YYYY-MM-DD format. Leave empty for today. Use tomorrow's date for questions about tomorrow.

    Returns:
        Dict[str, Any]: date, pricing_type ("time_of_use"), currency ("USD"), unit ("per_kWh"),
        hourly_rates (hour 0-23 with rate, period "off_peak" | "mid_peak" | "peak", demand_charge),
        and a summary with the cheapest and most expensive hours, the average rate, and the periods as
        clock ranges (off_peak_window "22:00-06:00", peak_window "16:00-21:00" on weekdays, None at weekends).
        Off-peak is 22:00-06:00 every day; peak is 16:00-21:00 on weekdays only. A range ends where its last
        hour ends: 16:00-21:00 is hours 16, 17, 18, 19 and 20.
    """
    try:
        day = datetime.now() if date in (None, "", "today") else _parse_date(date, "date")
    except ValueError as e:
        return {"error": str(e)}
    day_s = day.strftime("%Y-%m-%d")
    weekend = day.weekday() >= 5
    rng = _seeded_rng("prices", day_s)
    day_factor = rng.uniform(0.95, 1.08)        # small day-to-day market variation
    rates = []
    for hour in range(24):
        period, mult, demand = _period(hour, weekend)
        rate = round(BASE_PRICE_PER_KWH * mult * day_factor * rng.uniform(0.98, 1.02), 4)
        rates.append({"hour": hour, "rate": rate, "period": period, "demand_charge": demand})
    ordered = sorted(rates, key=lambda r: r["rate"])
    return {
        "date": day_s, "day_of_week": day.strftime("%A"), "pricing_type": "time_of_use",
        "currency": "USD", "unit": "per_kWh", "source": "simulated",
        "hourly_rates": rates,
        "summary": {
            "cheapest_hours": sorted(r["hour"] for r in ordered[:6]),
            "most_expensive_hours": sorted(r["hour"] for r in ordered[-5:]),
            "min_rate": ordered[0]["rate"], "max_rate": ordered[-1]["rate"],
            "average_rate": round(sum(r["rate"] for r in rates) / 24, 4),
            "off_peak_hours": [r["hour"] for r in rates if r["period"] == "off_peak"],
            "peak_hours": [r["hour"] for r in rates if r["period"] == "peak"],
            "off_peak_window": hour_window(r["hour"] for r in rates if r["period"] == "off_peak"),
            "peak_window": hour_window(r["hour"] for r in rates if r["period"] == "peak") or None,
        },
    }


# --------------------------------------------------------------------------- history (database)

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None,
                       include_records: bool = False, max_records: int = 48) -> Dict[str, Any]:
    """
    Query this home's past energy consumption from the database for a date range (inclusive).

    Args:
        start_date (str): Start date in YYYY-MM-DD format. For "the last 7 days", 7 days before today.
        end_date (str): End date in YYYY-MM-DD format (inclusive). For "the last 7 days", yesterday: today is not over yet.
        device_type (str): Optional filter: "EV", "HVAC" or "appliance" (case and common names such as "electric car" are accepted),
            or one machine: "dishwasher", "washing machine" or "dryer" returns that machine's records only.
        include_records (bool): Also return individual hourly rows (default False; the summaries are usually enough).
        max_records (int): Most rows to return when include_records is True (default 48).

    Returns:
        Dict[str, Any]: totals (kWh, USD), a per-device breakdown (kWh, USD, average cost per kWh, devices seen),
        a per-day table, an hour-of-day profile (average kWh and cost per hour), the date span the data actually
        covers, and rows only if requested.
    """
    try:
        start_dt = _parse_date(start_date, "start_date")
        end_dt = _parse_date(end_date, "end_date") + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date"}
        dt_filter = _normalise_device_type(device_type)
        # "dishwasher" normalises to the stored type "appliance"; keep the machine so only its records count.
        # Without this, a dishwasher question was answered with every appliance's total (32.78 kWh a day).
        machine = APPLIANCE_NAMES.get(str(device_type or "").strip().lower())
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        records = [r for r in records if r.timestamp < end_dt]
        if dt_filter:
            records = [r for r in records if r.device_type == dt_filter]
        if machine:
            records = [r for r in records if r.device_name == machine]
        devices: Dict[str, Dict[str, Any]] = {}
        days: Dict[str, Dict[str, float]] = {}
        hours: Dict[int, Dict[str, float]] = {}
        for r in records:
            d = devices.setdefault((machine if machine else r.device_type) or "unknown",
                                   {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0, "device_names": set()})
            d["consumption_kwh"] += r.consumption_kwh; d["cost_usd"] += r.cost_usd or 0; d["records"] += 1
            if r.device_name:
                d["device_names"].add(r.device_name)
            day = days.setdefault(r.timestamp.strftime("%Y-%m-%d"), {"consumption_kwh": 0.0, "cost_usd": 0.0})
            day["consumption_kwh"] += r.consumption_kwh; day["cost_usd"] += r.cost_usd or 0
            h = hours.setdefault(r.timestamp.hour, {"kwh": 0.0, "cost": 0.0, "n": 0})
            h["kwh"] += r.consumption_kwh; h["cost"] += r.cost_usd or 0; h["n"] += 1
        n_days = max(1, len(days))
        result = {
            "start_date": start_date, "end_date": end_date, "device_type": dt_filter, "device_name": machine,
            "total_records": len(records),
            "data_covers": ({"first": records[0].timestamp.isoformat(), "last": records[-1].timestamp.isoformat(), "days_with_data": len(days)} if records else None),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            # the household's (or the filtered device's) use per day, so a model never adds device averages itself
            # (a live answer summed three rounded averages to 297.93 kWh; the total over the days is 297.94)
            "average_daily_kwh": round(sum(r.consumption_kwh for r in records) / n_days, 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "device_breakdown": {k: {"consumption_kwh": round(v["consumption_kwh"], 2), "cost_usd": round(v["cost_usd"], 2),
                                     "average_cost_per_kwh": round(v["cost_usd"] / v["consumption_kwh"], 4) if v["consumption_kwh"] else None,
                                     "average_daily_kwh": round(v["consumption_kwh"] / n_days, 2),
                                     "records": v["records"], "device_names": sorted(v["device_names"])}
                                 for k, v in sorted(devices.items())},
            "daily": [{"date": k, "consumption_kwh": round(v["consumption_kwh"], 2), "cost_usd": round(v["cost_usd"], 2)} for k, v in sorted(days.items())],
            "hour_of_day_profile": [{"hour": k, "average_kwh": round(v["kwh"] / v["n"], 2), "average_cost_usd": round(v["cost"] / v["n"], 3)} for k, v in sorted(hours.items())],
        }
        if not records:
            result["note"] = ("no usage records in this range" + (f" for device_type {dt_filter!r}; stored types are EV, HVAC, appliance" if dt_filter else ""))
        if include_records:
            cap = max(1, min(int(max_records), 500))
            result["records"] = [{"timestamp": r.timestamp.isoformat(), "consumption_kwh": round(r.consumption_kwh, 3),
                                  "device_type": r.device_type, "device_name": r.device_name, "cost_usd": round(r.cost_usd or 0, 3)}
                                 for r in records[:cap]]
            result["records_truncated"] = len(records) > cap
        return result
    except Exception as e:
        return {"error": f"Failed to query energy usage: {e}"}


@tool
def query_solar_generation(start_date: str, end_date: str, include_records: bool = False,
                           max_records: int = 48) -> Dict[str, Any]:
    """
    Query this home's past solar generation from the database for a date range (inclusive).

    Args:
        start_date (str): Start date in YYYY-MM-DD format. For "the last 7 days", 7 days before today.
        end_date (str): End date in YYYY-MM-DD format (inclusive). For "the last 7 days", yesterday: today is not over yet.
        include_records (bool): Also return individual hourly rows (default False).
        max_records (int): Most rows to return when include_records is True (default 48).

    Returns:
        Dict[str, Any]: total generation (kWh), average per day with data, a per-day table with each day's weather,
        an hour-of-day profile, generation by weather condition, the date span the data covers, and rows only if requested.
    """
    try:
        start_dt = _parse_date(start_date, "start_date")
        end_dt = _parse_date(end_date, "end_date") + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date"}
        records = [r for r in db_manager.get_generation_by_date_range(start_dt, end_dt) if r.timestamp < end_dt]
        days: Dict[str, Dict[str, Any]] = {}
        hours: Dict[int, List[float]] = {}
        weather: Dict[str, Dict[str, float]] = {}
        for r in records:
            d = days.setdefault(r.timestamp.strftime("%Y-%m-%d"), {"generation_kwh": 0.0, "weather_condition": r.weather_condition})
            d["generation_kwh"] += r.generation_kwh
            hours.setdefault(r.timestamp.hour, []).append(r.generation_kwh)
            w = weather.setdefault(r.weather_condition or "unknown", {"generation_kwh": 0.0, "days": set()})
            w["generation_kwh"] += r.generation_kwh; w["days"].add(r.timestamp.date())
        total = sum(r.generation_kwh for r in records)
        result = {
            "start_date": start_date, "end_date": end_date, "total_records": len(records),
            "data_covers": ({"first": records[0].timestamp.isoformat(), "last": records[-1].timestamp.isoformat(), "days_with_data": len(days)} if records else None),
            "total_generation_kwh": round(total, 2),
            # Average over days that have data. The starter divided by calendar days in the range,
            # which understates the average whenever the range runs past the last stored day.
            "average_daily_generation": round(total / len(days), 2) if days else 0.0,
            "daily": [{"date": k, "generation_kwh": round(v["generation_kwh"], 2), "weather_condition": v["weather_condition"]} for k, v in sorted(days.items())],
            "hour_of_day_profile": [{"hour": h, "average_kwh": round(sum(v) / len(v), 2)} for h, v in sorted(hours.items())],
            "by_weather": {k: {"generation_kwh": round(v["generation_kwh"], 2), "days": len(v["days"]),
                               "average_daily_kwh": round(v["generation_kwh"] / len(v["days"]), 2)} for k, v in sorted(weather.items())},
        }
        if not records:
            result["note"] = "no solar records in this range"
        if include_records:
            cap = max(1, min(int(max_records), 500))
            result["records"] = [{"timestamp": r.timestamp.isoformat(), "generation_kwh": round(r.generation_kwh, 3),
                                  "weather_condition": r.weather_condition, "temperature_c": round(r.temperature_c or 0, 1),
                                  "solar_irradiance": round(r.solar_irradiance or 0, 1)} for r in records[:cap]]
            result["records_truncated"] = len(records) > cap
        return result
    except Exception as e:
        return {"error": f"Failed to query solar generation: {e}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Summarise this home's energy use and solar generation over the most recent hours of stored data.

    Args:
        hours (int): Number of hours to look back from now (1 to 720, default 24).

    Returns:
        Dict[str, Any]: usage totals and per-device breakdown, generation total and the weather recorded,
        and the time span the stored data actually covers (it can end before now).
    """
    try:
        if not isinstance(hours, int) or not 1 <= hours <= 720:
            return {"error": f"hours must be an integer from 1 to 720, got {hours!r}"}
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        breakdown: Dict[str, Dict[str, float]] = {}
        for r in usage_records:
            d = breakdown.setdefault(r.device_type or "unknown", {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0})
            d["consumption_kwh"] += r.consumption_kwh; d["cost_usd"] += r.cost_usd or 0; d["records"] += 1
        for d in breakdown.values():
            d["consumption_kwh"] = round(d["consumption_kwh"], 2); d["cost_usd"] = round(d["cost_usd"], 2)
        conditions = sorted({r.weather_condition for r in generation_records if r.weather_condition})
        # The starter always reported "sunny" here. It now reports what was recorded.
        return {
            "time_period_hours": hours,
            "data_covers": ({"first": min(r.timestamp for r in usage_records).isoformat(),
                             "last": max(r.timestamp for r in usage_records).isoformat()} if usage_records else None),
            "usage": {"total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                      "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                      "device_breakdown": breakdown},
            "generation": {"total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                           # Most frequent condition recorded in the window (the starter always said "sunny").
                           "average_weather": (max(conditions, key=lambda c: sum(r.weather_condition == c for r in generation_records))
                                               if conditions else "unknown"),
                           "weather_recorded": conditions},
        }
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {e}"}


# --------------------------------------------------------------------------- knowledge base (RAG)

def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=VOCAREUM_BASE_URL, api_key=os.getenv("VOCAREUM_API_KEY"))


def document_manifest(docs_dir: Path = DOCS_DIR) -> Dict[str, str]:
    """sha256 of every .txt document, keyed by file name. Decides whether the index is stale."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(docs_dir).glob("*.txt"))}


def load_and_split_documents(docs_dir: Path = DOCS_DIR) -> list:
    """Load every .txt file in the documents folder and split it into overlapping chunks with stable ids."""
    documents = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        for doc in TextLoader(str(path), encoding="utf-8").load():
            doc.metadata["source"] = path.name          # file name, not an absolute path
            documents.append(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    counters: Dict[str, int] = {}
    for c in chunks:
        i = counters.get(c.metadata["source"], 0)
        counters[c.metadata["source"]] = i + 1
        c.metadata["chunk"] = i
        c.id = f"{c.metadata['source']}#{i}"
    return chunks


def build_vectorstore(force: bool = False, embeddings=None) -> Chroma:
    """
    Return the persisted knowledge base, building it if it does not exist or if the documents changed.

    The starter loaded two hard-coded files and never rebuilt an existing index, so documents added later
    were silently ignored. This version indexes every .txt file and compares a manifest of file hashes:
    any added, removed or edited document triggers a rebuild. Chunk ids are stable, so a rebuild never
    stores duplicates.
    """
    embeddings = embeddings or _embeddings()
    manifest = document_manifest()
    if not manifest:
        raise FileNotFoundError(f"no .txt documents in {DOCS_DIR.relative_to(BASE_DIR)}")
    stale = force or not MANIFEST_FILE.exists() or json.loads(MANIFEST_FILE.read_text()) != manifest
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                   persist_directory=str(VECTORSTORE_DIR), collection_metadata={"hnsw:space": "cosine"})
    if stale:
        existing = store.get(include=[])["ids"]
        if existing:
            store.delete(ids=existing)
        chunks = load_and_split_documents()
        store.add_documents(chunks, ids=[c.id for c in chunks])
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return store


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the EcoHome knowledge base of energy-saving articles (EV charging, HVAC and thermostats,
    appliances, solar, batteries, seasonal tips, smart-home automation) and return the most relevant passages.

    Args:
        query (str): What to look for, e.g. "EV charging with solar panels".
        max_results (int): Number of passages to return, 1 to 8 (default 5).

    Returns:
        Dict[str, Any]: query, total_results, and tips with rank, content, source (document file name)
        and relevance_score (0 to 1, higher is more relevant). Cite the source when you use a tip.
    """
    try:
        if not query or not str(query).strip():
            return {"error": "query is required"}
        k = max(1, min(int(max_results), 8))
        store = build_vectorstore()
        hits = store.similarity_search_with_relevance_scores(query, k=k)
        return {
            "query": query, "total_results": len(hits),
            "tips": [{"rank": i + 1, "content": d.page_content, "source": d.metadata.get("source", "unknown"),
                      "relevance_score": round(float(s), 3)} for i, (d, s) in enumerate(hits)],
        }
    except Exception as e:
        return {"error": f"Failed to search energy tips: {e}"}


# --------------------------------------------------------------------------- savings

ANNUALISE = {"one_time": 0, "day": 365, "week": 52, "month": 12, "year": 1}


HISTORY_DEVICES = ("EV", "HVAC", "appliance")
# appliances are stored as type "appliance" with the machine in device_name
APPLIANCE_NAMES = {"dishwasher": "Dishwasher", "washing machine": "Washing Machine", "washer": "Washing Machine", "dryer": "Dryer"}


def _device_daily_profile(device: str, name: Optional[str] = None, days: int = 7) -> Optional[Dict[str, Any]]:
    """A stored device's average use per day, overall and in each hour of the day, from the usage history.

    Uses the last `days` complete days (midnight to midnight, ending at the start of today), so the figure
    does not depend on how much of today has passed. With `name` (e.g. "Dishwasher"), only that machine's
    records count, averaged over the days the device type has data. Returns None when there are no records.
    """
    end = datetime.combine(datetime.now().date(), datetime.min.time())
    start = end - timedelta(days=days)
    records = [r for r in db_manager.get_usage_by_date_range(start, end) if r.device_type == device and r.timestamp < end]
    n_days = len({r.timestamp.date() for r in records})
    if name:
        records = [r for r in records if r.device_name == name]
    if not n_days or not records:
        return None
    per_hour = {h: 0.0 for h in range(24)}
    for r in records:
        per_hour[r.timestamp.hour] += r.consumption_kwh
    per_hour = {h: v / n_days for h, v in per_hour.items()}
    return {"daily_kwh": sum(per_hour.values()), "per_hour_kwh": per_hour, "days": n_days,
            "first": start.date().isoformat(), "last": (end - timedelta(days=1)).date().isoformat()}


@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float = None,
                             optimized_usage_kwh: float = None, price_per_kwh: float = None,
                             optimized_price_per_kwh: float = None, period: str = "day",
                             date: str = None, current_hours: List[int] = None,
                             optimized_hours: List[int] = None, reduction_percent: float = None,
                             upfront_cost_usd: float = None, location: str = None) -> Dict[str, Any]:
    """
    Calculate savings from using less energy or from shifting the same energy to cheaper hours, and, given what a
    change costs up front, how long it takes to pay for itself (return on investment).

    For EV, HVAC and appliances leave the kWh and the prices out: the tool takes them from the data.
      - a reduction across the day: device_type, date and reduction_percent, e.g. {"device_type": "HVAC",
        "date": "2026-11-20", "reduction_percent": 10}. The usage is the device's average daily kWh over the
        last 7 complete days; the price is that date's rates weighted by the hours the device is used.
      - a reduction in certain hours only (e.g. a higher setpoint during the peak): add current_hours, e.g.
        {"device_type": "HVAC", "date": "2026-11-25", "reduction_percent": 10, "current_hours": [16, 17, 18, 19, 20]}.
        The usage is the device's average daily kWh in those hours, priced at those hours' rates.
      - a shift: device_type, date, current_hours and optimized_hours, e.g. {"device_type": "EV", "date":
        "2026-11-20", "current_hours": [16, 17, 18, 19, 20], "optimized_hours": [0, 1, 2, 3, 4, 5]}. The kWh
        moved is the device's average daily use in current_hours; each price is the average rate of its hours.
        current_hours are the dearer hours the use leaves and optimized_hours the cheaper ones it moves to; a
        shift into dearer hours is refused, because it is not a saving.
      - the cost of one stated amount (e.g. "add 40 kWh"): current_usage_kwh, period "one_time", date and
        current_hours only, set to the hours you recommend, e.g. {"device_type": "EV", "current_usage_kwh": 40,
        "period": "one_time", "date": "2026-11-20", "current_hours": [0, 1, 2, 3, 4, 5]}; current_cost_usd is the
        cost. There is no saving to report: do not invent a "before" schedule to compare against.
      - tonight's hours: list them in the order they run, from the evening past midnight, with date set to today,
        e.g. "current_hours": [22, 23, 0, 1, 2, 3, 4, 5]. The hours after midnight are priced at the next day's
        rates (tonight's 00:00-06:00 is tomorrow morning, not this morning). Hours listed in clock order from 0
        ([0, 1, 2, 22, 23]) are all the date's own hours.
      - solar: when the hours the energy is used in (optimized_hours for a shift, current_hours for one stated amount)
        include daylight hours, also pass location (e.g. "San Francisco, CA"). The tool reads that date's forecast for
        the location and counts the panels' estimated output in those hours as free, up to the kWh used; only the rest is
        priced at the grid rates. A move from the evening peak into the solar hours is then valued with the solar it
        uses, e.g. {"device_type": "EV", "date": "2026-11-20", "current_hours": [16, 17, 18, 19, 20], "optimized_hours":
        [10, 11, 12, 13, 14], "location": "San Francisco, CA"}. Without location such a call is refused.
      - payback: add upfront_cost_usd (e.g. 250 for a smart thermostat) to any of the above; payback_years is
        upfront_cost_usd / annual_savings_usd (payback_months is the same in months) and first_year_roi_percent
        is annual_savings_usd / upfront_cost_usd x 100.
    Pass current_usage_kwh with another period only for a device the history does not record (e.g. a pool pump).
    A cut and a shift are separate calculations: reduction_percent cannot be combined with optimized_hours.
    The summary sentence states the kWh, the hours (as clock ranges ending where the last hour ends, so
    hours 16-20 are "16:00-21:00") and each cost with its period: quote it rather than restating the figures.

    Args:
        device_type (str): Device being optimised. In the history: "EV", "HVAC", "appliance" (all appliances), and
            "dishwasher", "washing machine" or "dryer" (that machine only); "electric car", "thermostat" and similar
            names are accepted. Any other device needs current_usage_kwh.
        current_usage_kwh (float): Energy now in one period, in kWh. Leave out for EV, HVAC and appliances, except
            for one stated amount with period "one_time".
        optimized_usage_kwh (float): Energy after the change in one period, for a device the history does not
            record. Otherwise use reduction_percent.
        price_per_kwh (float): Price now, USD per kWh, only when no date is given. With date (and hours) the tool
            looks the rates up; passing prices as well is an error.
        optimized_price_per_kwh (float): Price after the change, only with price_per_kwh.
        period (str): What one usage figure covers: "one_time", "day", "week", "month" or "year" (default "day").
            Usage read from the history is per day.
        date (str): YYYY-MM-DD whose time-of-use rates to use.
        current_hours (List[int]): Hours of the day (0-23) the energy is used now, e.g. [16, 17, 18, 19, 20].
        optimized_hours (List[int]): Hours of the day (0-23) it would be used instead, e.g. [0, 1, 2, 3, 4, 5].
        reduction_percent (float): Share of the usage saved, 0-100 (e.g. 10 for a 10% cut). Not with optimized_usage_kwh
            or optimized_hours.
        upfront_cost_usd (float): What the change costs to make (equipment, installation), for the payback period.
        location (str): Where the home is (e.g. "San Francisco, CA"); needed whenever the hours include daylight, so
            the forecast's solar in those hours is counted.

    Returns:
        Dict[str, Any]: current and optimised cost, savings in kWh, USD and percent per period, the projected
        annual saving (negative when the change costs more), the inputs used, where the usage and the prices came
        from, the hours as clock ranges, payback_years and first_year_roi_percent when upfront_cost_usd is given (None
        when the change saves nothing), and a one-sentence summary that states each figure with its period.
    """
    if period not in ANNUALISE:
        return {"error": f"period must be one of {sorted(ANNUALISE)}, got {period!r}"}
    if optimized_usage_kwh is not None and reduction_percent is not None:
        return {"error": "give optimized_usage_kwh or reduction_percent, not both"}
    if reduction_percent is not None:
        try:
            reduction_percent = float(reduction_percent)
        except (TypeError, ValueError):
            return {"error": "reduction_percent must be a number from 0 to 100"}
        if not 0 <= reduction_percent <= 100:
            return {"error": "reduction_percent must be a number from 0 to 100"}
    if reduction_percent is not None and optimized_hours:
        # a live run combined a shift with reduction_percent 100: the 'move' then cost $0.00 afterwards and the whole
        # cost was reported as the saving from moving it
        return {"error": "reduction_percent cannot be combined with optimized_hours: a shift moves the same kWh to other hours. "
                         "Value the shift (current_hours and optimized_hours) and any cut (reduction_percent) as separate calculations"}
    if upfront_cost_usd is not None:
        try:
            upfront_cost_usd = float(upfront_cost_usd)
        except (TypeError, ValueError):
            return {"error": "upfront_cost_usd must be a number"}
        if upfront_cost_usd <= 0:
            return {"error": "upfront_cost_usd must be more than 0"}
    shifting = bool(current_hours) and bool(optimized_hours)
    reducing_in_hours = bool(current_hours) and not optimized_hours and reduction_percent is not None
    # the cost of one stated amount in the hours given: no "before" and "after" (a live answer invented a 00:00-07:00
    # "current" schedule to report a $0.26 saving on a question that only asked what 40 kWh would cost)
    pricing = (bool(current_hours) and not optimized_hours and reduction_percent is None and period == "one_time"
               and current_usage_kwh is not None and optimized_usage_kwh is None)
    if (current_hours or optimized_hours) and not (shifting or reducing_in_hours or pricing):
        return {"error": "for a shift, give current_hours and optimized_hours together (with date to look up the rates); "
                         "for a reduction in certain hours, give current_hours with reduction_percent; for the cost of one "
                         "stated amount, give current_usage_kwh, period 'one_time', date and current_hours"}
    hours_now = hours_new = None
    if current_hours or optimized_hours:
        try:
            hours_now = [int(h) for h in current_hours]
            hours_new = [int(h) for h in optimized_hours] if shifting else list(hours_now)
        except (TypeError, ValueError):
            return {"error": "current_hours and optimized_hours must be lists of whole hours 0-23"}
        if not all(0 <= h <= 23 for h in hours_now + hours_new):
            return {"error": "hours must be between 0 and 23"}
    device = _normalise_device_type(device_type)
    machine = APPLIANCE_NAMES.get(str(device_type).strip().lower())
    label = machine or device
    in_history = device in HISTORY_DEVICES
    # Two inputs the model supplied wrongly in live runs, refused here rather than caught afterwards:
    # prices typed in next to a date (two off-peak rates gave a 'saving' of -$0.21 a year for moving charging
    # OFF the peak), and a recorded device's kWh typed in for a recurring period (a daily average labelled
    # 'year', a week's total labelled 'day', every appliance's use labelled 'dishwasher').
    if (price_per_kwh is not None or optimized_price_per_kwh is not None) and (date or hours_now):
        return {"error": "leave price_per_kwh and optimized_price_per_kwh out: with date the tool looks up that date's "
                         "time-of-use rates for the hours given, and prices passed as well would replace them"}
    if in_history and current_usage_kwh is None and (price_per_kwh is not None or optimized_price_per_kwh is not None):
        # a live run typed in $0.1644 for HVAC (no date): the history's own device is priced from that date's table
        return {"error": f"leave price_per_kwh out for {label}: give date and the tool prices its recorded use with that date's "
                         "time-of-use rates, weighted by the hours it is used"}
    if in_history and period != "one_time" and (current_usage_kwh is not None or optimized_usage_kwh is not None):
        return {"error": f"leave current_usage_kwh and optimized_usage_kwh out for {label}: the tool reads its average daily "
                         "use from the history (period 'day'); use reduction_percent for a cut and current_hours/optimized_hours "
                         "for a shift. A kWh you supply is accepted only with period 'one_time', for a single stated amount"}
    needs_history = in_history and current_usage_kwh is None
    try:
        profile = _device_daily_profile(device, machine) if needs_history else None
        # read only to recognise the device's own figures; a stated amount is still priced at the plain average of its hours
        recorded = _device_daily_profile(device, machine) if pricing and in_history and not needs_history else None
    except Exception as e:
        return {"error": f"could not read the usage history: {e}"}
    if recorded is not None:
        # One stated amount is priced as given ("add 40 kWh"). A recorded device's own daily use typed in as that amount is
        # not a stated amount: two live 'three ways' answers priced the appliances' 32.78 kWh a day at the peak and then
        # said moving them off-peak saved nothing, because a price for one set of hours compares nothing.
        try:
            typed = float(current_usage_kwh)
        except (TypeError, ValueError):
            return {"error": "usage and prices must be numbers"}
        own = {"average daily use": recorded["daily_kwh"],
               f"average daily use in {hour_window(current_hours)}": sum(recorded["per_hour_kwh"][int(h)] for h in set(current_hours)
                                                                         if 0 <= int(h) <= 23)}
        for what, value in own.items():
            if abs(typed - round(value, 2)) < 0.006:
                return {"error": f"{typed:g} kWh is {label}'s {what} from the history, not a stated amount: to value moving that use, "
                                 "leave current_usage_kwh out and give date, current_hours (the dearer hours it leaves, e.g. the peak "
                                 "[16, 17, 18, 19, 20]) and optimized_hours (the cheaper hours, e.g. [0, 1, 2, 3, 4, 5]); the tool reads "
                                 "the use in current_hours itself"}

    # ---- usage: from the history (then per day), or from the caller
    if current_usage_kwh is None:
        if not in_history:
            return {"error": f"the usage history records only {', '.join(HISTORY_DEVICES)}; for {device_type!r} pass current_usage_kwh and the period it covers"}
        if profile is None:
            return {"error": f"no {label} records in the last 7 complete days; pass current_usage_kwh with period 'one_time' for a stated amount"}
        period = "day"
        span = f"over {profile['first']} to {profile['last']} ({profile['days']} days of history)"
        if hours_now:
            cur = sum(profile["per_hour_kwh"][h] for h in set(hours_now))
            usage_from = f"average daily {label} use in hours {sorted(set(hours_now))} ({hour_window(hours_now)}) {span}"
        else:
            cur = profile["daily_kwh"]
            usage_from = f"average daily {label} use {span}"
    else:
        try:
            cur = float(current_usage_kwh)
        except (TypeError, ValueError):
            return {"error": "usage and prices must be numbers"}
        usage_from = "usage supplied by the caller"
    try:
        if optimized_usage_kwh is not None:
            opt = float(optimized_usage_kwh)
        else:
            opt = cur * (1 - reduction_percent / 100) if reduction_percent is not None else cur
    except (TypeError, ValueError):
        return {"error": "usage and prices must be numbers"}

    # ---- prices: that date's time-of-use rates, or the caller's when no date is given
    night = None   # the next day's table when hours run past midnight
    if price_per_kwh is not None:
        try:
            p_now = float(price_per_kwh)
            p_new = p_now if optimized_price_per_kwh is None else float(optimized_price_per_kwh)
        except (TypeError, ValueError):
            return {"error": "usage and prices must be numbers"}
        prices_from = "prices supplied by the caller"
    elif date:
        table = get_electricity_prices.invoke({"date": date})
        if "error" in table:
            return {"error": f"could not look up prices: {table['error']}"}

        def next_table(t):
            """The following day's table: hours listed past midnight ([22, 23, 0, 1, ...]) are that night's, so the
            hours after the wrap are the next date's. A live answer priced 'from 22:00 tonight to 06:00 tomorrow' with
            this morning's 00:00-06:00 rates ($0.1124 average; the night's real average was $0.1083)."""
            if not (_wrap_at(hours_now) or _wrap_at(hours_new)):
                return None
            nxt = get_electricity_prices.invoke({"date": (_parse_date(t["date"], "date") + timedelta(days=1)).strftime("%Y-%m-%d")})
            return None if "error" in nxt else nxt

        def rates_for(table, nxt=None):
            """(price now, price after, how) from one date's time-of-use table; the same rule for any date.

            With nxt, the hours a list gives after it wraps past midnight are priced from nxt (the night that starts on
            the table's date)."""
            rate = {r["hour"]: r["rate"] for r in table["hourly_rates"]}
            later = {r["hour"]: r["rate"] for r in nxt["hourly_rates"]} if nxt else rate
            day = f"time-of-use rates for {table['date']} ({table['day_of_week']})"
            if nxt and (_wrap_at(hours_now) or _wrap_at(hours_new)):
                day += f" until midnight, then {nxt['date']} ({nxt['day_of_week']})"

            def seq(hours):
                """{hour: rate} for a list in the order given; hours after a wrap past midnight are the next day's."""
                k = _wrap_at(hours) if nxt else None
                out = {}
                for i, h in enumerate(hours):
                    out.setdefault(h, later[h] if k is not None and i >= k else rate[h])
                return out
            if shifting:
                rn, rw = seq(hours_now), seq(hours_new)
                return (sum(rn.values()) / len(rn), sum(rw.values()) / len(rw),
                        f"{day}: average of hours {hours_now} now, {hours_new} after the change")
            if hours_now:
                w = profile["per_hour_kwh"] if profile is not None else {}
                rn = seq(hours_now)
                weight = sum(w.get(h, 0) for h in rn)
                pr = (sum(r * w.get(h, 0) for h, r in rn.items()) / weight if weight > 0
                      else sum(rn.values()) / len(rn))
                return pr, pr, f"{day}: hours {hours_now} ({hour_window(hours_now)})" + (" weighted by use" if weight > 0 else " averaged")
            if profile is not None and profile["daily_kwh"] > 0:
                # a reduction applies across the hours the device is used, so weight each hour's rate by its use
                w = profile["per_hour_kwh"]
                pr = sum(rate[h] * w[h] for h in range(24)) / sum(w.values())
                return pr, pr, f"{day}: weighted by the hours {label} is used in the history (the whole day)"
            pr = sum(rate.values()) / 24
            return pr, pr, f"{day}: average of all 24 hours"
        night = next_table(table)
        p_now, p_new, prices_from = rates_for(table, night)
    else:
        return {"error": "a price is needed: give date so the tool looks up the time-of-use rates "
                         "(with current_hours and optimized_hours for a shift), or pass price_per_kwh from get_electricity_prices"}
    if min(cur, opt) < 0 or min(p_now, p_new) < 0:
        return {"error": "usage and prices cannot be negative"}
    # ---- solar: energy used in daylight hours is covered first by the panels' forecast output in those hours
    # (live answers moved EV charging "to the solar hours" and could only price it as mid-peak grid power, because this
    # tool knew nothing of the panels; the question it answers asks to maximise solar)
    # Solar counts only where the energy is used after the change (optimized_hours; current_hours for one stated amount):
    # the hours it leaves are priced as the history metered them, so a move from the peak to off-peak is unchanged.
    solar_now = solar_new = 0.0
    solar_note = ""
    dest = (hours_new if shifting else hours_now) or []
    core = set(CORE_SOLAR_HOURS)
    # A move OUT of the solar hours gives up solar the use had: a live example moved 24.72 kWh of EV charging from 11:00-14:00
    # to off-peak and reported $1.10 a day saved, pricing the midday hours as if the panels were not there.
    leaving = set(hours_now or []) & core if shifting else set()
    if (shifting or pricing) and date and price_per_kwh is None and (set(dest) & core or leaving):
        if not location or not str(location).strip():
            return {"error": "these hours include the solar hours (09:00-16:00), when the panels produce: pass location (e.g. "
                             "'San Francisco, CA') so the tool counts the forecast's solar in those hours before pricing the rest at grid rates"}
        try:
            sun = _solar_by_hour(str(location), date)
        except Exception as e:
            return {"error": f"could not read the forecast's solar for {date}: {e}"}
        in_hours = lambda hs: sum(sun.get(h, 0.0) for h in set(hs))
        solar_new = min(opt, in_hours(hours_new)) if shifting else 0.0
        solar_now = min(cur, in_hours(hours_now)) if pricing else min(cur, in_hours(leaving))
        if solar_now or solar_new:
            solar_note = (f"solar from the {location} forecast for {date}, assuming the panels' output in those hours goes to this "
                          "use first and only the rest is bought from the grid")
    cost_now, cost_new = (cur - solar_now) * p_now, (opt - solar_new) * p_new
    if pricing:
        cost_new = cost_now
    if shifting and date and price_per_kwh is None and cost_new > cost_now + 1e-9:
        # A shift values moving use OUT of dearer hours. A live 'three ways' answer called this with current_hours
        # 22:00-06:00 (off-peak, where the appliances already run) and optimized_hours 16:00-21:00 (peak), then printed
        # "would cost an additional $1.53 per day" under its advice to run the appliances off-peak.
        # the costs compared include the solar on each side: a live call moved EV charging out of 11:00-14:00 and read
        # "00:00-06:00 ($0.1073/kWh) cost more than 11:00-14:00 ($0.1519/kWh)", true only once the midday solar is counted
        return {"error": f"optimized_hours {hour_window(hours_new)} (${cost_new:.2f} a day: {opt - solar_new:.2f} kWh from the grid at "
                         f"${p_new:.4f}/kWh" + (f", {solar_new:.2f} kWh from solar" if solar_new else "") + f") cost more than current_hours "
                         f"{hour_window(hours_now)} (${cost_now:.2f} a day: {cur - solar_now:.2f} kWh from the grid at ${p_now:.4f}/kWh"
                         + (f", {solar_now:.2f} kWh from the forecast's solar there" if solar_now else "") + f"): moving {cur:.2f} kWh a day there would cost "
                         f"${cost_new - cost_now:.2f} more, which is not a saving. Set current_hours to the dearer hours the use "
                         "would leave (for example the peak [16, 17, 18, 19, 20]) and optimized_hours to the cheaper hours it "
                         "moves to (for example off-peak [0, 1, 2, 3, 4, 5]); the tool reads how much is used in current_hours"}
    saving = cost_now - cost_new
    annual = saving * ANNUALISE[period]
    # A daily habit read from the history recurs on weekdays and at weekends, whose tariffs differ (no peak at weekends):
    # the yearly figure prices each of the next 7 days and scales the week to 365 days. Live answers valued the same EV move at $1035.13 a year on a Saturday's rates and
    # $3249.68 on a Friday's, in one notebook; a payback depended on which day the model picked.
    blend = None
    if (date and price_per_kwh is None and period == "day" and needs_history and not pricing
            and not (solar_now or solar_new)):
        # the same 7 days (from today) for every calculation, so two answers valuing one change agree on its year
        start = datetime.combine(datetime.now().date(), datetime.min.time())
        week = []
        for k in range(7):
            t_k = get_electricity_prices.invoke({"date": (start + timedelta(days=k)).strftime("%Y-%m-%d")})
            if "error" in t_k:
                week = []
                break
            pn, pw, _ = rates_for(t_k, next_table(t_k))
            week.append(((start + timedelta(days=k)).weekday() >= 5, cur * pn - opt * pw))
        if week:
            annual = sum(v for _, v in week) * 365 / 7
            wd = [v for w, v in week if not w]
            we = [v for w, v in week if w]
            blend = {"weekday_savings_usd": round(sum(wd) / len(wd), 2), "weekend_savings_usd": round(sum(we) / len(we), 2),
                     "year_priced_from": f"{start:%Y-%m-%d} to {start + timedelta(days=6):%Y-%m-%d}",
                     "date_weekday": _parse_date(date, "date").strftime("%A")}
    per = period.replace("_", " ")
    each = {"one_time": "", "day": " a day", "week": " a week", "month": " a month", "year": " a year"}[period]
    win_now, win_new = (hour_window(hours_now), hour_window(hours_new)) if hours_now else ("", "")
    # The summary names what each figure belongs to, so a quoted cost cannot drift onto another kWh figure
    # (a live answer put the $15.85 cost of 61.63 kWh of peak charging next to the EV's 218.91 kWh daily total).
    if pricing:
        what = None
    elif shifting:
        what = f"Moving {cur:.2f} kWh{each} of {label} use from {win_now} to {win_new}"
    elif reduction_percent is not None or opt != cur:
        cut = f"{reduction_percent:g}% less" if reduction_percent is not None else "less"
        where = f" in {win_now}" if hours_now else (" across the whole day" if usage_from.startswith("average daily") else "")
        what = f"Using {cut} {label} energy{where} ({cur:.2f} kWh{each} down to {opt:.2f} kWh, same price)"
    else:
        what = f"{cur:.2f} kWh of {label} use" + (f" in {win_now}" if win_now else "")
    verb = f"saves ${saving:.2f}" if saving >= 0 else f"costs ${-saving:.2f} more"
    if pricing and solar_now:
        summary = (f"{cur:.2f} kWh of {label} use in {win_now} costs ${cost_now:.2f}: {solar_now:.2f} kWh from the forecast's solar and "
                   f"{cur - solar_now:.2f} kWh from the grid at an average ${p_now:.4f}/kWh over those hours.")
    elif pricing:
        summary = f"{cur:.2f} kWh of {label} use in {win_now} costs ${cost_now:.2f} (an average ${p_now:.4f}/kWh over those hours)."
    elif solar_new or solar_now:
        now_part = (f"${cost_now:.2f} now at ${p_now:.4f}/kWh" + (f" after {solar_now:.2f} kWh from the forecast's solar in "
                    f"{hour_window(leaving)}" if solar_now else ""))
        new_part = f"${cost_new:.2f} after: {solar_new:.2f} kWh from the forecast's solar in {win_new} and {opt - solar_new:.2f} kWh from the grid at ${p_new:.4f}/kWh"
        summary = (f"{what} {verb} per {per} ({now_part}; {new_part})"
                   + ((f", which is ${annual:.2f} a year for days like {date}." if annual >= 0 else f", which is ${-annual:.2f} a year more.")
                      if period not in ("year", "one_time") else "."))
    else:
        if blend:
            yearly = (f" on {blend['date_weekday']} {date}; over the 7 days {blend['year_priced_from']} a weekday saves "
                      f"${blend['weekday_savings_usd']:.2f} on average and a weekend day ${blend['weekend_savings_usd']:.2f}, which is "
                      + (f"${annual:.2f} a year." if annual >= 0 else f"${-annual:.2f} a year more."))
        else:
            yearly = ((f", which is ${annual:.2f} a year." if annual >= 0 else f", which is ${-annual:.2f} a year more.")
                      if period not in ("year", "one_time") else ".")
        summary = f"{what} {verb} per {per} (${cost_now:.2f} now at ${p_now:.4f}/kWh, ${cost_new:.2f} after at ${p_new:.4f}/kWh)" + yearly
    if date and price_per_kwh is None and night is not None:
        summary += (f" The hours after midnight are priced at {night['day_of_week']} {night['date']}'s rates (the night that "
                    f"starts on {_parse_date(date, 'date'):%A} {date}).")
    payback = months = roi = None
    if upfront_cost_usd is not None:
        if annual > 0:
            payback, roi = round(upfront_cost_usd / annual, 2), round(annual / upfront_cost_usd * 100, 1)
            months = round(upfront_cost_usd / annual * 12, 1)   # from the unrounded figure, not 12 x the rounded years
            summary += (f" An upfront cost of ${upfront_cost_usd:.2f} pays for itself in {payback:.2f} years ({months:.1f} months; "
                        f"{roi:.1f}% of the cost back each year).")
        else:
            summary += f" An upfront cost of ${upfront_cost_usd:.2f} is never paid back: the change saves nothing a year."
    return {
        "device_type": device_type, "period": period,
        "current_usage_kwh": round(cur, 2), "optimized_usage_kwh": round(opt, 2),
        "usage_from": usage_from, "reduction_percent": reduction_percent,
        "current_hours": hours_now, "optimized_hours": hours_new if shifting else None,
        "current_window": win_now or None, "optimized_window": (win_new if shifting else None) or None,
        "price_per_kwh": round(p_now, 4), "optimized_price_per_kwh": round(p_new, 4),
        "prices_from": prices_from,
        "current_cost_usd": round(cost_now, 2), "optimized_cost_usd": round(cost_new, 2),
        "savings_kwh": round(cur - opt, 2),
        "savings_usd": round(saving, 2),
        "savings_percentage": round(saving / cost_now * 100, 1) if cost_now > 0 else 0.0,
        "annual_savings_usd": round(annual, 2),
        "upfront_cost_usd": upfront_cost_usd, "payback_years": payback, "payback_months": months, "first_year_roi_percent": roi,
        "solar_kwh_now": round(solar_now, 2), "solar_kwh_after": round(solar_new, 2), "solar_from": solar_note or None,
        "annual_basis": (f"the 7 days {blend['year_priced_from']} (weekdays and weekend days) x 365/7"
                         if blend else (f"365 days like {date}" if date and period == "day" else None)),
        **({k: v for k, v in blend.items() if k in ("weekday_savings_usd", "weekend_savings_usd")} if blend else {}),
        "note": f"savings_usd is per {per}; annual_savings_usd is per year" + ("; negative means the change costs more" if saving < 0 else ""),
        "summary": summary,
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
