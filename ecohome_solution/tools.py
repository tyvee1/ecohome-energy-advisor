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


def _daylight_curve(hour: int) -> float:
    """0 at night, rising to 1 at 12:00; matches the seeded solar data (generation from 06:00 to 18:00)."""
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, 1 - abs(hour - 12) / 6)


# --------------------------------------------------------------------------- weather

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
        daily.append({"date": day, "condition": condition, "estimated_solar_kwh": round(day_kwh, 2),
                      "best_solar_hours": [h["hour"] for h in hourly[-24:] if h["estimated_solar_kwh"] >= 0.6 * max(1e-9, max(x["estimated_solar_kwh"] for x in hourly[-24:]))]})
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

    Args:
        location (str): City and region, e.g. "San Francisco, CA".
        days (int): Number of days to forecast, 1 to 7. Use 2 when the question is about tomorrow.

    Returns:
        Dict[str, Any]: location, forecast_days, source ("simulated" or "open-meteo"),
        current conditions, a daily summary (condition, estimated_solar_kwh, best_solar_hours),
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
        and a summary with the cheapest and most expensive hours and the average rate.
        Off-peak is 22:00-06:00 every day; peak is 16:00-21:00 on weekdays only.
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
        },
    }


# --------------------------------------------------------------------------- history (database)

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None,
                       include_records: bool = False, max_records: int = 48) -> Dict[str, Any]:
    """
    Query this home's past energy consumption from the database for a date range (inclusive).

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        device_type (str): Optional filter: "EV", "HVAC" or "appliance" (case and common names such as "electric car" are accepted).
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
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        records = [r for r in records if r.timestamp < end_dt]
        if dt_filter:
            records = [r for r in records if r.device_type == dt_filter]
        devices: Dict[str, Dict[str, Any]] = {}
        days: Dict[str, Dict[str, float]] = {}
        hours: Dict[int, Dict[str, float]] = {}
        for r in records:
            d = devices.setdefault(r.device_type or "unknown", {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0, "device_names": set()})
            d["consumption_kwh"] += r.consumption_kwh; d["cost_usd"] += r.cost_usd or 0; d["records"] += 1
            if r.device_name:
                d["device_names"].add(r.device_name)
            day = days.setdefault(r.timestamp.strftime("%Y-%m-%d"), {"consumption_kwh": 0.0, "cost_usd": 0.0})
            day["consumption_kwh"] += r.consumption_kwh; day["cost_usd"] += r.cost_usd or 0
            h = hours.setdefault(r.timestamp.hour, {"kwh": 0.0, "cost": 0.0, "n": 0})
            h["kwh"] += r.consumption_kwh; h["cost"] += r.cost_usd or 0; h["n"] += 1
        n_days = max(1, len(days))
        result = {
            "start_date": start_date, "end_date": end_date, "device_type": dt_filter,
            "total_records": len(records),
            "data_covers": ({"first": records[0].timestamp.isoformat(), "last": records[-1].timestamp.isoformat(), "days_with_data": len(days)} if records else None),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
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
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
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


@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float,
                             optimized_usage_kwh: float, price_per_kwh: float = 0.12,
                             optimized_price_per_kwh: float = None, period: str = "day",
                             date: str = None, current_hours: List[int] = None,
                             optimized_hours: List[int] = None) -> Dict[str, Any]:
    """
    Calculate savings from using less energy, from shifting the same energy to cheaper hours, or both.

    For a time shift, give date, current_hours and optimized_hours: the tool then looks up that day's
    time-of-use rates itself and averages them over each set of hours, so no rate has to be copied by hand.

    Args:
        device_type (str): Device being optimised, e.g. "EV", "HVAC", "dishwasher".
        current_usage_kwh (float): Energy used now, per period, in kWh.
        optimized_usage_kwh (float): Energy used after the change, per period, in kWh (equal to current when only the time changes).
        price_per_kwh (float): Price paid now, USD per kWh (default 0.12). Ignored when current_hours and date are given.
        optimized_price_per_kwh (float): Price after the change. Defaults to price_per_kwh. Ignored when optimized_hours and date are given.
        period (str): What one usage figure covers: "one_time", "day", "week", "month" or "year" (default "day").
        date (str): YYYY-MM-DD whose time-of-use rates to use with current_hours and optimized_hours.
        current_hours (List[int]): Hours of the day (0-23) the energy is used now, e.g. [16, 17, 18, 19, 20].
        optimized_hours (List[int]): Hours of the day (0-23) it would be used instead, e.g. [0, 1, 2, 3, 4, 5].

    Returns:
        Dict[str, Any]: current and optimised cost, savings in kWh, USD and percent per period, the projected
        annual saving, the inputs used, where the prices came from, and a one-sentence summary that states
        each figure with its period.
    """
    if period not in ANNUALISE:
        return {"error": f"period must be one of {sorted(ANNUALISE)}, got {period!r}"}
    try:
        cur, opt, p_now = float(current_usage_kwh), float(optimized_usage_kwh), float(price_per_kwh)
        p_new = p_now if optimized_price_per_kwh is None else float(optimized_price_per_kwh)
    except (TypeError, ValueError):
        return {"error": "usage and prices must be numbers"}
    prices_from = "prices supplied by the caller"
    if current_hours or optimized_hours:
        if not (date and current_hours and optimized_hours):
            return {"error": "to look up time-of-use rates, give date, current_hours and optimized_hours together"}
        try:
            hours_now, hours_new = [int(h) for h in current_hours], [int(h) for h in optimized_hours]
        except (TypeError, ValueError):
            return {"error": "current_hours and optimized_hours must be lists of whole hours 0-23"}
        if not all(0 <= h <= 23 for h in hours_now + hours_new):
            return {"error": "hours must be between 0 and 23"}
        table = get_electricity_prices.invoke({"date": date})
        if "error" in table:
            return {"error": f"could not look up prices: {table['error']}"}
        rate = {r["hour"]: r["rate"] for r in table["hourly_rates"]}
        p_now = sum(rate[h] for h in hours_now) / len(hours_now)
        p_new = sum(rate[h] for h in hours_new) / len(hours_new)
        prices_from = (f"time-of-use rates for {table['date']} ({table['day_of_week']}): average of hours "
                       f"{hours_now} now, {hours_new} after the change")
    if min(cur, opt) < 0 or min(p_now, p_new) < 0:
        return {"error": "usage and prices cannot be negative"}
    cost_now, cost_new = cur * p_now, opt * p_new
    saving = cost_now - cost_new
    annual = saving * ANNUALISE[period]
    per = period.replace("_", " ")
    summary = (f"Saving ${saving:.2f} per {per} (${cost_now:.2f} now, ${cost_new:.2f} after)"
               + (f", which is ${annual:.2f} per year." if period not in ("year", "one_time") else "."))
    return {
        "device_type": device_type, "period": period,
        "current_usage_kwh": cur, "optimized_usage_kwh": opt,
        "price_per_kwh": round(p_now, 4), "optimized_price_per_kwh": round(p_new, 4),
        "prices_from": prices_from,
        "current_cost_usd": round(cost_now, 2), "optimized_cost_usd": round(cost_new, 2),
        "savings_kwh": round(cur - opt, 2),
        "savings_usd": round(saving, 2),
        "savings_percentage": round(saving / cost_now * 100, 1) if cost_now > 0 else 0.0,
        "annual_savings_usd": round(annual, 2),
        "note": f"savings_usd is per {per}; annual_savings_usd is per year",
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
