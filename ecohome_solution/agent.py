"""
EcoHome Energy Advisor agent, built as an explicit LangGraph graph.

The starter used LangGraph's prebuilt ``create_react_agent``. That works, but it hides
the three things this project asks you to define: the state schema, the nodes and the
edges. This file builds the tool-calling loop by hand so each part is visible, and adds a
validation step before an answer is accepted:

    START -> agent --(tool calls?)--> tools -> agent -> ...
                   \\--(answer)--> validate --(required data missing, retries left)--> agent
                                           \\--(complete, or retries used up)--> END

* **Schema** (``AdvisorState``): the conversation (``messages``, merged by the
  ``add_messages`` reducer so each node only returns what it adds), plus the
  context the advisor works in (``context``) and a counter of model turns
  (``steps``) used to stop a run that keeps calling tools.
* **Nodes**: ``agent`` calls the model with the system prompt, the context and the
  conversation; ``tools`` runs the tool calls the model asked for.
* **Edges**: a conditional edge after ``agent`` goes to ``tools`` when the last
  message contains tool calls and to ``validate`` otherwise; ``tools`` always returns
  to ``agent``; ``validate`` sends the run back to ``agent`` or ends it.
* **Validation** (``validate`` node): a model can answer from what it already knows and
  skip a data source the answer depends on. ``DATA_RULES`` names, for each kind of
  question, the tools whose results the answer must rest on (a quantified saving needs
  the price table and ``calculate_energy_savings``; suggestions need the knowledge base;
  using more of your own solar needs the forecast and the solar history). If a required
  tool has not returned a successful result, the node sends the model back with a note
  naming the missing tools and why, at most ``max_validation_retries`` times. It also checks
  that the answer USES what came back, where code can tell: every cited tip file must be one
  that search_energy_tips returned in this conversation, and an answer giving suggestions must
  cite at least one; an answer about using more of your own solar must quote a figure from the
  solar history; an answer with a saving must quote a figure from calculate_energy_savings, and
  that calculation must use the time-of-use price table (looked up by the tool, or rates that
  appear in it) and the right period for a daily figure; a calculation that describes no change
  is sent back. Whatever is still missing or unresolved after the retries is recorded in
  ``missing_required_tools`` and ``content_issues`` rather than hidden.

Context awareness: every run is told today's date, the day of the week and the
current hour, because "tomorrow" or "this week" cannot be answered without them,
and a model does not know the date on its own.

Error handling: a tool that raises is turned into an error message the model can
read (``ToolNode(handle_tool_errors=True)``); the number of model turns is capped
(``max_steps``); a failure calling the model is returned as a readable final
message with ``error`` set in the result, rather than crashing the caller.
"""
import json
import os
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOL_KIT

load_dotenv()

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


# Data each kind of question must be answered from: (name, pattern on the question, required tools, why).
# These are the advisor's own rules, written from what each answer needs, not from any test case.
DATA_RULES = [
    ("quantified savings",
     re.compile(r"\bhow much (?:can|could|would|will|do) i save\b|\bsavings\b|\breduce (?:my )?(?:energy use|energy usage|electricity use|bill|costs?)\b"
                r"|\bpay (?:for )?itself\b|\bpay ?back\b|\bROI\b|\breturn on (?:the )?investment\b|\bworth (?:it|buying)\b", re.I),
     ("get_electricity_prices", "calculate_energy_savings"),
     "a saving must be calculated with calculate_energy_savings from the time-of-use price table"),
    ("usage history",
     re.compile(r"\busage history\b|\bmy usage\b|\bbased on my (?:usage|history|energy use)\b", re.I),
     ("query_energy_usage",),
     "the answer must rest on the household's recorded usage"),
    ("own solar",
     re.compile(r"\bmy own solar\b|\binstead of buying from the grid\b|\bself[- ]consum", re.I),
     ("get_weather_forecast", "query_solar_generation", "query_energy_usage"),
     "using more solar needs the forecast, the panels' recorded generation, and the household's use it is set against"),
    ("a specific day",
     re.compile(r"\b(?:tomorrow|tonight|this week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|this weekend)\b", re.I),
     ("get_weather_forecast",),
     "a plan for a specific day depends on that day's forecast (solar generation and temperature)"),
    ("prices",
     re.compile(r"\bprices?\b|\brates\b|\bcheapest\b|\bminimi[sz]e (?:the )?cost\b|\boff-peak\b|\bpeak\b", re.I),
     ("get_electricity_prices",),
     "an answer about prices or peak hours must use that day's time-of-use rates"),
    ("suggestions",
     re.compile(r"\b(?:suggest|suggestions?|ways to|ways i can|tips|best practices?)\b", re.I),
     ("search_energy_tips",),
     "suggestions must come from the knowledge base, citing each tip's source file"),
]


def required_tools(question: str) -> list:
    """Tools whose successful results an answer to this question must rest on, in a stable order."""
    needed = []
    for _, pattern, tools, _ in DATA_RULES:
        if pattern.search(question or ""):
            needed += [t for t in tools if t not in needed]
    return needed


def _reasons(question: str, missing: list) -> str:
    return "; ".join(f"{', '.join(t for t in tools if t in missing)}: {why}"
                     for _, pattern, tools, why in DATA_RULES
                     if pattern.search(question or "") and any(t in missing for t in tools))


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITED = re.compile(r"\btip_[A-Za-z0-9_]+\.txt\b")
# "(source: energy_saving_tips)", "Source: the knowledge base": any attribution, checked against the files a search returned
SOURCE_CLAIM = re.compile(r"\bsources?\s*:\s*([^)\n]+)", re.I)


def _numbers(text: str) -> set:
    """Numbers in a text that can identify data: decimals, or whole numbers above 31 (not hours or days)."""
    out = set()
    for raw in NUMBER.findall(text or ""):
        value = float(raw.replace(",", ""))
        if "." in raw or value > 31:
            out.add(value)
    return out


def _payloads(messages, tool_name: str) -> list:
    """Parsed results of one tool's successful calls."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == tool_name and m.status != "error":
            try:
                data = json.loads(m.content) if isinstance(m.content, str) else m.content
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and "error" not in data:
                out.append(data)
    return out


DATA_KEY_WORDS = ("kwh", "usd", "rate", "price", "generation", "temperature", "cost", "saving", "irradiance", "percent", "payback")
COUNT_KEY_WORDS = ("record", "hour", "days", "count", "dimension", "relevance")


def data_figures(payload) -> set:
    """The figures in a tool result that describe the data: kWh, USD, rates, generation, temperatures.

    Counts, hours, record totals and dates are left out, so an answer saying "78°F" is not taken to quote a
    result that happened to contain 78 records. Walks nested dicts and lists; the key that holds a number decides.
    """
    out = set()
    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, str(k).lower())
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and x:
            if any(w in key for w in DATA_KEY_WORDS) and not any(w in key for w in COUNT_KEY_WORDS):
                out.add(float(x))
    walk(payload)
    return out


def same_figure(a: float, b: float) -> bool:
    """Two figures are the same as printed: to the hundredth for kWh and dollars, to 4 decimals for a rate under $1."""
    return abs(a - b) < (0.006 if max(abs(a), abs(b)) >= 1 else 0.00006)


def _quotes(found: set, wanted: set) -> bool:
    """True when the answer states one of the wanted figures as printed."""
    return any(same_figure(a, b) for a in found for b in wanted)


def _matches(found: set, wanted: set) -> bool:
    return any(abs(a - b) <= max(0.011, 0.001 * abs(b)) for a in found for b in wanted)


PERIOD_DAYS = {"day": (1, 1), "week": (7, 7), "month": (28, 31), "year": (365, 366)}


def _quoted_calcs(messages, answer: str) -> list:
    """calculate_energy_savings results whose saving (per period or per year) the answer states."""
    found = _numbers(answer)
    # magnitudes: "$-0.21" and "costs $0.21 more" both read as 0.21, and a -0.21 saving must still count as quoted
    figures = lambda c: {abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd") if isinstance(c.get(k), (int, float)) and c.get(k)}
    return [c for c in _payloads(messages, "calculate_energy_savings") if _matches(found, figures(c))]


def usage_period_errors(messages, answer: str) -> list:
    """Quoted savings calculations whose usage figure and period do not describe the same span of time.

    calculate_energy_savings multiplies the saving per period by the periods in a year, so the usage it is
    given must be the usage for ONE period. Two mistakes are caught, both against query_energy_usage results:
      - a daily figure (a device's average_daily_kwh, or one day's consumption) given with a period other than 'day';
      - a multi-day total (a device's consumption_kwh, or total_consumption_kwh, over N days) given with a period
        that is not N days long, e.g. last week's total charging passed as one day's, which multiplies the
        saving by seven and the annual figure by 365.
    """
    issues, daily, totals = [], set(), []
    for d in _payloads(messages, "query_energy_usage"):
        daily |= {v.get("average_daily_kwh") for v in d.get("device_breakdown", {}).values() if isinstance(v, dict)}
        daily |= {v.get("consumption_kwh") for v in d.get("daily", []) if isinstance(v, dict)}
        days = (d.get("data_covers") or {}).get("days_with_data") or 0
        if days > 1:
            for name, v in d.get("device_breakdown", {}).items():
                if isinstance(v, dict) and v.get("consumption_kwh"):
                    totals.append((v["consumption_kwh"], days, f"{name} total", v.get("average_daily_kwh")))
            if d.get("total_consumption_kwh"):
                totals.append((d["total_consumption_kwh"], days, "all-device total", round(d["total_consumption_kwh"] / days, 2)))
    daily.discard(None)
    for c in _quoted_calcs(messages, answer):
        if str(c.get("usage_from", "")).startswith("average daily"):
            continue   # the tool took the usage from the history itself, per day: nothing to mismatch
        used, period = float(c.get("current_usage_kwh") or 0), c.get("period")
        if period != "day" and _matches({used}, daily):
            issues.append(f"current_usage_kwh {c.get('current_usage_kwh')} is a daily figure from query_energy_usage, "
                          f"but period was '{period}': call calculate_energy_savings again leaving current_usage_kwh out "
                          "(the tool then takes the device's daily use from the history, with period 'day')")
            continue
        span = PERIOD_DAYS.get(period, (1, 1) if period == "one_time" else None)   # one use is at most a day's use
        for total, days, label, avg in totals:
            if span and _matches({used}, {total}) and not span[0] <= days <= span[1]:
                issues.append(f"current_usage_kwh {c.get('current_usage_kwh')} is the {days}-day {label} from query_energy_usage, "
                              f"but period was '{period}', so the saving and its annual figure are for the wrong span of time: call "
                              "calculate_energy_savings again leaving current_usage_kwh out, so the tool takes the device's "
                              f"daily use from the history (about {avg} kWh a day), with period 'day'")
                break
    return issues


TIME = r"(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?\s?[Mm]\b\.?|(\d{1,2}):(\d{2})"
TIME_RANGE = re.compile(rf"(?:{TIME})\s*(?:-|\u2013|\u2014|to|until|through|and)\s*(?:{TIME})")
TIME_ONE = re.compile(TIME)
RATE = re.compile(r"\$\s?(0\.\d{3,4})(?!\d)")


def _hour(groups) -> Optional[int]:
    """Hour of day (0-23) from one TIME match's five groups, or None."""
    h12, _, ampm, h24, _ = groups
    if h12 is not None:
        h = int(h12)
        return None if not 1 <= h <= 12 else (h % 12) + (12 if ampm.lower() == "p" else 0)
    h = int(h24)
    return h if h <= 24 else None


def _span(a: int, b: int) -> set:
    """Hours a range covers, read the way the tools write ranges: the end is where the last hour ENDS.

    16:00-21:00 is hours 16-20 (the peak); 22:00-06:00 is 22, 23 and 0-5 (off-peak, wrapping past midnight);
    00:00-01:00 is the single hour 0. Earlier versions of this check read the end as included, so "00:00 - 06:00: $0.1111 to
    $0.1591" passed (0.1591 is the 06:00 mid-peak rate) and "18:00-19:00: $0.2601" passed (that is 19:00's rate).
    """
    a, b = a % 24, b % 24
    if a == b:
        return {a}
    return set(range(a, b)) if a < b else set(range(a, 24)) | set(range(0, b))


def _ranges(line: str) -> list:
    """The hour ranges a line names, as (hours, matched text), plus single times as one-hour sets (text None)."""
    out, taken = [], []
    for m in TIME_RANGE.finditer(line):
        # "X and Y" is a range only after "between" ("between 4 PM and 8 PM"); "at 00:00 and 03:00" lists two hours
        if re.search(r"\band\b", m.group(0)) and not re.search(r"between\s*$", line[:m.start()], re.I):
            continue
        a, b = _hour(m.groups()[:5]), _hour(m.groups()[5:])
        if a is not None and b is not None:
            out.append((_span(a, b), m.group(0)))
            taken.append(m.span())
    for m in TIME_ONE.finditer(line):
        if not any(s <= m.start() < e for s, e in taken) and _hour(m.groups()) is not None:
            out.append(({_hour(m.groups()) % 24}, None))
    return out


def _clean_lines(answer: str) -> list:
    return (answer or "").replace("**", "").replace("__", "").splitlines()


def _window(hours) -> str:
    """Hours as clock ranges ending where the last hour ends, matching tools.hour_window."""
    hs = sorted({h % 24 for h in hours})
    if not hs:
        return ""
    runs, start, prev = [], hs[0], hs[0]
    for h in hs[1:]:
        if h != prev + 1:
            runs.append((start, prev)); start = h
        prev = h
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 23:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return ", ".join(f"{x:02d}:00-{(y + 1) % 24:02d}:00" for x, y in runs)


def rate_hour_errors(messages, answer: str) -> list:
    """Hourly prices the answer puts against the wrong hours, decided in code against get_electricity_prices.

    Read line by line: the hours a line names (single times, and ranges such as 16:00 - 21:00 or 10 PM to
    12 AM, whose end is where the last hour ends) and the per-kWh prices on the same line ($0.xxx or $0.xxxx).
      - A price is reported when it is exactly a rate from the table for some OTHER hour and is not the rate of
        any hour the line names: the signature of a rate copied from the wrong row. Rounded or averaged prices
        match no row and are left alone.
      - A price range ("$0.2582 to $0.2604") next to a single hour range must be that range's lowest and highest
        rate on one of the days retrieved: an earlier run gave "16:00 - 20:00: $0.2582 to $0.2604" when those hours
        run from $0.2555 to $0.2604.
      - One price next to a range of several hours must be that range's average, highest or lowest rate, or the
        price a calculation used for exactly those hours.
    """
    by_hour, by_rate, tables = {}, {}, []
    for d in _payloads(messages, "get_electricity_prices"):
        table = {}
        for r in d.get("hourly_rates", []):
            if isinstance(r, dict) and isinstance(r.get("rate"), (int, float)):
                by_hour.setdefault(r.get("hour"), set()).add(r["rate"])
                by_rate.setdefault(round(r["rate"], 4), set()).add(r.get("hour"))
                table[r.get("hour")] = r["rate"]
        if table:
            tables.append(table)
    calc_prices = []   # (hours, price) a calculation priced: a use-weighted price is not the plain average
    for c in _payloads(messages, "calculate_energy_savings"):
        if c.get("current_hours") and isinstance(c.get("price_per_kwh"), (int, float)):
            calc_prices.append((set(c["current_hours"]), c["price_per_kwh"]))
        if c.get("optimized_hours") and isinstance(c.get("optimized_price_per_kwh"), (int, float)):
            calc_prices.append((set(c["optimized_hours"]), c["optimized_price_per_kwh"]))
    issues = []
    for line in _clean_lines(answer):
        found = _ranges(line)
        hours = set().union(*(h for h, _ in found)) if found else set()
        if not hours:
            continue
        flagged = False
        for m in RATE.finditer(line):
            rate = round(float(m.group(1)), 4)
            rows = by_rate.get(rate)
            if not rows or any(abs(x - rate) < 0.00005 for h in hours for x in by_hour.get(h, ())):
                continue
            named = ", ".join(f"{h:02d}:00" for h in sorted(hours))
            actual = "; ".join(f"{h:02d}:00 ${'/'.join(str(x) for x in sorted(by_hour[h]))}" for h in sorted(hours) if h in by_hour)
            issues.append(f"${m.group(1)} is the {', '.join(f'{h:02d}:00' for h in sorted(rows))} rate, not a rate for {named} "
                          f"(get_electricity_prices gives {actual or 'no rate for those hours'}; a range ends where its last hour ends, "
                          "so 00:00-06:00 is hours 0-5): take each hour's rate from hourly_rates")
            flagged = True
        spans = [(h, t) for h, t in found if t is not None]
        pr = [(a or c, b or d) for a, b, c, d in PRICE_RANGE.findall(line)]   # "$x to $y" or "between $x and $y"
        one = RATE.findall(line)
        if not flagged and not pr and len(one) == 1 and len(spans) == 1 and len(spans[0][0]) >= 2:
            # one rate for several hours must be their average, highest or lowest (or a calculation's price for exactly
            # those hours): a live answer said peak rates "spike to $0.2582 (from 16:00-21:00)"; that is the 16:00 rate,
            # and the peak runs from $0.2519 to $0.2604
            span, text = spans[0]
            fits = [t for t in tables if span <= set(t)]
            allowed = {p for hs, p in calc_prices if hs == span}
            for t in fits:
                vals = [t[h] for h in span]
                allowed |= {min(vals), max(vals), sum(vals) / len(vals)}
            if fits and not any(_printed(v, one[0]) for v in allowed):
                vals = [fits[0][h] for h in span]
                issues.append(f"${one[0]} is given for '{text}', but those hours run from ${min(vals):.4f} to ${max(vals):.4f} "
                              f"(average ${sum(vals) / len(vals):.4f}) in get_electricity_prices: quote that range or its average, "
                              "or name the single hour the rate belongs to")
        if flagged or len(spans) != 1 or len(pr) != 1:
            continue
        lo, hi = sorted((round(float(pr[0][0]), 4), round(float(pr[0][1]), 4)))
        span, text = spans[0]
        fits = [t for t in tables if span <= set(t)]
        if fits and not any(abs(min(t[h] for h in span) - lo) < 0.00005 and abs(max(t[h] for h in span) - hi) < 0.00005 for t in fits):
            t = fits[0]
            issues.append(f"'{text}' is quoted at ${pr[0][0]} to ${pr[0][1]}, but those hours ({_window(span)}) run from "
                          f"${min(t[h] for h in span):.4f} to ${max(t[h] for h in span):.4f} in get_electricity_prices: quote the lowest and "
                          "highest rate of exactly the hours named")
    return issues


PRICE_RANGE = re.compile(r"\$\s?(0\.\d{3,4})\s*(?:-|\u2013|\u2014|to)\s*\$?\s?(0\.\d{3,4})(?!\d)"
                         r"|between\s+\$\s?(0\.\d{3,4})\s+and\s+\$\s?(0\.\d{3,4})(?!\d)", re.I)
MONEY = re.compile(r"\$\s?-?\s?(\d[\d,]*(?:\.\d+)?)")
KWH = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*kWh\b", re.I)
MONTHS = {m: i for i, ms in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1) for m in [ms]}
DATE_TOKEN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b"
                        r"|\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
PEAK_WORD = re.compile(r"(?<![-\w])(?<!off )(?<!mid )peak\b", re.I)


def _printed(value: float, text: str) -> bool:
    """True when `text` (a number as printed in the answer) is `value` rounded to the decimals printed."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(abs(value) - float(text.replace(",", ""))) <= 0.5 * 10 ** -decimals + 1e-9


def _tool_values(messages) -> set:
    """Every data figure any tool returned (magnitudes), plus numbers written in retrieved tips."""
    out = set()
    for m in messages:
        if not isinstance(m, ToolMessage) or m.status == "error":
            continue
        try:
            data = json.loads(m.content) if isinstance(m.content, str) else m.content
        except (TypeError, ValueError):
            continue
        out |= {abs(v) for v in data_figures(data)}
        if m.name == "search_energy_tips":
            out |= {float(x.replace(",", "")) for x in NUMBER.findall(json.dumps(data))}
    return out


def money_errors(messages, answer: str) -> list:
    """Dollar figures in the answer that no tool returned, i.e. money the model worked out itself.

    Costs and savings come from calculate_energy_savings and rates from get_electricity_prices; a figure that
    is a printed rounding of one of those is fine ($275 for 275.12). An earlier laundry answer multiplied
    rates itself and subtracted wrongly ($0.5202 - $0.2276 written as a $0.066 saving).
    """
    values = _tool_values(messages)
    if not any(isinstance(m, ToolMessage) for m in messages):
        return []
    bad = []
    for m in MONEY.finditer((answer or "").replace("**", "")):
        text = m.group(1).rstrip(".,")
        if not text or float(text.replace(",", "")) == 0:
            continue
        if not any(_printed(v, text) for v in values) and f"${text}" not in bad:
            bad.append(f"${text}")
    if not bad:
        return []
    return [f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not a figure any tool returned: never work out money yourself. "
            "Quote rates from get_electricity_prices and costs or savings from calculate_energy_savings (call it for any cost "
            "you state, e.g. one stated amount with period 'one_time' and the hours), or leave the figure out"]


def _year_hint(messages) -> int:
    for name in ("get_weather_forecast", "get_electricity_prices", "query_solar_generation", "query_energy_usage"):
        for d in _payloads(messages, name):
            for v in json.dumps(d)[:4000].split('"'):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                    return int(v[:4])
    return datetime.now().year


def _dates_on(line: str, year: int) -> list:
    out = []
    for m in DATE_TOKEN.finditer(line):
        try:
            if m.group(1):
                out.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d"))
            else:
                mon, day = (m.group(4), m.group(5)) if m.group(4) else (m.group(7), m.group(6))
                out.append(datetime(year, MONTHS[mon.lower()[:3]], int(day)).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return out


def _dated_kwh(messages) -> dict:
    """kWh figures that belong to one date: forecast days and hours, solar history days, usage days."""
    by_date = {}
    add = lambda day, v: by_date.setdefault(day, set()).add(abs(float(v))) if isinstance(v, (int, float)) and v else None
    for d in _payloads(messages, "get_weather_forecast"):
        for x in d.get("daily", []):
            add(x.get("date"), x.get("estimated_solar_kwh"))
        for x in d.get("hourly", []):
            add(x.get("date"), x.get("estimated_solar_kwh"))
    for d in _payloads(messages, "query_solar_generation"):
        for x in d.get("daily", []):
            add(x.get("date"), x.get("generation_kwh"))
    for d in _payloads(messages, "query_energy_usage"):
        for x in d.get("daily", []):
            add(x.get("date"), x.get("consumption_kwh"))
    return by_date


def date_figure_errors(messages, answer: str) -> list:
    """kWh figures stated for a date they do not belong to, or for a date no tool covered.

    The forecast starts today, so a 7-day forecast run on Thursday 19 Nov ends on Wednesday 25 Nov; an
    earlier answer added "Thursday, November 26: 11.41 kWh", which is Tuesday 24 Nov's figure. Only lines that
    name exactly one date are judged, and only figures some tool returned for another date are reported.
    """
    by_date = _dated_kwh(messages)
    if not by_date:
        return []
    year, issues = _year_hint(messages), []
    for line in _clean_lines(answer):
        days = set(_dates_on(line, year))
        if len(days) != 1:
            continue
        day = days.pop()
        for m in KWH.finditer(line):
            raw = m.group(1)
            if "." not in raw and float(raw.replace(",", "")) <= 31:
                continue
            if day in by_date and any(_printed(v, raw) for v in by_date[day]):
                continue
            others = sorted(k for k, vs in by_date.items() if k != day and any(_printed(v, raw) for v in vs))
            if not others:
                continue
            if day in by_date:
                issues.append(f"{raw} kWh is the figure for {', '.join(others)}, not for {day}: take each day's figure from its own entry")
            else:
                covered = sorted(by_date)
                issues.append(f"no tool returned data for {day} (the data retrieved covers {covered[0]} to {covered[-1]}; the forecast "
                              f"starts today), yet the answer gives it {raw} kWh, which is the figure for {', '.join(others)}: "
                              "list only the days a tool returned")
    return issues


def window_errors(messages, answer: str) -> list:
    """Hour ranges written one hour short, and ranges that are not the hours a quoted saving was calculated for.

      - the peak period written as 16:00-20:00 (its last hour, 20:00, left out: the period is 16:00-21:00);
      - a day's best solar hours written as 10:00-14:00 when the hours are 10-14 (10:00-15:00);
      - a quoted saving or cost calculated for hours 0-5 (00:00-06:00) while the answer names 00:00-05:00, or for
        18-21 while the answer names 18:00-21:00. A range that is a whole time-of-use period (22:00-06:00,
        16:00-21:00) describes the tariff and is left alone.
    """
    issues = []
    peaks, periods = [], []
    for d in _payloads(messages, "get_electricity_prices"):
        p = sorted(r.get("hour") for r in d.get("hourly_rates", []) if isinstance(r, dict) and r.get("period") == "peak")
        o = {r.get("hour") for r in d.get("hourly_rates", []) if isinstance(r, dict) and r.get("period") == "off_peak"}
        if len(p) >= 2 and p == list(range(p[0], p[-1] + 1)):
            peaks.append(p)
        periods += [set(p), o]
    best = {}
    for d in _payloads(messages, "get_weather_forecast"):
        for x in d.get("daily", []):
            b = sorted(x.get("best_solar_hours") or [])
            if len(b) >= 2 and b == list(range(b[0], b[-1] + 1)):
                best[x.get("date")] = b
    year = _year_hint(messages)
    found = _numbers(answer)
    seen = set()
    def note(msg):
        if msg not in seen:
            seen.add(msg); issues.append(msg)
    lines = _clean_lines(answer)
    for line in lines:
        spans = [(h, t) for h, t in _ranges(line) if t is not None]
        if not spans:
            continue
        if PEAK_WORD.search(line):
            for p in peaks:
                for h, t in spans:
                    if h == set(range(p[0], p[-1])):
                        note(f"'{t}' leaves out the {p[-1]:02d}:00 peak hour: the peak period is {_window(p)} (peak_window), "
                             "because a range ends where its last hour ends")
        days = set(_dates_on(line, year))
        cands = [best[x] for x in days if x in best] if days else (list(best.values()) if re.search(r"\bsolar\b", line, re.I) else [])
        for b in cands:
            for h, t in spans:
                if h == set(range(b[0], b[-1])):
                    note(f"'{t}' is one hour short of the best solar hours {b} ({_window(b)}, best_solar_window): a range ends where its last hour ends")
    quoted = []
    for c in _payloads(messages, "calculate_energy_savings"):
        figs = {abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd", "current_cost_usd", "optimized_cost_usd")
                if isinstance(c.get(k), (int, float)) and c.get(k)}
        if _matches(found, figs):
            quoted.append(c)
    # a range that is exactly the hours of another quoted calculation belongs to that one: with an EV move to
    # 00:00-06:00 and an appliance calculation over 22:00-06:00, the EV's 00:00-06:00 is not a short appliance window
    by_period = {}
    for d in _payloads(messages, "get_electricity_prices"):
        for r in d.get("hourly_rates", []):
            if isinstance(r, dict):
                by_period.setdefault(r.get("period"), set()).add(r.get("hour"))

    def tariff_line(line: str, h: set) -> bool:
        """'Mid-Peak Hours (7:00-16:00): Rates are as follows' (a live answer) lists the tariff, not the priced hours."""
        for word, per in ((r"mid-?\s?peak", "mid_peak"), (r"off-?\s?peak", "off_peak"), (PEAK_WORD.pattern, "peak")):
            if re.search(word, line, re.I) and h <= by_period.get(per, set()):
                return True
        return False
    for c in quoted:
        own = [set(o.get(k)) for o in quoted if o is not c for k in ("current_hours", "optimized_hours") if o.get(k)]
        figs = [abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd", "current_cost_usd", "optimized_cost_usd", "current_usage_kwh")
                if isinstance(c.get(k), (int, float)) and c.get(k)]
        for key in ("current_hours", "optimized_hours"):
            hs = set(c.get(key) or [])
            if len(hs) < 2:
                continue
            usage = c.get("current_usage_kwh")
            for line in lines:
                # a one-hour range counts when the line states the calculation's kWh beside it: a live answer gave
                # "usage during peak (16:00-17:00): 5.91 kWh" for 5.91 kWh calculated over 16:00-18:00
                beside = isinstance(usage, (int, float)) and usage and any(_printed(usage, k) for k in KWH.findall(line))
                for h, t in _ranges(line):
                    if (t is None or (len(h) < 2 and not beside) or h == hs or not (h & hs) or not (h < hs or h > hs) or h in periods
                            or h in own or (tariff_line(line, h) and not any(_printed(v, x.rstrip(".,")) for x in MONEY.findall(line) + KWH.findall(line)
                                                 for v in figs))):
                        continue
                    note(f"the quoted saving was calculated for {_window(hs)} ({key} {sorted(hs)}), but the answer names '{t}' "
                         f"({_window(h)}): name exactly the hours that were priced, or calculate the hours you recommend")
    return issues


def _sections(answer: str) -> list:
    """The answer split at headings and numbered items, each a list of clauses."""
    blocks, cur = [], []
    for line in _clean_lines(answer):
        if re.match(r"\s*(?:#{1,6}\s|\d+\.\s)", line) and cur:
            blocks.append(cur); cur = []
        # clauses: sentences, and the items of a list joined by semicolons ("2391.47 kWh; $285.89 a year" are two facts)
        lead = re.match(r"(\s*(?:#{1,6}\s*)?\d+\.\s+)(.*)", line)   # keep "3." with its heading, not as a sentence of its own
        prefix, rest = (lead.group(1), lead.group(2)) if lead else ("", line)
        parts = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(*\-])|;\s*", rest) if x.strip()]
        if prefix:
            parts = [prefix + (parts[0] if parts else "")] + parts[1:]
        cur += parts
    if cur:
        blocks.append(cur)
    return blocks


def calc_description_errors(messages, answer: str) -> list:
    """A quoted saving described as something other than what calculate_energy_savings calculated.

      - a cost or saving written next to a kWh figure that is not the usage it was calculated from (an earlier run put the
        $15.85 cost of 61.63 kWh of peak-hour charging next to the EV's 218.91 kWh daily total);
      - a cut across the whole day at the same price described as a peak-hours change or a move off-peak (earlier:
        a 10% appliance cut presented as "$5.49 at peak, $4.94 off-peak"; a whole-day 10% HVAC cut called "a 10%
        reduction ... during peak hours").
    """
    issues = []
    sections = _sections(answer)
    def context(i):
        """A section plus the one before it when that one has no figures of its own ('### Why' above '### Estimated Savings')."""
        prev = sections[i - 1] if i > 0 and not MONEY.search(" ".join(sections[i - 1])) else []
        return " ".join(prev + sections[i])
    all_calcs = _payloads(messages, "calculate_energy_savings")
    for c in all_calcs:
        money = [abs(c.get(k)) for k in ("current_cost_usd", "optimized_cost_usd", "savings_usd", "annual_savings_usd")
                 if isinstance(c.get(k), (int, float)) and c.get(k)]
        others = [abs(o.get(k)) for o in all_calcs if o is not c for k in ("current_cost_usd", "optimized_cost_usd", "savings_usd", "annual_savings_usd")
                  if isinstance(o.get(k), (int, float)) and o.get(k) and abs(o.get(k)) not in money]
        usage = [abs(c.get(k)) for k in ("current_usage_kwh", "optimized_usage_kwh", "savings_kwh") if isinstance(c.get(k), (int, float)) and c.get(k)]
        has = lambda sent: any(_printed(v, t.rstrip(".,")) for t in MONEY.findall(sent) for v in money)
        whole_day_cut = (not c.get("current_hours") and c.get("price_per_kwh") == c.get("optimized_price_per_kwh")
                         and (c.get("savings_kwh") or 0) > 0)
        for bi, block in enumerate(sections):
            if not any(has(x) for x in block):
                continue
            for sent in block:
                if has(sent):
                    kwh = [k for k in KWH.findall(sent) if "." in k or float(k.replace(",", "")) > 31]
                    # two calculations can round to the same costs (a live washing machine and dryer, $0.14 / $0.46 / $0.32):
                    # a sentence holding another quoted calculation's own kWh and money is that calculation's
                    twin = any(o is not c and any(_printed(abs(o.get(kk)), t.rstrip(".,")) for t in MONEY.findall(sent)
                                                  for kk in ("savings_usd", "current_cost_usd", "optimized_cost_usd", "annual_savings_usd")
                                                  if isinstance(o.get(kk), (int, float)) and o.get(kk))
                               and any(_printed(abs(o.get(kk)), k) for k in kwh for kk in ("current_usage_kwh", "optimized_usage_kwh")
                                       if isinstance(o.get(kk), (int, float)) and o.get(kk))
                               for o in all_calcs)
                    if kwh and not twin and not any(_printed(u, k) for k in kwh for u in usage):
                        issues.append(f"the answer puts {kwh[0]} kWh next to a figure calculate_energy_savings worked out for "
                                      f"{c.get('current_usage_kwh')} kWh ({c.get('usage_from')}): state the kWh the calculation used, "
                                      f"as its summary does ('{c.get('summary', '')[:120]}')")
                same_price_cut = (c.get("price_per_kwh") == c.get("optimized_price_per_kwh") and (c.get("savings_kwh") or 0) > 0
                                  and not c.get("optimized_hours"))
                moving = r"off-?peak|\bshift\w*|\bmov(?:e|ing)\b"
                # the move is claimed in the sentence with the figure, or in a figure-free section just above a section
                # that quotes only this calculation ('### Why' then '### Estimated Savings'); a summary listing several
                # suggestions ("EV Charging Shift: $3249.68" beside the HVAC figure) is not that
                own_only = not any(_printed(v, t.rstrip(".,")) for t in MONEY.findall(" ".join(block)) for v in others)
                if same_price_cut and not whole_day_cut and has(sent) and (
                        re.search(moving, sent, re.I) or (own_only and re.search(moving, context(bi), re.I))):
                    issues.append(f"the {c.get('reduction_percent') or ''}% saving is a cut at the same price ('{str(c.get('summary', ''))[:110]}'), "
                                  "but the answer presents it as moving use off-peak: value the move with current_hours and "
                                  "optimized_hours (same kWh, cheaper hours) and quote that")
                    break
                if whole_day_cut and PEAK_WORD.search(sent) or whole_day_cut and re.search(r"off-?peak", sent, re.I):
                    if has(sent) or re.search(r"\d\s?%|\breduc", sent, re.I):
                        pct = f"{c.get('reduction_percent'):g}% " if isinstance(c.get("reduction_percent"), (int, float)) else ""
                        issues.append(f"the {pct}saving was calculated as a cut across the whole day at the "
                                      f"same price ('{c.get('summary', '')[:110]}'), but the answer describes it in terms of peak or off-peak "
                                      "hours: for a cut in certain hours call calculate_energy_savings with current_hours and reduction_percent; "
                                      "for moving use to cheaper hours, with current_hours and optimized_hours")
                        break
    return list(dict.fromkeys(issues))


def tonight_errors(messages, answer: str, today: Optional[str] = None) -> list:
    """'Tonight' in an answer built only on another day's prices (an earlier answer quoted tomorrow's 22:00 rate as tonight's)."""
    if not re.search(r"\btonight\b", answer or "", re.I):
        return []
    if today is None:
        firsts = [d.get("daily", [{}])[0].get("date") for d in _payloads(messages, "get_weather_forecast") if d.get("daily")]
        today = firsts[0] if firsts else None
    dates = {d.get("date") for d in _payloads(messages, "get_electricity_prices")}
    if not today or not dates or today in dates:
        return []
    return [f"the answer says 'tonight', but the prices retrieved are for {', '.join(sorted(dates))}, not today ({today}): "
            f"call get_electricity_prices for {today} to quote tonight's hours, or name the date of the night you mean"]


STATED_KWH = re.compile(r"(\d+(?:\.\d+)?)\s*kWh\b", re.I)


def stated_amount_errors(question: str, messages, answer: str) -> list:
    """A question that names an amount of energy ("add 40 kWh") priced for a different amount.

    A live answer priced 48.96 kWh (the EV's usual overnight use, read from the history because no kWh was given)
    for a question about adding 40 kWh. Every successful savings calculation the answer quotes must use the stated kWh.
    """
    stated = {float(x) for x in STATED_KWH.findall(question or "")}
    if not stated:
        return []
    found = _numbers(answer)
    for c in _payloads(messages, "calculate_energy_savings"):
        figs = {abs(c.get(k)) for k in ("current_cost_usd", "optimized_cost_usd", "savings_usd", "annual_savings_usd")
                if isinstance(c.get(k), (int, float)) and c.get(k)}
        hs, ho = c.get("current_hours"), c.get("optimized_hours")
        if _matches(found, figs) and hs and ho and sorted(hs) != sorted(ho) and abs(c.get("savings_usd") or 0) > 0:
            return [f"the question asks what {sorted(stated)[0]:g} kWh will cost, but the answer reports a saving from moving it from "
                    f"{_window(hs)} to {_window(ho)}, a 'before' schedule the customer never described: price the amount once with "
                    "current_usage_kwh, period 'one_time', date and current_hours set to the hours you recommend, and quote current_cost_usd"]
        if _matches(found, figs) and not any(abs((c.get("current_usage_kwh") or 0) - n) < 0.006 for n in stated):
            n = sorted(stated)[0]
            return [f"the question states {n:g} kWh, but the cost you quote is for {c.get('current_usage_kwh')} kWh ({c.get('usage_from')}): "
                    f"price the stated amount with current_usage_kwh={n:g}, period 'one_time', date, and current_hours and "
                    "optimized_hours both set to the hours you recommend"]
    return []


def _calls(messages) -> dict:
    """tool_call_id -> (tool name, arguments) for every tool call the model made."""
    out = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                out[tc.get("id")] = (tc.get("name"), tc.get("args") or {})
    return out


def unretried_calc_errors(messages) -> list:
    """A savings calculation that returned an error and was never made again for that device.

    A live answer said it "couldn't calculate specific savings" for the EV because its call left out the date;
    the tool said what was missing, and nothing sent the model back to fix the call.
    """
    calls, issues, state = _calls(messages), [], {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "calculate_energy_savings":
            name, args = calls.get(m.tool_call_id, (None, {}))
            device = str(args.get("device_type", "")).strip().lower()
            try:
                err = (json.loads(m.content) if isinstance(m.content, str) else m.content).get("error")
            except (TypeError, ValueError, AttributeError):
                err = None
            state[device] = err if (m.status == "error" or err) else None
    for device, err in state.items():
        if err:
            issues.append(f"calculate_energy_savings for '{device or 'a device'}' returned an error and was not made again: {str(err)[:200]}. "
                          "Fix the call as the error says and use its result; do not tell the customer it could not be calculated")
    return issues


def no_change_errors(question: str, messages, answer: str) -> list:
    """A calculation that compares a schedule with itself (the same hours before and after) used as a finding.

    Pricing one stated amount is the one legitimate use (the question names its kWh). A live answer moved appliance
    use 'from 22:00-06:00 to 22:00-06:00' and told the customer that moving appliances off-peak saves nothing.
    """
    if STATED_KWH.search(question or ""):
        return []
    found = _numbers(answer)
    says_none = re.search(r"\bno (?:real )?sav|\bdid not (?:yield|save|reduce)|\bremain(?:ed|s)? (?:at |the same)|\bno (?:difference|change) in cost"
                          r"|\bcost remains the same", answer or "", re.I)
    for c in _payloads(messages, "calculate_energy_savings"):
        hs, ho = c.get("current_hours"), c.get("optimized_hours")
        costs = {abs(c.get(k)) for k in ("current_cost_usd", "optimized_cost_usd") if isinstance(c.get(k), (int, float)) and c.get(k)}
        # judged only when the answer uses it: a live run made such a call, then valued the real shift and quoted only that
        used = _matches(found, costs) or says_none
        # a calculation that only priced use in one set of hours compares nothing: "no savings ... the cost remains the
        # same" from it is a false finding (a live answer said moving appliances off-peak saves nothing)
        priced_only = hs and not ho and c.get("reduction_percent") is None and not c.get("savings_usd")
        if priced_only and says_none and _matches(found, costs):
            return [f"the answer says there is no saving, but its calculation only priced {c.get('current_usage_kwh')} kWh in "
                    f"{_window(hs)}; it compares nothing. To value moving that use, call calculate_energy_savings with "
                    "current_hours (when it runs now) and optimized_hours (the cheaper hours), and quote that result"]
        if used and hs and ho and sorted(hs) == sorted(ho) and not c.get("savings_usd") and not c.get("savings_kwh"):
            return [f"a calculation moved use from {_window(hs)} to the same hours, so it compares a schedule with itself and finds no saving: "
                    "set current_hours to when the use happens now (for example the peak 16:00-21:00) and optimized_hours to the "
                    "cheaper hours, and quote that result"]
    return []


def calc_kwh_errors(messages, answer: str) -> list:
    """A quoted saving whose kWh, read from the history, is never stated, so the reader cannot tell what it is for.

    A live dishwasher answer gave "Current usage: 10.15 kWh" (the whole day) beside "$0.45 at peak" (the cost of the
    1.75 kWh used in 16:00-21:00) and never stated the 1.75 kWh.
    """
    found, issues = _numbers(answer), []
    for c in _payloads(messages, "calculate_energy_savings"):
        figs = {abs(c.get(k)) for k in ("current_cost_usd", "optimized_cost_usd", "savings_usd", "annual_savings_usd")
                if isinstance(c.get(k), (int, float)) and c.get(k)}
        kwh = c.get("current_usage_kwh")
        if (_matches(found, figs) and isinstance(kwh, (int, float)) and kwh and str(c.get("usage_from", "")).startswith("average daily")
                and not _quotes(found, {kwh})):
            issues.append(f"the answer quotes a saving calculated for {kwh} kWh ({c.get('usage_from')}) but never states those {kwh} kWh: "
                          f"say which energy the figures are for, as the summary does ('{str(c.get('summary', ''))[:110]}')")
    return issues


CHEAPEST = re.compile(r"\b(?:cheapest|lowest[- ]cost|least expensive|most favou?rable)\s+(?:electricity\s+|grid\s+)?"
                      r"(?:rates?|prices?|tariffs?|hours|times?|periods?)\b"
                      r"|\bbest\s+(?:electricity\s+|grid\s+)?(?:rates?|prices?|tariffs?)\b", re.I)   # "best hours" means solar


def cheapest_claim_errors(messages, answer: str) -> list:
    """Hours called the cheapest when they are not: a live answer said 10:00-15:00 had 'the most favorable electricity
    rates' (mid-peak, $0.1586-$0.1628) when off-peak that day was $0.1110-$0.1138."""
    tables = []
    for d in _payloads(messages, "get_electricity_prices"):
        rows = {r.get("hour"): (r.get("rate"), r.get("period")) for r in d.get("hourly_rates", []) if isinstance(r, dict)}
        if rows:
            tables.append(rows)
    if not tables:
        return []
    issues = []
    for block in _sections(answer):
        text = " ".join(block)
        if not CHEAPEST.search(text):
            continue
        # the hours the claim is about: ranges in the claiming sentence; the section's ranges only if it names none
        # ("the peak rates (16:00-21:00) are higher" in the same section is not called cheap)
        # "This is not the cheapest grid rate but utilizes free solar energy" (a live answer) claims nothing, and a claim
        # whose own sentence names no hours ("making it the cheapest time to charge", about off-peak) is not tied to
        # whatever hours the section names elsewhere (it was read as calling the solar window cheapest)
        claim = [x for x in block if CHEAPEST.search(x) and not re.search(
            r"\b(?:not|isn't|is not|aren't|never)\s+(?:the\s+|as\s+)?(?:cheapest|lowest|least expensive|most favou?rable)", x, re.I)]
        spans = [(h, t) for x in claim for h, t in _ranges(x) if t is not None and len(h) >= 2]
        # ... except a claim that points back ("I recommend charging between 10:00 AM and 3:00 PM. This timeframe aligns
        # with ... the most favorable electricity rates", a live answer): it takes the hours of the sentence just before it
        for i, x in enumerate(block):
            if x in claim and not _ranges(x) and i > 0 and re.match(r"^\W*(?:this|these|that|those|it|they)\b", x, re.I):
                spans += [(h, t) for h, t in _ranges(block[i - 1]) if t is not None and len(h) >= 2]
        for h, t in spans:
            for rows in tables:
                cheap = {x for x, (_, per) in rows.items() if per == "off_peak"}
                if cheap and h <= set(rows) and not h <= cheap:
                    lo = min(rows[x][0] for x in cheap); hi = max(rows[x][0] for x in cheap)
                    issues.append(f"the answer calls '{t}' the cheapest or most favourable rates, but those hours include "
                                  f"{', '.join(sorted({rows[x][1].replace('_', '-') for x in h}))} hours; the cheapest are off-peak "
                                  f"{_window(cheap)} at ${lo:.4f}-${hi:.4f}. If the hours are chosen for solar, say so and keep the rates apart")
                    break
    return list(dict.fromkeys(issues))


CONDITIONS = (("partly cloudy", "partly_cloudy"), ("partly sunny", "partly_cloudy"), ("sunny", "sunny"), ("clear", "sunny"),
              ("cloudy", "cloudy"), ("overcast", "cloudy"), ("rainy", "rainy"), ("rain", "rainy"))
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _condition_in(text: str):
    low = text.lower()
    for word, cond in CONDITIONS:
        if re.search(rf"\b{word}\b", low):
            return word, cond
    return None


def _conditions_named(text: str) -> set:
    """Every distinct condition a text names, longest phrase first ('partly cloudy' is not also 'cloudy')."""
    low, found = (text or "").lower(), set()
    for word, cond in CONDITIONS:
        if re.search(rf"\b{word}\b", low):
            found.add(cond)
            low = re.sub(rf"\b{word}\b", " ", low)
    return found


def condition_errors(messages, answer: str) -> list:
    """Weather conditions stated for days whose forecast says otherwise.

      - one condition claimed for the whole forecast ("The forecast indicates sunny weather" over six days of which
        Sunday is partly cloudy and Tuesday cloudy: a live answer, after the prompt asked for each day's condition);
      - a window tied to a condition ("10:00-15:00 on sunny days, 11:00-14:00 on partly cloudy days" when sunny
        Saturday's best hours are 11:00-14:00: a live answer).

    A sentence that names no day takes the days its own line names ("- Sunday, 2026-11-22: Run it 11:00-14:00. The
    forecast is partly cloudy ..." is about Sunday: a live answer, sent back twice as a claim about the whole week),
    and a sentence naming two or more conditions ("the forecasted sunny and partly cloudy days") claims no single one.
    """
    days = {}
    for d in _payloads(messages, "get_weather_forecast"):
        for x in d.get("daily", []):
            days[x.get("date")] = x
    if len(days) < 2:
        return []
    today = sorted(days)[0]
    tomorrow = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    by_weekday = {str(x.get("day_of_week", "")).lower(): k for k, x in days.items() if x.get("day_of_week")}
    year = _year_hint(messages)
    named = set(_dates_on(answer or "", year)) | {by_weekday[w] for w in WEEKDAYS if re.search(rf"\b{w}\b", (answer or "").lower()) and w in by_weekday}
    named &= set(days)
    def days_in(text):
        own = set(_dates_on(text, year)) | {by_weekday[w] for w in WEEKDAYS if re.search(rf"\b{w}\b", text.lower()) and w in by_weekday}
        if re.search(r"\btomorrow\b", text, re.I):
            own.add(tomorrow)
        if re.search(r"\btoday\b", text, re.I):
            own.add(today)
        return own & set(days)
    issues = []
    lines = []
    for line in _clean_lines(answer):
        rest = re.sub(r"^\s*(?:#{1,6}\s*)?(?:\d+\.|[-*\u2022])\s+", "", line)
        parts = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(*\-])|;\s*", rest) if x.strip()]
        lines.append((days_in(line), parts))
    for line_days, sents in lines:
        for sent in sents:
            found = _condition_in(sent)
            if not found:
                continue
            word, cond = found
            tied = False
            for w, cnd in CONDITIONS:   # every "<range> on <condition> days" in the sentence, not just the first
                for m in re.finditer(rf"(\d{{1,2}}:\d{{2}}\s*(?:-|\u2013|to)\s*\d{{1,2}}:\d{{2}})\s+on\s+{w}\s+days", sent, re.I):
                    tied = True
                    h = [x for x, t in _ranges(m.group(1)) if t is not None]
                    wrong = [k for k, x in sorted(days.items()) if k != today and x.get("condition") == cnd and h
                             and set(x.get("best_solar_hours") or []) != h[0]]
                    if wrong:
                        x = days[wrong[0]]
                        issues.append(f"'{m.group(0)}' does not hold: {x.get('day_of_week')} {wrong[0]} is {cnd.replace('_', ' ')} with best "
                                      f"hours {x.get('best_solar_window')}: give each day its own window from best_solar_window")
            if tied:
                continue
            if not re.search(r"\bforecast|\bweather\b|\bthis week\b", sent, re.I):
                continue
            # "a mix of sunny, partly cloudy, and cloudy days", "the forecasted sunny and partly cloudy days" (live answers)
            # name several conditions, not one
            if len(_conditions_named(sent)) >= 2:
                continue
            own = days_in(sent) or line_days
            covered = own or (named if len(named) >= 2 else set())
            off = sorted(k for k in covered if days[k].get("condition") != cond)
            if off:
                listed = "; ".join(f"{days[k].get('day_of_week')} {k} {str(days[k].get('condition')).replace('_', ' ')}" for k in off)
                issues.append(f"the answer calls the forecast {word}, but {listed}: give each day the condition its forecast entry states")
    return list(dict.fromkeys(issues))


DEVICE_WORDS = re.compile(r"\b(?:EV|electric (?:car|vehicle)|charg\w+|HVAC|thermostat|air condition\w*|dishwasher|washing machine|dryer|laundry|appliances?)\b", re.I)
MOVE_WORDS = re.compile(r"\b(?:off-?peak|peak hours|shift\w*|schedul\w+|move|moving)\b", re.I)


def _device_group(word: str) -> str:
    w = word.lower()
    if re.search(r"\bev\b|electric|charg|car\b|vehicle", w):
        return "ev"
    if re.search(r"hvac|thermostat|air condition|heat|cool", w):
        return "hvac"
    return "appliance"


def unquantified_errors(question: str, messages, answer: str) -> list:
    """In an answer asked for savings, a suggestion to move a device's use with no calculated saving behind it.

    A live 'three ways' answer valued two suggestions with calculate_energy_savings and left the third (appliances
    off-peak) with "can lead to additional savings", offering to calculate it if asked. Applies only when the
    question asks for savings or for suggestions from the usage history.
    """
    asked = {name for name, pattern, _, _ in DATA_RULES if pattern.search(question or "")}
    if not asked & {"quantified savings", "usage history"}:
        return []
    calcs = _payloads(messages, "calculate_energy_savings")
    money = {abs(c.get(k)) for c in calcs for k in ("savings_usd", "annual_savings_usd", "current_cost_usd", "optimized_cost_usd")
             if isinstance(c.get(k), (int, float)) and c.get(k)}
    valued = set()   # device groups whose saving the answer quotes
    found = _numbers(answer)
    for c in calcs:
        figs = {abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd") if isinstance(c.get(k), (int, float)) and c.get(k)}
        named = " ".join(str(c.get(k) or "") for k in ("device_type", "usage_from", "summary")).strip()
        if _matches(found, figs) and named:   # a calculation that names no device marks nothing as valued
            valued.add(_device_group(named))
    issues, heading = [], ""
    for block in _sections(answer):
        head = block[0] if block else ""
        if re.match(r"\s*#{1,6}\s", head) and not re.match(r"\s*#{1,6}\s*\d+\.", head):
            heading = head
        text = " ".join(block)
        if not (re.match(r"\s*(?:#{1,6}\s*)?\d+\.", head) and DEVICE_WORDS.search(text) and MOVE_WORDS.search(text)):
            continue
        # a numbered tip under a "Tips" heading, or an item about a device whose saving is already quoted elsewhere
        # (a live answer valued the appliance shift in suggestion 3 and repeated it as an "Additional Tips" item)
        groups = {_device_group(w) for w in DEVICE_WORDS.findall(text)}
        if re.search(r"\btips?\b", heading, re.I) or (groups and groups <= valued):
            continue
        if not any(_printed(v, t.rstrip(".,")) for t in MONEY.findall(text) for v in money):
            issues.append(f"the suggestion '{head.strip('# ').strip()[:70]}' gives no saving: value it with calculate_energy_savings "
                          "(for appliances: device_type 'appliance', date, current_hours [16, 17, 18, 19, 20], optimized_hours "
                          "[0, 1, 2, 3, 4, 5]) and quote the result; never offer to calculate later what the tool can calculate now")
    return issues


LOW_WORDS = re.compile(r"\b(?:lowest|cheapest|minimum|bottoms? out)\b", re.I)
HIGH_WORDS = re.compile(r"\b(?:highest|most expensive|maximum|peak(?:s|ing)? at|up to|as high as|top(?:s|ping)? out)\b", re.I)


def superlative_rate_errors(messages, answer: str) -> list:
    """'Lowest', 'highest' and 'up to' claims about rates, checked against the price tables.

      - a rate called the lowest must be the lowest rate of some period (off-peak, mid-peak, peak or the whole day) on a
        day retrieved; a live answer said off-peak rates "are lowest at $0.1112 and $0.1138" (the lowest is $0.1110);
      - a rate given as the highest or 'up to' must be the highest of some period;
      - an hour called the most expensive must be the hour with the highest rate (a live answer called Wednesday 16:00
        "when the rates are highest": 16:00 is that peak's cheapest hour, $0.2445; the highest is 19:00, $0.2473).
    """
    tables = []
    for d in _payloads(messages, "get_electricity_prices"):
        rows = {r.get("hour"): (r.get("rate"), r.get("period")) for r in d.get("hourly_rates", []) if isinstance(r, dict)}
        if rows:
            tables.append(rows)
    if not tables:
        return []
    lows, highs = set(), set()
    for rows in tables:
        for per in ("off_peak", "mid_peak", "peak", None):
            vals = [r for r, p in rows.values() if per is None or p == per]
            if vals:
                lows.add(round(min(vals), 4)); highs.add(round(max(vals), 4))
    issues = []
    for line in _clean_lines(answer):
        for sent in re.split(r"(?<=[.!?])\s+|;\s*", line):   # clauses: a superlative applies within its own clause
            rates = [round(float(x), 4) for x in RATE.findall(sent)]
            low, high = LOW_WORDS.search(sent), HIGH_WORDS.search(sent)
            if rates and low and not high:
                after = [round(float(x), 4) for x in RATE.findall(sent[low.start():])]
                if after and not any(abs(after[0] - v) < 0.00005 for v in lows):
                    issues.append(f"${after[0]:.4f} is called the lowest, but it is not the lowest rate of any period retrieved "
                                  f"(lowest by period: {', '.join(f'${v:.4f}' for v in sorted(lows))}): quote the minimum from the table")
            if rates and high and not low:
                after = [round(float(x), 4) for x in RATE.findall(sent[high.start():])]
                if after and not any(abs(after[0] - v) < 0.00005 for v in highs):
                    issues.append(f"${after[0]:.4f} is given as the highest ('{high.group(0)}'), but it is not the highest rate of any period "
                                  f"retrieved (highest by period: {', '.join(f'${v:.4f}' for v in sorted(highs))})")
            if not rates and re.search(r"\b(?:rates?|prices?)\b", sent, re.I) and (low or high) and not (low and high):
                # a deadline is not the hour the claim is about: "the lowest available rates before 7 AM" (a live answer)
                timed = re.sub(rf"\b(?:before|by|until|till|after|from)\s+(?:{TIME})", " ", sent, flags=re.I)
                singles = [h for h, t in _ranges(timed) if t is None]
                spans = [h for h, t in _ranges(timed) if t is not None]
                if len(singles) == 1 and not spans:
                    hour = next(iter(singles[0]))
                    best = [max(rows, key=lambda x: rows[x][0]) if high else min(rows, key=lambda x: rows[x][0]) for rows in tables]
                    if hour not in best:
                        word = "highest" if high else "lowest"
                        issues.append(f"{hour:02d}:00 is called the hour when rates are {word}, but the {word} rate is at "
                                      f"{', '.join(f'{b:02d}:00' for b in best)} in the table retrieved: name that hour, or drop the claim")
    return list(dict.fromkeys(issues))


COVER = re.compile(r"\b(?:more than )?enough(?: \w+){0,2} to (?:cover|power|meet|run)\b|\bcover(?:s|ing)? (?:all|most|the bulk) of\b|\bfully cover", re.I)
SHARE_COVER = re.compile(r"\bcover(?:s|ing)? (?:a )?(?:significant|large|good|substantial|big|major|considerable) (?:portion|part|share|amount|fraction)\b", re.I)


def coverage_errors(messages, answer: str) -> list:
    """Solar said to cover the home's needs when the figures say it cannot.

    A live answer called the 35.64 kWh forecast "more than enough to cover your typical daily energy needs" two lines
    above the home's 297.93 kWh a day. A claim about a named smaller load ("enough to cover the dishwasher's 1.75 kWh")
    is judged against that load; a claim about 'your needs' against the household's daily use from query_energy_usage.
    """
    solar = set()
    for d in _payloads(messages, "get_weather_forecast"):
        solar |= {float(x.get("estimated_solar_kwh")) for x in d.get("daily", []) if isinstance(x.get("estimated_solar_kwh"), (int, float))}
        solar |= {float(x.get("solar_kwh_in_best_window")) for x in d.get("daily", []) if isinstance(x.get("solar_kwh_in_best_window"), (int, float))}
    for d in _payloads(messages, "query_solar_generation"):
        solar |= {v for v in data_figures(d)}
    homes = []
    for d in _payloads(messages, "query_energy_usage"):
        if d.get("device_type") or d.get("device_name"):
            continue
        if isinstance(d.get("average_daily_kwh"), (int, float)):
            homes.append(d["average_daily_kwh"])
        else:
            homes.append(sum(v.get("average_daily_kwh", 0) for v in (d.get("device_breakdown") or {}).values()))
    issues = []
    daily = [x.get("estimated_solar_kwh") for d in _payloads(messages, "get_weather_forecast") for x in d.get("daily", [])]
    daily += [x.get("generation_kwh") for d in _payloads(messages, "query_solar_generation") for x in d.get("daily", [])]
    daily = [v for v in daily if isinstance(v, (int, float))]
    best = max(daily) if daily else None   # one day's solar, not the week's total
    for line in _clean_lines(answer):
        for sent in re.split(r"(?<=[.!?])\s+|;\s*", line):
            # a share of a named smaller load ("Your average daily appliance use is 32.78 kWh, so this window can cover a
            # significant portion": a live answer, about the appliances, not the home) is judged against that load
            named_load = [k for k in (float(x.replace(",", "")) for x in KWH.findall(sent))
                          if not any(_printed(v, f"{k}") for v in solar) and homes and k < 0.9 * max(homes)]
            if SHARE_COVER.search(sent) and best and named_load:
                if best < 0.25 * max(named_load):
                    issues.append(f"the solar is said to cover a large share of {max(named_load)} kWh, but the best day's {best} kWh is "
                                  f"{best / max(named_load):.0%} of it: state the share")
                continue
            if SHARE_COVER.search(sent) and best and homes and best < 0.25 * max(homes):
                # a live answer: "on sunny days, your panels can produce enough energy to cover a significant portion of
                # your daily usage" when the best day's 38.47 kWh is 13% of the home's 297.94 kWh
                issues.append(f"the solar is said to cover a large share of the home's use, but the best day's {best} kWh is "
                              f"{best / max(homes):.0%} of {max(homes)} kWh a day: state the share")
                continue
            if not COVER.search(sent):
                continue
            kwh = [float(k.replace(",", "")) for k in KWH.findall(sent)]
            sun = [k for k in kwh if any(_printed(v, f"{k}") for v in solar)]
            loads = [k for k in kwh if k not in sun]
            need = max(loads) if loads else (max(homes) if homes else None)
            if sun and need and max(sun) < need:
                share = round(max(sun) / need * 100)
                issues.append(f"{max(sun)} kWh of solar is said to cover the need, but the need is {need} kWh a day "
                              f"(about {share}% covered): say how much of the use the solar covers")
    return list(dict.fromkeys(issues))


PAYBACK_Q = re.compile(r"\bpay (?:for )?itself\b|\bpay ?back\b|\bROI\b|\breturn on (?:the )?investment\b|\bworth (?:it|buying)\b", re.I)


def payback_errors(question: str, messages, answer: str) -> list:
    """A payback question that states the upfront cost, answered without a payback calculated from it.

    A live answer to "A smart thermostat costs $250 ... how long would it take to pay for itself?" said "Since I
    cannot provide the upfront cost, you can calculate the payback period by dividing..." after dropping
    upfront_cost_usd from its call. The payback must come from calculate_energy_savings with that cost.
    """
    if not PAYBACK_Q.search(question or ""):
        return []
    costs = [float(x.replace(",", "")) for x in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", question or "")]
    if not costs:
        return []
    calcs = _payloads(messages, "calculate_energy_savings")
    with_cost = [c for c in calcs if isinstance(c.get("upfront_cost_usd"), (int, float))
                 and any(abs(c["upfront_cost_usd"] - k) < 0.005 for k in costs)]
    said = [t for t in NUMBER.findall(answer or "") if "." in t]   # a printed rounding counts ('about 0.9 years' for 0.87)
    if with_cost and any(_printed(c["payback_years"], t) for c in with_cost if isinstance(c.get("payback_years"), (int, float)) for t in said):
        return []
    return [f"the question gives an upfront cost of ${costs[0]:g} and asks how long it takes to pay for itself, but the answer "
            f"gives no payback from it: call calculate_energy_savings with upfront_cost_usd={costs[0]:g} (and date) and quote "
            "payback_years, payback_months and first_year_roi_percent as returned; never ask the customer to work it out"]


def _today(messages, today: Optional[str]) -> Optional[str]:
    """Today's date: as given, else the forecast's first day (the forecast starts today)."""
    if today:
        return today
    firsts = [d.get("daily", [{}])[0].get("date") for d in _payloads(messages, "get_weather_forecast") if d.get("daily")]
    return firsts[0] if firsts else None


def tonight_range_errors(messages, answer: str, today: Optional[str] = None) -> list:
    """A price range given for tonight's hours that is not tonight's.

    Tonight runs past midnight: 22:00-06:00 tonight is today's 22:00 and 23:00 and tomorrow's 00:00-05:00. A live
    answer listed "Tonight (November 19): Off-Peak Rates (22:00-06:00): $0.1029 to $0.1067"; $0.1067 is 03:00 on the
    19th, already past, and tonight's hours run from $0.1029 to $0.1138. A line is about tonight when it says so, or
    when it sits under a 'Tonight ...:' label. Judged only when both days' prices were retrieved.
    """
    if not re.search(r"\btonight\b", answer or "", re.I):
        return []
    today = _today(messages, today)
    if not today:
        return []
    tomorrow = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    tables = {}
    for d in _payloads(messages, "get_electricity_prices"):
        rows = {r.get("hour"): r.get("rate") for r in d.get("hourly_rates", []) if isinstance(r, dict)}
        if rows and d.get("date"):
            tables[d["date"]] = rows
    issues, label = [], None   # label: indentation of a 'Tonight ...:' line that governs the deeper lines below it
    for line in _clean_lines(answer):
        indent = len(line) - len(line.lstrip())
        if label is not None and line.strip() and indent <= label:
            label = None
        here = re.search(r"\btonight\b", line, re.I)
        if here and line.rstrip().endswith(":"):
            label = indent
            continue
        if not (here or label is not None):
            continue
        spans = [(h, t) for h, t in _ranges(line) if t is not None and len(h) >= 2]
        pr = [(a or c, b or d) for a, b, c, d in PRICE_RANGE.findall(line)]
        if len(spans) != 1 or len(pr) != 1:
            continue
        hours, text = spans[0]
        night = [(today if h >= 12 else tomorrow, h) for h in hours]
        if any(day not in tables or h not in tables[day] for day, h in night):
            continue
        vals = [tables[day][h] for day, h in night]
        lo, hi = sorted((float(pr[0][0]), float(pr[0][1])))
        if abs(min(vals) - lo) > 0.00005 or abs(max(vals) - hi) > 0.00005:
            issues.append(f"'{text}' tonight is quoted at ${pr[0][0]} to ${pr[0][1]}, but tonight's hours run from ${min(vals):.4f} to "
                          f"${max(vals):.4f}: the hours before midnight are {today}'s rates and the hours after midnight are "
                          f"{tomorrow}'s (get_electricity_prices for each date)")
    return list(dict.fromkeys(issues))


def tonight_rate_errors(messages, answer: str, today: Optional[str] = None) -> list:
    """A rate quoted for tonight that is this morning's.

    Tonight runs past midnight, so its hours after midnight are tomorrow's. A live answer said "use the off-peak hours
    from 22:00 tonight to 06:00 tomorrow morning ... the minimum rate being $0.1110": $0.1110 is today's 03:00 rate,
    already past, and only today's prices had been retrieved. Every per-kWh rate the answer quotes that is one of
    today's 00:00-06:00 rates and none of tonight's (today's 22:00 and 23:00, tomorrow's 00:00-05:00 when retrieved) is
    reported, with the call that gets tomorrow's rates when they are missing.
    """
    if not re.search(r"\btonight\b", answer or "", re.I):
        return []
    today = _today(messages, today)
    if not today:
        return []
    tomorrow = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    tables = {}
    for d in _payloads(messages, "get_electricity_prices"):
        rows = {r.get("hour"): r.get("rate") for r in d.get("hourly_rates", []) if isinstance(r, dict)}
        if rows and d.get("date"):
            tables[d["date"]] = rows
    if today not in tables:
        return []
    morning = {round(float(tables[today][h]), 4): h for h in range(6) if h in tables[today]}
    tonight = {round(float(tables[today][h]), 4) for h in (22, 23) if h in tables[today]}
    tonight |= {round(float(tables[tomorrow][h]), 4) for h in range(6) if h in tables.get(tomorrow, {})}
    issues = []
    for m in re.finditer(r"\$\s?(0\.\d{3,4})\b", answer or ""):
        r = round(float(m.group(1)), 4)
        if r in morning and r not in tonight:
            fetch = ("" if tomorrow in tables else f"; only {today}'s prices were retrieved: call get_electricity_prices for {tomorrow}")
            issues.append(f"${m.group(1)} is {today}'s {morning[r]:02d}:00 rate, this morning's, but the answer is about tonight, whose hours "
                          f"after midnight are {tomorrow}'s{fetch}, and quote tonight's rates")
    return list(dict.fromkeys(issues))


SOLAR_CLAIM = re.compile(r"\b(?:best|optimal|peak|prime|top|sunniest)\s+(?:solar|sun(?:light|shine)?)\b"
                         r"|\bsolar\s+(?:hours|window|peak)\b"
                         r"|\b(?:optimal|best|ideal)\s+(?:charging\s+|running\s+|run\s+)?(?:window|hours|times?)\b", re.I)


def solar_window_errors(messages, answer: str, today: Optional[str] = None) -> list:
    """Best-solar windows that are not the forecast's best_solar_window for the days they are given for.

    Each forecast day has its own window (on 19 Nov: Friday 10:00-15:00, Saturday 11:00-14:00). Live answers gave
    "the best solar hours are from 10:00 to 16:00" for Friday, and one charging window, 10:00 AM - 3:00 PM, for
    Friday, Saturday, Monday and Wednesday. The range judged is the first one after a best-solar phrase (or after
    'optimal window' in a section about solar, when the range is in daylight). It is given for the days its
    sentence names, else the days its section names ('tomorrow' and 'today' count, and '<condition> days' means every
    day with that condition); with no day named, it must at least be some day's window. A window one hour short is
    left to window_errors.
    """
    days = {}
    for d in _payloads(messages, "get_weather_forecast"):
        for x in d.get("daily", []):
            if x.get("date") and x.get("best_solar_hours"):
                days[x["date"]] = x
    if not days:
        return []
    today = _today(messages, today) or sorted(days)[0]
    tomorrow = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    by_weekday = {str(x.get("day_of_week", "")).lower(): k for k, x in days.items()}
    year = _year_hint(messages)

    def named(text: str) -> set:
        out = set(_dates_on(text, year)) | {by_weekday[w] for w in WEEKDAYS if w in by_weekday and re.search(rf"\b{w}\b", text, re.I)}
        if re.search(r"\btomorrow\b", text, re.I):
            out.add(tomorrow)
        if re.search(r"\btoday\b", text, re.I):
            out.add(today)
        for word, cond in CONDITIONS:
            if re.search(rf"\bon\s+{word}\s+days\b", text, re.I):
                out |= {k for k, x in days.items() if x.get("condition") == cond and k != today}
        return out & set(days)

    issues = []
    for block in _sections(answer):
        btext = " ".join(block)
        solar_block = re.search(r"\bsolar\b|\bsun", btext, re.I)
        for sent in block:
            for m in SOLAR_CLAIM.finditer(sent):
                generic = not re.search(r"solar|sun", m.group(0), re.I)
                if generic and not solar_block:
                    continue
                rng = next(((h, t) for h, t in _ranges(sent[m.end():m.end() + 60]) if t is not None), None)
                if not rng or len(rng[0]) < 2 or (generic and not rng[0] <= set(range(6, 19))):
                    continue
                hours, text = rng
                given = named(sent) or named(btext)
                # "10:00-15:00 on most days, with a slight variation on ... Sunday, where the window is 11:00-14:00"
                # (a live answer) gives Sunday its own window in the same sentence
                own = [h for h, t in _ranges(sent) if t is not None]
                if given:
                    wrong = [k for k in sorted(given) if set(days[k]["best_solar_hours"]) != hours
                             and hours != set(range(min(days[k]["best_solar_hours"]), max(days[k]["best_solar_hours"])))
                             and set(days[k]["best_solar_hours"]) not in own]
                elif not any(set(x["best_solar_hours"]) == hours for x in days.values()):
                    wrong = sorted(days)
                else:
                    wrong = []
                if wrong:
                    listed = "; ".join(f"{days[k].get('day_of_week')} {k} {days[k].get('best_solar_window')}" for k in wrong)
                    issues.append(f"'{text}' is given as the best solar hours, but best_solar_window is {listed}: "
                                  "give each day the window its forecast entry states")
    return list(dict.fromkeys(issues))


TEMP_UNIT = r"(?:°\s*([FC])\b|\s?degrees?\s+(Fahrenheit|Celsius)\b|\s?degrees?\b)"
TEMP_ONE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*" + TEMP_UNIT, re.I)
TEMP_PAIR = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:°\s*[FC]?\s*)?(?:to|-|–|and)\s*(\d+(?:\.\d+)?)\s*" + TEMP_UNIT, re.I)


def _temp_unit(short: Optional[str], word: Optional[str]) -> Optional[str]:
    unit = (short or word or "")[:1].upper()
    return unit if unit in ("F", "C") else None


def _temps(text: str) -> list:
    """(number as written, 'F' / 'C' / None) for every temperature in the text; '2 to 4 degrees Fahrenheit' gives both."""
    out = []
    for m in TEMP_PAIR.finditer(text or ""):
        u = _temp_unit(m.group(3), m.group(4))
        out += [(m.group(1), u), (m.group(2), u)]
    for m in TEMP_ONE.finditer(text or ""):
        out.append((m.group(1), _temp_unit(m.group(2), m.group(3))))
    return out


def grounded_temperatures(messages) -> tuple:
    """Temperatures the tools gave: forecast values (°C) and those written in retrieved tips, as (F, C, no unit) sets."""
    f, c, u = set(), set(), set()

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and "temp" in key.lower():
            c.add(float(x))
    for d in _payloads(messages, "get_weather_forecast"):
        walk(d)
    for d in _payloads(messages, "search_energy_tips"):
        for t in d.get("tips", []) if isinstance(d, dict) else []:
            for raw, unit in _temps(str(t.get("content", ""))):
                {"F": f, "C": c, None: u}[unit].add(float(raw))
    return f, c, u


def temperature_ok(raw: str, unit: Optional[str], grounded: tuple) -> bool:
    """A temperature (or a change in temperature) as written is a tool's figure or a unit conversion of one."""
    f, c, u = grounded
    f_ok = (any(_printed(x, raw) for x in f | u)
            or any(_printed(x * 9 / 5 + 32, raw) or _printed(x * 9 / 5, raw) for x in c | u))
    c_ok = (any(_printed(x, raw) for x in c | u)
            or any(_printed((x - 32) * 5 / 9, raw) or _printed(x * 5 / 9, raw) for x in f | u))
    return f_ok if unit == "F" else c_ok if unit == "C" else (f_ok or c_ok)


def temperature_errors(messages, answer: str, question: Optional[str] = None) -> list:
    """Temperatures no tool gave: a thermostat setting must come from a tip that search_energy_tips returned.

    The forecast gives temperatures in °C; settings and setbacks are in the knowledge base. Live answers advised
    "78°F (25.6°C)" after a search whose results did not contain it, and "increase by 2°F (to around 78°F)" with no
    search at all; the figure came from an example in the instructions. A conversion of a figure a tool gave
    (25.6°C for a tip's 78°F, 87.1°F for the forecast's 30.6°C) counts as that figure.
    """
    if not any(isinstance(m, ToolMessage) for m in messages):
        return []
    grounded = grounded_temperatures(messages)
    asked = {raw for raw, _ in _temps(question or "")}
    bad = []
    for raw, unit in _temps((answer or "").replace("**", "")):
        label = f"{raw}°{unit}" if unit else f"{raw} degrees"
        if raw in asked or temperature_ok(raw, unit, grounded) or label in bad:
            continue
        bad.append(label)
    if not bad:
        return []
    return [f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not a temperature any tool returned: the forecast gives "
            "temperatures in °C and thermostat settings come from search_energy_tips. Search for the guidance, give the "
            "setting or change the tip states (citing its file), convert units only from those figures, or leave the figure out"]


ESTIMATE_HEADING = re.compile(r"^\s*#{1,6}\s*(?:\d+\.\s*)?((?:(?:estimated|expected|potential|projected)\s+)?(?:annual\s+|daily\s+)?"
                              r"(?:savings?|costs?))\s*:?\s*$", re.I)
ESTIMATE_LABEL = re.compile(r"^\s*(?:[-*]\s*)?((?:estimated|expected|potential|projected)\s+(?:annual\s+|daily\s+)?(?:savings?|costs?))"
                            r"\s*:\s*(.*)$", re.I)
AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?(?!\d|\.\d)")   # a cost or saving; a per-kWh rate has 3-4 decimals


def empty_estimate_errors(answer: str) -> list:
    """A savings or cost section with no saving or cost in it.

    Live answers wrote "### Estimated Savings" and then "you can save significantly" (EV charging), "you would incur
    higher costs" (laundry) and "you can potentially save on electricity costs" (solar), with no figure. A heading's
    section runs to the next heading; a label ("Estimated Savings: ...") covers its own line, or the list under it.
    """
    lines = _clean_lines(answer)
    issues = []
    for i, line in enumerate(lines):
        head, label = ESTIMATE_HEADING.match(line), ESTIMATE_LABEL.match(line)
        if head:
            body, name = [], head.group(1)
            for nxt in lines[i + 1:]:
                if re.match(r"\s*#{1,6}\s", nxt):
                    break
                body.append(nxt)
        elif label:
            name = label.group(1)
            body = [label.group(2)] if label.group(2).strip() else []
            if not body:
                for nxt in lines[i + 1:]:
                    if not nxt.strip() or re.match(r"\s*#{1,6}\s", nxt) or ESTIMATE_LABEL.match(nxt):
                        break
                    body.append(nxt)
        else:
            continue
        if not AMOUNT.search(" ".join(body)):
            issues.append(f"the '{name.strip()}' section gives no saving or cost: value the move from the peak to off-peak "
                          "with calculate_energy_savings from the recorded usage (device_type, date, current_hours [16, 17, 18, 19, 20], "
                          "optimized_hours [0, 1, 2, 3, 4, 5]) and quote its result there, or leave the section out; never "
                          "assume a usage figure the history does not hold")
    return list(dict.fromkeys(issues))


MOSTLY = re.compile(r"\b(?:primarily|mostly|mainly|largely|predominantly|chiefly|the (?:majority|bulk) of)\b", re.I)


def share_claim_errors(messages, answer: str) -> list:
    """A device said to be used mostly at peak when its own figures say otherwise.

    A live answer said the EV used "218.91 kWh per day, primarily during peak hours"; the calculation it quoted
    moved the 61.63 kWh used in 16:00-21:00, which is 28% of the day.
    """
    peak = set()
    for d in _payloads(messages, "get_electricity_prices"):
        peak |= {r.get("hour") for r in d.get("hourly_rates", []) if isinstance(r, dict) and r.get("period") == "peak"}
    daily = {}
    for d in _payloads(messages, "query_energy_usage"):
        if d.get("device_name"):
            continue
        for name, v in (d.get("device_breakdown") or {}).items():
            if isinstance(v, dict) and isinstance(v.get("average_daily_kwh"), (int, float)) and v["average_daily_kwh"] > 0:
                daily[_device_group(name)] = v["average_daily_kwh"]
    at_peak = {}
    for c in _payloads(messages, "calculate_energy_savings"):
        if (peak and set(c.get("current_hours") or []) == peak and str(c.get("usage_from", "")).startswith("average daily")
                and isinstance(c.get("current_usage_kwh"), (int, float))):
            at_peak[_device_group(str(c.get("device_type", "")))] = c["current_usage_kwh"]
    issues = []
    for line in _clean_lines(answer):
        for sent in re.split(r"(?<=[.!?])\s+|;\s*", line):
            if not (MOSTLY.search(sent) and PEAK_WORD.search(sent)):
                continue
            for g in {_device_group(w) for w in DEVICE_WORDS.findall(sent)}:
                if g in at_peak and g in daily and at_peak[g] < 0.5 * daily[g]:
                    issues.append(f"the answer says {g.upper() if g in ('ev', 'hvac') else g} use is mostly at peak, but "
                                  f"{at_peak[g]:.2f} kWh of its {daily[g]:.2f} kWh a day ({at_peak[g] / daily[g]:.0%}) is in the peak "
                                  f"{_window(peak)}: state the peak share as the figures give it")
    return list(dict.fromkeys(issues))


SOLAR_HVAC_CUT = re.compile(
    r"\b(?:reduc\w*|cut\w*|lower\w*|less|minimi[sz]\w*)\s+(?:your\s+|the\s+)?(?:HVAC|cooling|heating|air[- ]condition\w*|AC)\b"
    r"[^.;]{0,40}?\b(?:during|when|while|in)\b[^.;]{0,40}?\b(?:solar|sun)"
    r"|\b(?:higher|raise|increase)\b[^.;]{0,40}?\b(?:temperature|setpoint|set point|thermostat)\b[^.;]{0,30}?"
    r"\b(?:during|when|while|in)\b[^.;]{0,30}?\b(?:solar|sun)", re.I)


def solar_hvac_cut_errors(question: str, answer: str) -> list:
    """In an answer about using more of the home's own solar, advice to run the HVAC less while the sun is out.

    That sends more of the solar to the grid. A live answer told the customer to "set your thermostat to a higher
    temperature during peak solar hours ... to reduce HVAC usage when solar generation is high"; using the solar means
    pre-cooling (or pre-heating) in those hours and easing off at the evening peak.
    """
    asked = {name for name, pattern, _, _ in DATA_RULES if pattern.search(question or "")}
    if "own solar" not in asked:
        return []
    m = SOLAR_HVAC_CUT.search((answer or "").replace("**", ""))
    if not m:
        return []
    return [f"'{m.group(0)[:90]}' cuts HVAC use while the panels produce, which sends that solar to the grid: to use more of "
            "your own solar, run the HVAC in the solar hours (pre-cool or pre-heat, as search_energy_tips describes) and ease "
            "off at the evening peak"]


def _quoted_calcs_of(messages, answer: str, keys=("savings_usd", "annual_savings_usd", "current_cost_usd", "optimized_cost_usd")) -> list:
    """Successful calculate_energy_savings results whose money figures the answer quotes."""
    found = _numbers(answer)
    return [c for c in _payloads(messages, "calculate_energy_savings")
            if _matches(found, {abs(c.get(k)) for k in keys if isinstance(c.get(k), (int, float)) and c.get(k)})]


def _calc_date(c: dict) -> Optional[str]:
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(c.get("prices_from", "")))
    return m.group(0) if m else None


def solar_shift_errors(messages, answer: str) -> list:
    """A quoted saving for moving use into the solar hours that was priced with no solar counted.

    calculate_energy_savings knows nothing of the panels: a move into 10:00-15:00 is priced as mid-peak grid power.
    Live answers moved the EV's 61.63 kWh of peak charging "to the solar hours" and quoted the $5.98 a day of cheaper
    grid power as the solar saving, on days whose forecast solar was 35.64 kWh (and, on cloudy Tuesday, 11.41 kWh),
    while off-peak charging saved $8.90. calculate_energy_savings now counts the forecast's solar in those hours when given
    location (and refuses daylight hours without it), so only a result without its solar fields is reported here.
    """
    mid = {}
    for d in _payloads(messages, "get_electricity_prices"):
        mid[d.get("date")] = {r.get("hour") for r in d.get("hourly_rates", []) if isinstance(r, dict) and r.get("period") == "mid_peak"}
    solar = {}
    for d in _payloads(messages, "get_weather_forecast"):
        for x in d.get("daily", []):
            solar[x.get("date")] = x
    issues = []
    for c in _quoted_calcs_of(messages, answer, ("savings_usd", "annual_savings_usd", "optimized_cost_usd")):
        ho = set(c.get("optimized_hours") or [])
        if not ho or not c.get("current_hours") or c.get("solar_from"):   # the tool counted the forecast's solar (the fields are always present)
            continue
        hn = set(c.get("current_hours") or [])
        if hn and hn <= set(range(9, 16)):
            # a live example moved 24.72 kWh of EV charging OUT of 11:00-14:00 and reported $1.10 a day saved, pricing the
            # midday hours as grid power although the panels cover much of them
            issues.append(f"the saving quoted for moving {c.get('current_usage_kwh')} kWh out of {_window(hn)} prices those hours as grid "
                          "power, ignoring the solar the use has there: call calculate_energy_savings again with location; a move out "
                          "of the solar hours rarely saves anything")
            continue
        day = _calc_date(c)
        daytime = mid.get(day) or set(range(6, 16))
        if ho <= daytime and ho & set(range(9, 16)):
            s = solar.get(day, {})
            covers = (f"; the forecast's solar for {day} is {s.get('estimated_solar_kwh')} kWh in {s.get('best_solar_window')}"
                      if s.get("estimated_solar_kwh") is not None else "")
            issues.append(f"the saving quoted for moving {c.get('current_usage_kwh')} kWh to {_window(ho)} is grid power at mid-peak rates "
                          f"(${c.get('optimized_price_per_kwh')}/kWh), with no solar counted{covers}: call calculate_energy_savings "
                          "again with location so it counts the forecast's solar in those hours, and compare that with the move to "
                          "off-peak (optimized_hours [0, 1, 2, 3, 4, 5])")
    return list(dict.fromkeys(issues))


REC_HEAD = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\d+\.\s*)?recommendations?\b\s*:?", re.I)


def recommendation_calc_errors(messages, answer: str) -> list:
    """A saving quoted for a move to other hours than the ones the answer recommends.

    A live laundry answer recommended 10:00-15:00 and quoted the $1.12 a day of a move to 22:00-03:00. Judged per
    section: the hours named in its recommendation (a 'Recommendation' heading or label) against each quoted shift's
    optimized_hours; a single 'Recommendation' section at the top is judged against every shift the answer quotes.
    """
    shifts = [c for c in _quoted_calcs_of(messages, answer, ("savings_usd", "annual_savings_usd"))
              if c.get("current_hours") and c.get("optimized_hours")]
    if not shifts:
        return []
    issues = []
    for block in _sections(answer):
        rec = [x for x in block if REC_HEAD.match(x)]
        if not rec:
            continue
        head_only = re.match(r"\s*#{1,6}\s*recommendations?\s*:?\s*$", block[0], re.I)
        text = " ".join(block[1:] if head_only else rec)
        named = [h for h, t in _ranges(text) if t is not None and len(h) >= 2]
        if not named:
            continue
        own = [c for c in shifts if any(_printed(v, t.rstrip(".,")) for t in MONEY.findall(" ".join(block))
                                        for v in (abs(c.get("savings_usd") or 0), abs(c.get("annual_savings_usd") or 0)) if v)]
        for c in (own or (shifts if head_only else [])):
            ho = set(c["optimized_hours"])
            if not any(h & ho for h in named):
                issues.append(f"the saving quoted (${c.get('savings_usd')} a day) is for moving use to {_window(ho)}, but the "
                              f"recommendation names {', '.join(_window(h) for h in named)}: recommend the hours that were valued, "
                              "or value the hours you recommend")
    return list(dict.fromkeys(issues))


def section_window_errors(messages, answer: str) -> list:
    """A section quoting a saving that names a whole tariff period when the saving was calculated for part of it.

    A live answer quoted the $0.15 a day of a 10% HVAC cut calculated over 16:00-18:00 as "reducing HVAC usage by 10%
    during the peak hours (16:00-21:00)". window_errors leaves whole periods alone on a line (they usually describe the
    tariff); here the period is named in the same section as the saving, and the calculated hours are named nowhere in it.
    """
    periods = []
    for d in _payloads(messages, "get_electricity_prices"):
        for per in ("peak", "off_peak"):
            hs = {r.get("hour") for r in d.get("hourly_rates", []) if isinstance(r, dict) and r.get("period") == per}
            if hs:
                periods.append(hs)
    quoted = _quoted_calcs_of(messages, answer, ("savings_usd", "annual_savings_usd"))
    issues = []
    for block in _sections(answer):
        text = " ".join(block)
        named = [(h, t) for h, t in _ranges(text) if t is not None and len(h) >= 2]
        for c in quoted:
            if not any(_printed(v, t.rstrip(".,")) for t in MONEY.findall(text)
                       for v in (abs(c.get("savings_usd") or 0), abs(c.get("annual_savings_usd") or 0)) if v):
                continue
            hours = [set(c.get(k)) for k in ("current_hours", "optimized_hours") if c.get(k)]
            if not hours or any(h in hours for h, _ in named):
                continue
            for h, t in named:
                if h in periods and any(h & x and h != x for x in hours):
                    issues.append(f"the saving quoted beside '{t}' was calculated for {', '.join(_window(x) for x in hours)} "
                                  f"('{str(c.get('summary', ''))[:90]}'): name the hours that were priced")
    return list(dict.fromkeys(issues))


def cited_temperature_errors(messages, answer: str) -> list:
    """A temperature given with a source file that does not contain it.

    Live answers set "78°F ... recommended by the U.S. Department of Energy (source: tip_hvac_optimization.txt)" and
    listed 68°F and 78°F under tip_hvac_optimization.txt and tip_seasonal_energy_management.txt; both figures are written
    only in tip_energy_savings.txt. A line's own citation is used when it has one, else every file the answer cites.
    """
    by_file = {}
    for d in _payloads(messages, "search_energy_tips"):
        for t in d.get("tips", []) if isinstance(d, dict) else []:
            by_file.setdefault(t.get("source"), []).append(str(t.get("content", "")))
    cited_all = set(CITED.findall(answer or "")) & set(by_file)
    if not cited_all:
        return []
    forecast = [m for m in messages if isinstance(m, ToolMessage) and m.name == "get_weather_forecast"]
    issues = []
    for line in _clean_lines(answer):
        temps = _temps(line)
        if not temps:
            continue
        cited = set(CITED.findall(line)) & set(by_file) or cited_all

        def grounded(files):
            fake = [ToolMessage(json.dumps({"tips": [{"source": f, "content": c} for f in files for c in by_file[f]]}),
                                tool_call_id="cite", name="search_energy_tips")]
            return grounded_temperatures(forecast + fake)
        mine, every = grounded(cited), grounded(set(by_file))
        for raw, unit in temps:
            if not temperature_ok(raw, unit, mine) and temperature_ok(raw, unit, every):
                where = sorted(f for f in by_file if temperature_ok(raw, unit, grounded({f})))
                issues.append(f"{raw}°{unit or ''} is given with {', '.join(sorted(cited))}, which does not contain it; it comes from "
                              f"{', '.join(where)}: cite the file each figure comes from")
    return list(dict.fromkeys(issues))


def large_cut_errors(question: str, messages, answer: str) -> list:
    """A cut of half or more of a device's use quoted as a saving the question never asked about.

    A live dishwasher answer added "if you were to reduce your dishwasher usage entirely (100% reduction), you could
    save $1.66 per day": not running the dishwasher is not a way to run it more cheaply.
    """
    asked = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", question or "")}
    issues = []
    for c in _quoted_calcs_of(messages, answer, ("savings_usd", "annual_savings_usd")):
        pct = c.get("reduction_percent")
        if isinstance(pct, (int, float)) and pct >= 50 and float(pct) not in asked:
            what = f" ({str(c.get('summary'))[:80]})" if c.get("summary") else ""
            issues.append(f"the answer quotes a {pct:g}% cut{what}, which means not using most of the {c.get('device_type') or 'device'}'s "
                          "energy at all; the question asks about no such cut: leave it out, or use a reduction a tip supports")
    return issues


def named_machine_errors(question: str, messages, answer: str) -> list:
    """A question about the washing machine and dryer answered with every appliance's use (the dishwasher too).

    A live laundry answer called 7.76 kWh "your washing machine and dryer combined"; that is every appliance's use in
    16:00-21:00. The tool values one machine with device_type "washing machine" or "dryer".
    """
    q = (question or "").lower()
    if not re.search(r"\bwashing machine|\bwasher\b|\bdryer\b", q) or re.search(r"\bappliances?\b", q):
        return []
    issues = []
    for c in _quoted_calcs_of(messages, answer):
        if str(c.get("device_type", "")).strip().lower() in ("appliance", "appliances") and "Dishwasher" not in str(c.get("usage_from")):
            issues.append(f"the saving quoted is for every appliance ({c.get('current_usage_kwh')} kWh, the dishwasher included), but the "
                          "question asks about the washing machine and dryer: call calculate_energy_savings with device_type "
                          "'washing machine' and 'dryer' and quote those")
            break
    return issues


def suggestion_tip_errors(question: str, messages, answer: str) -> list:
    """In a 'suggest ways ... based on my usage history' answer, a suggestion with no retrieved tip applied and cited.

    The review asked that the answer "apply the retrieved guidance to the household usage patterns, and cite the
    returned source files". A live answer gave its EV and appliance suggestions each a tip with its source and its
    HVAC suggestion none. Each numbered suggestion (outside a Tips section) must cite a file search_energy_tips returned.
    """
    asked = {name for name, pattern, _, _ in DATA_RULES if pattern.search(question or "")}
    if not {"suggestions", "usage history"} <= asked:
        return []
    returned = {t.get("source") for d in _payloads(messages, "search_energy_tips") for t in d.get("tips", [])} - {None}
    if not returned:
        return []
    items, heading = [], ""
    for block in _sections(answer):
        head = block[0] if block else ""
        if re.match(r"\s*#{1,6}\s", head) and not re.match(r"\s*#{1,6}\s*\d+\.", head):
            heading = head
            continue
        if re.match(r"\s*(?:#{1,6}\s*)?\d+\.\s", head) and not re.search(r"\btips?\b|\bsummary\b", heading, re.I):
            items.append(block)
    if len(items) < 2:
        return []
    issues = []
    for block in items:
        if not set(CITED.findall(" ".join(block))) & returned:
            name = re.sub(r"^\s*(?:#{1,6}\s*)?\d+\.\s*", "", block[0]).strip("*: ")[:60]
            issues.append(f"the suggestion '{name}' applies no retrieved tip: apply a passage search_energy_tips returned to this "
                          f"device's usage and cite its file (returned: {', '.join(sorted(returned))}); search again for this device "
                          "if none fits")
    return issues


def offpeak_listing_errors(messages, answer: str) -> list:
    """The off-peak period listed in parts that leave hours out.

    A live answer said "the off-peak hours, which are the cheapest, are from 00:00 to 06:00 and 22:00 to 23:00": the
    off-peak period is 22:00-06:00, so 23:00-24:00 is missing (ranges end where the last hour ends). A sentence about
    off-peak that names two or more ranges covering both sides of midnight is a listing of the period, so together they
    must make up the whole of off_peak_hours. One range ("off-peak hours (00:00-06:00)") is a choice within it.
    """
    off = set()
    for d in _payloads(messages, "get_electricity_prices"):
        off |= set((d.get("summary") or {}).get("off_peak_hours") or [])
    if not off:
        return []
    window = _window(off)
    issues = []
    for line in _clean_lines(answer):
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(*\-])", line):
            if not re.search(r"\boff-?\s?peak\b", sent, re.I):
                continue
            spans = [(h, t) for h, t in _ranges(sent) if t is not None and h]
            if len(spans) < 2:
                continue
            # "22:00-23:59" runs to midnight (a live answer): an end at :59 includes that hour
            spans = [(set(h) | {int(m.group(1)) % 24} if (m := re.search(r"(\d{1,2}):59\s*$", t)) else h, t) for h, t in spans]
            union = set().union(*(h for h, _ in spans))
            if union < off and any(h < 12 for h in union) and any(h >= 12 for h in union):
                listed = " and ".join(t for _, t in spans)
                issues.append(f"'{listed}' leaves {_window(off - union)} out of the off-peak period, which is {window} "
                              "(off_peak_window; a range ends where its last hour ends): give the whole period, or one range you choose within it")
    return list(dict.fromkeys(issues))


USE_PEAK = re.compile(r"\b(?:highest|most|heaviest|peak)\s+(?:energy\s+|electricity\s+)?(?:usage|use|consumption|demand)\b"
                      r"|\b(?:usage|use|consumption|demand)\s+(?:is\s+|peaks?\s+|tends\s+to\s+be\s+)?(?:highest|heaviest)\b", re.I)


def usage_peak_errors(messages, answer: str) -> list:
    """Hours called the household's highest use that leave out an hour used more.

    A live answer said "the highest usage during the evening peak hours (18:00-21:00)"; the history's hour-of-day
    profile has 21:00-22:00 at 5.86 kWh, above 18:00-19:00 at 5.68 kWh, so the busiest block is 18:00-22:00. The range
    must hold every neighbouring hour used more than its own least-used hour. Judged on the whole-house profile only, and
    not when the range is the tariff's peak or off-peak window (a statement about prices, not about use).
    """
    profile = {}
    for d in _payloads(messages, "query_energy_usage"):
        if d.get("device_type") in (None, "", "all") and len(d.get("device_breakdown") or {}) > 1:
            profile = {x.get("hour"): x.get("average_kwh") for x in d.get("hour_of_day_profile") or [] if isinstance(x, dict)}
    if not profile:
        return []
    tariff = []
    for d in _payloads(messages, "get_electricity_prices"):
        sm = d.get("summary") or {}
        tariff += [set(sm.get("peak_hours") or []), set(sm.get("off_peak_hours") or [])]
    issues = []
    for line in _clean_lines(answer):
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(*\-])", line):
            if not USE_PEAK.search(sent):
                continue
            spans = [(h, t) for h, t in _ranges(sent) if t is not None and len(h) >= 2]
            if len(spans) != 1 or any(spans[0][0] == t for t in tariff if t):
                continue
            hours, text = spans[0]
            if not all(h in profile for h in hours):
                continue
            low = min(hours, key=lambda h: profile[h])
            edge = {(min(hours) - 1) % 24, (max(hours) + 1) % 24} - set(hours)
            above = sorted(h for h in edge if h in profile and profile[h] > profile[low])
            if above:
                h = above[0]
                issues.append(f"'{text}' is given as the highest use, but {h:02d}:00-{(h + 1) % 24:02d}:00 averages {profile[h]} kWh, above "
                              f"{low:02d}:00-{(low + 1) % 24:02d}:00 at {profile[low]} kWh (hour_of_day_profile): name the hours as the "
                              f"profile gives them ({_window(set(hours) | {h})})")
    return list(dict.fromkeys(issues))


STOP = set("a an and are as at be by can for from in into is it its of on or that the this to with your you when which "
           "will so than then these those during per day kwh".split())


def _content_tokens(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower().replace("-", " ")) if t not in STOP]


def _grams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def cited_advice_errors(messages, answer: str) -> list:
    """A suggestion citing a returned file that does not hold its advice, when another returned file does.

    A live answer's third suggestion, "Delay-start large appliances like dishwashers, washing machines, and dryers to
    run during off-peak periods", cited tip_energy_savings.txt; that advice is in tip_smart_home_automation.txt
    ("Delay-start large appliances: dishwashers, washing machines and dryers ..."), which the same search returned.
    Lexical evidence, so it fires only when it is strong both ways: the item shares a four-word phrase (stop words
    removed) with a returned file it does not cite, and not even a two-word phrase with the files it cites.
    """
    texts = {}
    for d in _payloads(messages, "search_energy_tips"):
        for t in d.get("tips", []) or []:
            if t.get("source"):
                texts[t["source"]] = texts.get(t["source"], "") + " " + str(t.get("content", ""))
    if len(texts) < 2:
        return []
    issues = []
    for block in _sections(answer):
        body = " ".join(block)
        cited = set(CITED.findall(body)) & set(texts)
        if not cited:
            continue
        tokens = _content_tokens(CITED.sub(" ", body))
        own = set().union(*(_grams(_content_tokens(texts[c]), 2) for c in cited))
        if _grams(tokens, 2) & own:
            continue
        mine = _grams(tokens, 4)
        for src in sorted(set(texts) - cited):
            shared = mine & _grams(_content_tokens(texts[src]), 4)
            if shared:
                phrase = " ".join(sorted(shared)[0])
                name = re.sub(r"^\s*(?:#{1,6}\s*)?\d+\.\s*", "", block[0]).strip("*: #")[:60]
                issues.append(f"'{name}' cites {', '.join(sorted(cited))}, which does not hold its advice ('{phrase}' is in {src}, "
                              f"which the search also returned): cite the file the advice comes from")
                break
    return list(dict.fromkeys(issues))


def kwh_figure_errors(messages, answer: str, question: Optional[str] = None) -> list:
    """A kWh figure (with decimals) that no tool returned: energy the model added up or worked out itself.

    A live answer wrote that Saturday's solar window "is expected to generate approximately 15.24 kWh", adding the
    forecast's hours itself; the forecast gives 17.24 kWh for those hours (now returned as solar_kwh_in_best_window).
    Whole numbers (rules of thumb, a stated "40 kWh") and the question's own figures are left alone.
    """
    if not any(isinstance(m, ToolMessage) for m in messages):
        return []
    values = _tool_values(messages)
    asked = set(KWH.findall(question or ""))
    bad = []
    for raw in KWH.findall((answer or "").replace("**", "")):
        if "." not in raw or raw in asked or raw in bad:
            continue
        if not any(_printed(v, raw.replace(",", "")) for v in values):
            bad.append(raw)
    if not bad:
        return []
    return [f"{', '.join(k + ' kWh' for k in bad)} {'is' if len(bad) == 1 else 'are'} not a figure any tool returned: never add up or "
            "work out energy yourself. Quote the tools' figures (e.g. solar_kwh_in_best_window or estimated_solar_kwh from the "
            "forecast, the kWh calculate_energy_savings used), or leave the figure out"]


def calculation_basis(messages, answer: str) -> list:
    """The summary sentence of every successful calculate_energy_savings result whose figures the answer quotes."""
    found, out = _numbers(answer), []
    for c in _payloads(messages, "calculate_energy_savings"):
        figs = {abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd", "current_cost_usd", "optimized_cost_usd")
                if isinstance(c.get(k), (int, float)) and c.get(k)}
        if c.get("summary") and _matches(found, figs) and c["summary"] not in out:
            out.append(c["summary"])
    return out


def data_errors(messages, answer: str, today: Optional[str] = None, question: Optional[str] = None) -> list:
    """Every code-decidable figure check, in one place: the advisor's validation and the evaluation both call it."""
    return list(dict.fromkeys(usage_period_errors(messages, answer) + rate_hour_errors(messages, answer)
                              + money_errors(messages, answer) + date_figure_errors(messages, answer)
                              + window_errors(messages, answer) + calc_description_errors(messages, answer)
                              + tonight_errors(messages, answer, today) + stated_amount_errors(question, messages, answer)
                              + unretried_calc_errors(messages) + no_change_errors(question, messages, answer)
                              + calc_kwh_errors(messages, answer) + cheapest_claim_errors(messages, answer)
                              + condition_errors(messages, answer) + unquantified_errors(question, messages, answer)
                              + superlative_rate_errors(messages, answer) + coverage_errors(messages, answer)
                              + payback_errors(question, messages, answer) + tonight_range_errors(messages, answer, today) + tonight_rate_errors(messages, answer, today)
                              + solar_window_errors(messages, answer, today) + temperature_errors(messages, answer, question)
                              + empty_estimate_errors(answer) + share_claim_errors(messages, answer)
                              + solar_hvac_cut_errors(question, answer) + solar_shift_errors(messages, answer)
                              + recommendation_calc_errors(messages, answer) + section_window_errors(messages, answer)
                              + cited_temperature_errors(messages, answer) + large_cut_errors(question, messages, answer)
                              + named_machine_errors(question, messages, answer) + suggestion_tip_errors(question, messages, answer)
                              + kwh_figure_errors(messages, answer, question) + offpeak_listing_errors(messages, answer)
                              + usage_peak_errors(messages, answer) + cited_advice_errors(messages, answer)))


def content_issues(question: str, messages, answer: str, today: Optional[str] = None) -> list:
    """Ways the answer fails to use the data it retrieved, decided in code. Empty when there are none."""
    issues = []
    returned = {t.get("source") for d in _payloads(messages, "search_energy_tips") for t in d.get("tips", [])}
    bad = sorted(set(CITED.findall(answer or "")) - returned)
    if bad:
        issues.append(f"you cited {', '.join(bad)}, which no search_energy_tips result in this conversation returned: "
                      "cite only files that search returned (call search_energy_tips if you need a tip)")
    for claim in SOURCE_CLAIM.findall(answer or ""):
        if not set(CITED.findall(claim)) & returned:
            issues.append(f"the answer attributes a tip to '{claim.strip()[:60]}', which is not a file search_energy_tips returned: "
                          "call search_energy_tips and cite the file name it returns (tip_....txt), or state the tip without a source")
            break
    asked = {name for name, pattern, _, _ in DATA_RULES if pattern.search(question or "")}
    found = _numbers(answer)
    drafts = [m for m in _this_turn(messages)[:-1] if isinstance(m, AIMessage) and m.tool_calls
              and len(str(m.content or "").strip()) > 150]
    if drafts:
        issues.append("you wrote part of the answer before calling tools, and the customer sees only your last message: "
                      "write the complete answer now in one message (recommendation, why with the data, estimated savings, tips)")
    if "usage history" in asked:
        history = set()
        for d in _payloads(messages, "query_energy_usage"):
            history |= {v for v in data_figures(d) if v != int(v) or v > 31}
        for c in _payloads(messages, "calculate_energy_savings"):
            if str(c.get("usage_from", "")).startswith("average daily"):
                history |= {c.get("current_usage_kwh"), c.get("optimized_usage_kwh")} - {None}
        if history and not _quotes(found, history):
            issues.append("the answer does not say what the usage history shows: state the kWh behind each suggestion "
                          "(for example a device's daily use from query_energy_usage or the usage calculate_energy_savings read)")
    # any answer built on retrieved tips cites them (a live thermostat answer used the knowledge base and named no file)
    if (returned - {None}) and not (set(CITED.findall(answer or "")) & returned):
        issues.append("cite the source file name (tip_....txt) that search_energy_tips returned for each tip you use, "
                      "e.g. (source: " + sorted(r for r in returned if r)[0] + ")")
    if "own solar" in asked:
        history = set()
        for d in _payloads(messages, "query_solar_generation"):
            history |= {v for v in data_figures(d) if v != int(v) or v > 31}
        # a figure the forecast also contains is the forecast's (Sunday's forecast 22.51 is not last week's 22.52 average)
        forecast = set()
        for d in _payloads(messages, "get_weather_forecast"):
            forecast |= data_figures(d)
        if history and not _quotes(found - {f for f in found if _quotes({f}, forecast)}, history):
            issues.append("the answer does not use the solar history you retrieved: state what the panels produced "
                          "(for example the total or the daily average from query_solar_generation) and compare it with the forecast")
        use = set()
        for d in _payloads(messages, "query_energy_usage"):
            use |= {v for v in data_figures(d) if v != int(v) or v > 31}
        if use and not _quotes(found, use):
            issues.append("the answer does not set the solar against the household's use: state the daily kWh of the loads you "
                          "suggest moving (from query_energy_usage) beside the forecast, so the customer sees how much solar can cover; "
                          "never ask the customer for data a tool returns")
    if "quantified savings" in asked:
        calcs = _payloads(messages, "calculate_energy_savings")
        if calcs and all(c.get("savings_usd") == 0 and c.get("savings_kwh") == 0 for c in calcs):
            issues.append("calculate_energy_savings returned no saving because the inputs describe no change: "
                          "give reduction_percent for a reduction, or date, current_hours and optimized_hours for a shift")
        elif calcs:
            def figures_of(c):
                return {abs(c.get(k)) for k in ("savings_usd", "annual_savings_usd") if isinstance(c.get(k), (int, float)) and c.get(k)}
            quoted = [c for c in calcs if _matches(found, figures_of(c))]
            if not quoted:
                issues.append("the answer does not state the saving calculate_energy_savings returned: quote its summary sentence")
            paybacks = {c.get("payback_years") for c in calcs if isinstance(c.get("payback_years"), (int, float))}
            if paybacks and not any(_printed(v, t) for v in paybacks for t in NUMBER.findall(answer or "") if "." in t):
                issues.append(f"calculate_energy_savings gave a payback of {sorted(paybacks)[0]} years: quote payback_years as returned "
                              "rather than working it out")
            months = {c.get("payback_months") for c in calcs if isinstance(c.get("payback_months"), (int, float))}
            said = re.findall(r"(\d+(?:\.\d+)?)\s*months?\b", answer or "", re.I)
            if months and said and not any(_printed(v, t) for v in months for t in said):
                issues.append(f"the answer gives the payback as {said[0]} months, but calculate_energy_savings returned payback_months "
                              f"{sorted(months)[0]}: quote it as returned (12 x a rounded payback_years is not the same figure)")
            table = {r.get("rate") for d in _payloads(messages, "get_electricity_prices") for r in d.get("hourly_rates", [])}
            for c in quoted:
                looked_up = str(c.get("prices_from", "")).startswith("time-of-use rates")
                in_table = {c.get("price_per_kwh"), c.get("optimized_price_per_kwh")} <= table
                if not looked_up and not in_table:
                    issues.append(f"calculate_energy_savings used ${c.get('price_per_kwh')} and ${c.get('optimized_price_per_kwh')} per kWh, "
                                  "which are not time-of-use rates from get_electricity_prices (the usage history was billed on an older "
                                  "tariff): call it again without prices, giving date (and current_hours and optimized_hours for a "
                                  "shift, or reduction_percent for a reduction) so it looks the rates up")
    issues += data_errors(messages, answer, today, question)
    return issues


def _this_turn(messages) -> list:
    """Messages since the advisor last started answering: after the question, or after the latest validation note."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
                 or (isinstance(m, SystemMessage) and str(m.content).startswith("Validation:"))), default=-1)
    return list(messages[start + 1:])


def successful_tools(messages) -> set:
    """Names of tools that returned a result that is not an error."""
    ok = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            if m.status != "error" and not text.lstrip().startswith('{"error"'):
                ok.add(m.name)
    return ok


class AdvisorState(TypedDict, total=False):
    """Graph state: the conversation, the context, a step counter, and the validation record."""
    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    question: str
    steps: int
    required_tools: list
    missing_required_tools: list
    content_issues: list
    validation_retries: int
    retry: bool
    completed_by_code: list


TRANSIENT_MODULES = ("openai", "httpx", "httpcore")


def _origin(e: BaseException) -> str:
    """file:line of the innermost frame that raised the exception, with the path cut to its package."""
    tb = traceback.extract_tb(e.__traceback__)
    if not tb:
        return "unknown"
    frame = tb[-1]
    parts = frame.filename.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in TRANSIENT_MODULES or part in ("langchain_openai", "langchain_core", "langgraph"):
            return f"{'/'.join(parts[i:])}:{frame.lineno}"
    return f"{parts[-1]}:{frame.lineno}"


def describe_error(e: BaseException) -> dict:
    """Type, message and origin of an exception: what the run log needs to tell an endpoint fault from a code fault."""
    return {"type": type(e).__name__, "message": str(e)[:300], "origin": _origin(e), "transient": is_transient(e)}


def is_transient(e: BaseException) -> bool:
    """A fault in reaching the model, worth one more try; a fault in this project's own code is not retried.

    The course endpoint once answered with a body the OpenAI client could not parse (JSONDecodeError "Extra data"),
    which ended a whole test. The client's own retries cover HTTP errors but not a malformed body. An error counts
    as transient when it is the client's connection, timeout, rate-limit or server error, or when the innermost
    frame that raised it is inside the client libraries; the same JSONDecodeError raised by code here is a bug.
    """
    name = type(e).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "ConnectError", "ReadTimeout",
                "RemoteProtocolError", "ReadError"):
        return True
    tb = traceback.extract_tb(e.__traceback__)
    inner = tb[-1].filename.replace("\\", "/").split("/") if tb else []
    return isinstance(e, (ValueError, OSError)) and any(m in inner for m in TRANSIENT_MODULES)


class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o", max_steps: int = 12,
                 llm: Any = None, tools: Optional[list] = None, max_validation_retries: int = 2,
                 endpoint_attempts: int = 3, retry_wait: float = 2.0):
        """
        Build the Energy Advisor.

        Args:
            instructions: the system prompt (who the advisor is, what to do, how to answer).
            model: model name on the course endpoint (default "gpt-4o"). In evaluation, gpt-4o-mini kept missing
                layered rules (each day's own solar window, which hours a saving was calculated for, which file a figure
                came from) even after being sent back, so the default is the stronger model. Any model the endpoint
                serves can be passed instead.
            max_steps: most model turns per question; stops a run that keeps calling tools.
            llm: optional chat model to use instead of the course endpoint (used for offline tests).
            tools: optional tool list (defaults to every tool in tools.TOOL_KIT).
            max_validation_retries: how many times an answer missing required data is sent back (default 2).
            endpoint_attempts: how many times a question is tried when reaching the model fails in the client
                (a dropped connection or a response body it cannot parse); errors in this code are not retried.
            retry_wait: seconds to wait before the second attempt (doubled before the third).
        """
        if not instructions or not instructions.strip():
            raise ValueError("instructions must not be empty: the system prompt defines the advisor")
        self.instructions = instructions
        self.model = model
        self.max_steps = max_steps
        self.max_validation_retries = max_validation_retries
        self.endpoint_attempts = max(1, int(endpoint_attempts))
        self.retry_wait = retry_wait
        self.tools = list(tools or TOOL_KIT)

        # LLM: temperature 0 for repeatable recommendations; the course endpoint and key.
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=VOCAREUM_BASE_URL,
            api_key=os.getenv("VOCAREUM_API_KEY"),
            timeout=60,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    # ---------------------------------------------------------------- graph

    def _agent_node(self, state: AdvisorState) -> Dict[str, Any]:
        """Call the model with the instructions, the run's context and the conversation so far."""
        steps = state.get("steps", 0)
        if steps >= self.max_steps:
            return {"messages": [AIMessage(content=(
                "I stopped before finishing because this question needed more steps than allowed "
                f"({self.max_steps}). Please ask a narrower question."))], "steps": steps}
        system = [SystemMessage(content=self.instructions)]
        if state.get("context"):
            system.append(SystemMessage(content=f"Context for this request:\n{state['context']}"))
        reply = self.llm_with_tools.invoke(system + list(state["messages"]))
        return {"messages": [reply], "steps": steps + 1}

    @staticmethod
    def _route(state: AdvisorState) -> str:
        """Go to the tools when the model asked for any; otherwise validate the answer."""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "validate"

    def _validate_node(self, state: AdvisorState) -> Dict[str, Any]:
        """Check the answer rests on the data its question needs; if not, send the model back (bounded).

        Only successful tool results count, read from the ToolMessages themselves, so a call that errored
        or was never made is missing. The draft answer stays in the history; the note that follows it says
        exactly which tools are missing and why, and the model answers again after calling them.
        """
        question = state.get("question", "")
        needed = required_tools(question)
        missing = [t for t in needed if t not in successful_tools(state["messages"])]
        answer = state["messages"][-1].content if state["messages"] else ""
        today = re.search(r"Current date and time: (\d{4}-\d{2}-\d{2})", state.get("context") or "")
        issues = [] if missing else content_issues(question, state["messages"], answer if isinstance(answer, str) else str(answer),
                                                   today.group(1) if today else None)
        retries = state.get("validation_retries", 0)
        can_retry = retries < self.max_validation_retries and state.get("steps", 0) < self.max_steps
        if missing and can_retry:
            note = (f"Validation: the answer above is not final. Before answering, call {', '.join(missing)} "
                    f"({_reasons(question, missing)}). Then answer again using those results, and cite the source "
                    "file of every tip you use.")
        elif issues and can_retry:
            note = "Validation: the answer above is not final. Fix this and answer again: " + "; ".join(issues) + "."
        else:
            done = {"required_tools": needed, "missing_required_tools": missing, "content_issues": issues, "retry": False}
            fixed = self._add_calculation_basis(state, question, answer if isinstance(answer, str) else str(answer),
                                                issues, today.group(1) if today else None) if issues and not missing else None
            if fixed:
                done.update(fixed)
            return done
        return {"messages": [SystemMessage(content=note)], "required_tools": needed, "missing_required_tools": missing,
                "content_issues": issues, "validation_retries": retries + 1, "retry": True}

    @staticmethod
    def _add_calculation_basis(state: AdvisorState, question: str, answer: str, issues: list, today: Optional[str]) -> Optional[dict]:
        """With no retries left, add the calculator's own summary sentences under the answer if that settles issues.

        The summaries state each saving with its kWh, hours and period. A live answer was sent back twice for quoting
        savings without the kWh they were calculated for (10.62 kWh of HVAC in 16:00-21:00, 61.63 kWh of EV charging)
        and never added them. Code adds a 'How these figures were calculated' section listing the summary of every
        calculation the answer quotes, then checks the amended answer again; the amendment is kept only when it leaves
        fewer issues and raises none, and it is recorded in the state as 'completed_by_code'.
        """
        basis = calculation_basis(state["messages"], answer)
        if not basis:
            return None
        amended = answer.rstrip() + "\n\n### How these figures were calculated\n" + "\n".join(f"- {s}" for s in basis)
        again = content_issues(question, state["messages"], amended, today)
        if len(again) >= len(issues) or not set(again) <= set(issues):
            return None
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=amended, id=last.id)], "content_issues": again,
                "completed_by_code": [f"added the summary of {len(basis)} calculation(s) under the answer, settling "
                                      f"{len(issues) - len(again)} issue(s)"]}

    @staticmethod
    def _after_validation(state: AdvisorState) -> str:
        return "agent" if state.get("retry") else END

    def _build_graph(self):
        g = StateGraph(AdvisorState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        g.add_node("validate", self._validate_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, ["tools", "validate"])
        g.add_edge("tools", "agent")
        g.add_conditional_edges("validate", self._after_validation, ["agent", END])
        return g.compile()

    # ---------------------------------------------------------------- public API

    @staticmethod
    def build_context(context: Optional[str] = None, now: Optional[datetime] = None) -> str:
        """Today's date and time, a seven-day calendar, plus whatever the caller supplied (location, preferences).

        The calendar names the weekday of each of the next seven dates. Models are unreliable at calendar
        arithmetic: given only today's date, the advisor once took "Wednesday" to be a Monday. With the dates
        listed, "Wednesday" is a lookup, not a calculation.
        """
        now = now or datetime.now()
        tomorrow = now + timedelta(days=1)
        week = ", ".join(f"{d:%A} {d:%Y-%m-%d}" for d in (now + timedelta(days=k) for k in range(1, 8)))
        lines = [f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A}).",
                 f"Tomorrow is {tomorrow:%Y-%m-%d} ({tomorrow:%A}).",
                 f"The next seven days: {week}."]
        if context:
            lines.append(context.strip())
        return "\n".join(lines)

    def invoke(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question about energy optimization.
            context (str): Extra context for this request, e.g. "Location: San Francisco, CA".
                Today's date and time are always added.

        Returns:
            dict: the final graph state. ``messages`` holds the whole exchange (the question,
            every tool call and tool result, any validation note, and the final answer as the last
            message); ``steps`` counts model turns; ``required_tools``, ``missing_required_tools`` and
            ``content_issues`` record the validation; ``attempts`` and ``endpoint_errors`` record any failed
            tries (type, message, where it was raised); ``error`` is set only if every attempt failed.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        state = {"messages": [("user", question)], "context": self.build_context(context), "question": question,
                 "steps": 0, "validation_retries": 0}
        failures = []
        for attempt in range(1, self.endpoint_attempts + 1):
            try:
                result = self.graph.invoke(dict(state), config={"recursion_limit": 2 * self.max_steps + 5})
                result["attempts"] = attempt
                result["endpoint_errors"] = failures
                return result
            except Exception as e:
                failures.append(describe_error(e))
                if not is_transient(e) or attempt == self.endpoint_attempts:
                    break
                time.sleep(self.retry_wait * attempt)
        last = failures[-1]
        return {"messages": [HumanMessage(content=question), AIMessage(content=f"Sorry, I could not complete this request: {last['type']}.")],
                "context": state["context"], "steps": 0, "required_tools": required_tools(question),
                "missing_required_tools": required_tools(question), "content_issues": [],
                "attempts": len(failures), "endpoint_errors": failures,
                "error": f"{last['type']}: {last['message']} (raised in {last['origin']}; {len(failures)} attempt(s))"}

    def get_agent_tools(self):
        """Names of the tools available to the Energy Advisor."""
        return [t.name for t in self.tools]
