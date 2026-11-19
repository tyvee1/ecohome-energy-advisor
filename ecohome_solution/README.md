# EcoHome Energy Advisor

An AI agent that helps EcoHome customers with solar panels, an electric vehicle, a smart thermostat and household appliances lower their electricity cost. It reads weather forecasts, time-of-use prices, the home's own usage and solar history, and a knowledge base of energy-saving articles, then recommends when to run devices and what that saves.

Author: Ty

## Project structure

```
ecohome_solution/
├── models/
│   ├── __init__.py
│   └── energy.py                  # Database models (unchanged from the starter)
├── data/
│   ├── documents/                 # Knowledge base: 2 starter documents + 6 added
│   ├── energy_data.db             # SQLite database, created by 01_db_setup.ipynb
│   └── vectorstore/               # Chroma vector store, created by 02_rag_setup.ipynb
├── agent.py                       # The Energy Advisor: an explicit LangGraph graph
├── tools.py                       # Seven tools: weather, prices, usage, solar, recent summary, tips search, savings
├── requirements.txt               # Pinned packages
├── .env.example                   # Template for the API key
├── 01_db_setup.ipynb              # Database setup, sample data, usage charts
├── 02_rag_setup.ipynb             # Knowledge base setup and search tests
├── 03_run_and_evaluate.ipynb      # Agent instructions, 12 test cases, evaluation and report
└── README.md
```

## Setup

Python 3.12 was used (the course supports 3.11 to 3.13).

```bash
cd ecohome_solution
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your Vocareum key in .env
```

Packages added to the course's list: `matplotlib==3.11.2`, for the charts. Everything else is the course's pinned set.

Run the notebooks in order, from the `ecohome_solution` folder: `01_db_setup.ipynb`, `02_rag_setup.ipynb`, `03_run_and_evaluate.ipynb`. The notebooks in this repository are saved with the outputs of that run.

## How it works

### Tools (`tools.py`)

| Tool | What it returns | Data |
| --- | --- | --- |
| `get_weather_forecast(location, days)` | hourly temperature, condition, solar irradiance and estimated solar kWh, plus a daily summary | simulated (live from Open-Meteo when `ECOHOME_LIVE_WEATHER=1`) |
| `get_electricity_prices(date)` | 24 hourly time-of-use rates with period and demand charge, and the cheapest and most expensive hours | simulated |
| `query_energy_usage(start_date, end_date, device_type)` | totals, per-device breakdown, per-day table and hour-of-day profile | SQLite |
| `query_solar_generation(start_date, end_date)` | totals, per-day generation with weather, hour-of-day profile | SQLite |
| `get_recent_energy_summary(hours)` | recent use by device and recent generation | SQLite |
| `search_energy_tips(query, max_results)` | the most relevant knowledge-base passages with source and relevance score | Chroma |
| `calculate_energy_savings(...)` | savings from using less energy and/or shifting it to cheaper hours (given a date and the hours, it looks up the time-of-use rates itself), per period and per year, with a one-sentence summary that names each period | calculation |

**Weather and prices are simulated, deterministically.** Each value is a base curve multiplied by weights drawn from a random generator seeded with the location and date, so the same question on the same day returns the same data. Off-peak is 22:00 to 06:00 every day; peak is 16:00 to 21:00 on weekdays; the rest is mid-peak. Setting `ECOHOME_LIVE_WEATHER=1` switches the forecast to the free Open-Meteo API (no key needed), falling back to the simulation if it cannot be reached.

**Knowledge base.** `search_energy_tips` indexes every `.txt` file in `data/documents/`. It keeps a manifest of the documents' hashes and rebuilds the index when a document is added, removed or edited. Six documents were added to the two from the starter:

| Document | Topic required by the instructions |
| --- | --- |
| `tip_hvac_optimization.txt` | HVAC optimization strategies |
| `tip_smart_home_automation.txt` | Smart home automation tips |
| `tip_renewable_energy_integration.txt` | Renewable energy integration |
| `tip_seasonal_energy_management.txt` | Seasonal energy management |
| `tip_energy_storage_optimization.txt` | Energy storage optimization |
| `tip_ev_charging_and_time_of_use.txt` | (additional) EV charging and time-of-use pricing |

### Agent (`agent.py`)

The agent is a LangGraph `StateGraph` defined explicitly:

- **Schema** `AdvisorState`: `messages` (merged by the `add_messages` reducer), `context` and `steps`.
- **Nodes**: `agent` calls `gpt-4o-mini` (temperature 0) with the system prompt, the context and the conversation; `tools` is a `ToolNode` over the seven tools.
- **Edges**: `START → agent`; a conditional edge from `agent` to `tools` when the reply contains tool calls and to `END` otherwise; `tools → agent`.

`Agent(instructions, model="gpt-4o-mini")` and `Agent.invoke(question, context)` keep the starter's contract. `invoke` returns the final graph state; its `messages` hold the question, every tool call and result, and the answer.

- **Context awareness**: every request is given the current date, the day of the week, tomorrow's date and the next seven dates with their weekday names, so "tomorrow" and "Wednesday" resolve to real dates, plus the context the caller passes (the location).
- **Error handling**: tool exceptions come back to the model as error messages it can correct; tools return `{"error": ...}` for bad arguments; model turns are capped at 12 per question; a failure to reach the model returns a readable message with `error` set instead of raising.

### Evaluation (`03_run_and_evaluate.ipynb`)

- **12 test cases** across EV charging, thermostat settings, appliance scheduling, solar maximisation, cost savings and usage history. Each lists the tools it requires, the extra tools that are reasonable, and what a good answer contains.
- **`test_results`** logs, for every test, the full message history, the tools called, a decision log (each model turn's chosen tools and arguments, each tool result and whether it was an error), the time taken and the tokens used.
- **`evaluate_response`** scores ACCURACY, RELEVANCE, COMPLETENESS and USEFULNESS from 1 to 5 with feedback for each. The judge is `gpt-4o`, a stronger model than the advisor, and it sees the tool results so it can check the answer's figures. Two code checks run on the complete tool results: one lists any amount, kWh or percentage in the answer that does not appear in them; the other finds a saving stated for the wrong period (a daily saving called yearly). A period error caps ACCURACY at 2 out of 5, whatever the judge said, and the feedback records why.
- **`evaluate_tool_usage`** measures tool appropriateness (share of calls that went to required or reasonable tools) and tool completeness (share of required tools called), and reports tool errors and repeated calls, with feedback.
- **`generate_evaluation_report`** computes overall and per-metric scores, scores by scenario, efficiency (time, model turns, tokens), strengths, weaknesses and recommendations, all derived from the scores. **`display_evaluation_report`** prints it with a per-test table; a chart shows the scores and the savings the advisor calculated.

## Checking the rubric items

| Rubric item | Where to look |
| --- | --- |
| Database contains EnergyUsage and SolarGeneration | `01_db_setup.ipynb` section 6b. The model classes `EnergyUsage` and `SolarGeneration` are stored as the tables `energy_usage` and `solar_generation`. To check the file directly: `sqlite3 data/energy_data.db ".tables"` lists `energy_usage` and `solar_generation`. |
| Vector DB populated | `02_rag_setup.ipynb` sections 2 to 4 (documents loaded, chunks per document, vectors stored) |
| `get_weather_forecast`, `get_electricity_prices`, `search_energy_tips` | `tools.py` |
| LLM and graph (schema, nodes, edges) | `agent.py`: `AdvisorState`, `_agent_node`, `ToolNode`, `_build_graph` |
| `ECOHOME_SYSTEM_PROMPT` | `03_run_and_evaluate.ipynb` section 1 |
| At least 10 test cases | `03_run_and_evaluate.ipynb` section 2 (12 cases) |
| Decisions and tool usage logged in `test_results` | `03_run_and_evaluate.ipynb` section 3 |
| `evaluate_response`, `evaluate_tool_usage`, `generate_evaluation_report`, display | `03_run_and_evaluate.ipynb` sections 4 and 5 |

## Changes from the starter, and why

- `get_weather_forecast` and `get_electricity_prices` implemented (they returned nothing).
- `search_energy_tips` now indexes every document in the folder and rebuilds when documents change; the starter listed two files by name and never rebuilt an existing index, so added documents were ignored.
- File paths are resolved from `tools.py`'s own folder, so the tools find the database wherever Python is started.
- `query_energy_usage` and `query_solar_generation` return summaries instead of every hourly row. One week of usage was about 82,000 characters, 30 days about 354,000, most of the model's context window. Raw rows are available with `include_records=True`, capped.
- The device filter accepts `ev`, `electric car`, `appliances` and similar; the starter matched `EV`, `HVAC` and `appliance` exactly and returned nothing otherwise.
- `get_recent_energy_summary` reports the weather actually recorded; the starter always reported "sunny".
- `calculate_energy_savings` can value a time shift (same kWh at cheaper hours, with the rates looked up for the date) and annualises by the stated period; the starter could only value using fewer kWh and always multiplied by 365.
- `01_db_setup.ipynb` starts from empty tables (running it twice used to double every record), seeds the random generator, and generates data up to the current hour instead of stopping at midnight.
- `agent.py` builds the graph explicitly instead of using `create_react_agent`.

## What a manual check of the recorded run found

The judge passed all 12 tests. Every figure in the recorded answers was then checked by hand against the tool data, and four answers have errors the judge did not catch:

- **Overnight charging (test 2)** prices the charge at $0.10 per kWh, the average cost in the usage history (billed on an older flat tariff), instead of the time-of-use rates for those hours ($0.111 to $0.114). At those rates its 40 kWh plan costs $4.52, not $3.21. It also takes each hour's charge from the history's hourly averages instead of from a charger's power.
- **Thermostat on Wednesday (test 3)** reads the temperature one hour off (30.4 °C is the 14:00 forecast; 15:00 is 30.6 °C) and links to an ecohome.com page that no tool or document provided. The judge's remark that the forecast shows 22 °C is itself wrong: it was shown a shortened forecast, and temperatures are not covered by the code figure check.
- **Dishwasher savings (test 5)** costs the whole appliance category (dishwasher, washing machine and dryer together, about 31 kWh a day in the sample data) and presents the result as the dishwasher's saving ($1,629.51 a year). A dishwasher cycle uses roughly 1 to 2 kWh.
- **Three reductions (test 10)** uses $0.874 and $0.367 as prices per kWh. Neither is a per-kWh price in this data (the time-of-use rates run from about $0.11 to $0.26, and the history averages about $0.11), so its $5.07 and $2.53 savings are not supported.

What this shows: an LLM judge passes fluent answers with plausible numbers. The code checks found no saving stated for the wrong period in this run; these four answers point to the next checks worth adding: units (cost per hour against cost per kWh), temperatures in the figure check, and links that must come from a tool or a document.

## Limitations

- Weather is simulated unless live weather is switched on; prices are always simulated in this project.
- The sample usage and solar data are generated, not measured. The starter's generator has the EV drawing about 10 kWh in every hour of the day, so EV totals (over 200 kWh a day) and any EV savings are far larger than a real car's; the advisor's arithmetic is right for this data, not for a real household.
- The LLM judge is itself a model: its scores are an informed opinion, and runs differ slightly.

## License

This project is for educational purposes as part of the Udacity Course 2 curriculum.
