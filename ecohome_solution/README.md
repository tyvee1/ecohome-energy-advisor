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
├── 03_run_and_evaluate.ipynb      # Agent instructions, 13 test cases, evaluation and report
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

- **Schema** `AdvisorState`: `messages` (merged by the `add_messages` reducer), `context`, `question`, `steps`, and the validation record (`required_tools`, `missing_required_tools`, `content_issues`, `validation_retries`, and `completed_by_code` for anything code added to the answer).
- **Nodes**: `agent` calls `gpt-4o` (temperature 0) with the system prompt, the context and the conversation; `tools` is a `ToolNode` over the seven tools; `validate` checks an answer before it is accepted.
- **Edges**: `START → agent`; a conditional edge from `agent` to `tools` when the reply contains tool calls and to `validate` otherwise; `tools → agent`; a conditional edge from `validate` back to `agent` or to `END`.
- **Validation**: a model can answer from what it already knows and skip a data source the answer depends on. `DATA_RULES` in `agent.py` names, for each kind of question, the tools whose results the answer must rest on: a quantified saving needs `get_electricity_prices` and `calculate_energy_savings`; suggestions need `search_energy_tips` (with the source files cited); an answer based on usage history needs `query_energy_usage`; using more of your own solar needs `get_weather_forecast`, `query_solar_generation` and `query_energy_usage`; a payback or return-on-investment question needs the same tools as a saving; a plan for a named day (tomorrow, Wednesday, this week) needs `get_weather_forecast`; a question about prices or peak hours needs `get_electricity_prices`. Only tool results that came back without an error count. It also checks, in code, that the answer uses what came back: every cited tip file must be one that `search_energy_tips` returned in this conversation (any "source:" the answer names must be such a file), and an answer giving suggestions must cite at least one; an answer about using more of your own solar must quote a figure from the solar history (a forecast figure that happens to be close to a history figure does not count); an answer with a saving must quote a figure from `calculate_energy_savings`, that calculation must use the time-of-use price table (looked up by the tool, or rates that appear in it) rather than the usage history's older tariff, and the usage figure must cover the same span of time as the calculation's period (a daily figure with `period='day'`, never last week's total passed as one day's usage, which would multiply the saving by seven); a calculation that describes no change is sent back; an answer based on usage history must state the kWh behind its suggestions; and an answer written partly in a message that also called a tool (which the customer never sees) is sent back to be written whole. Any hourly price the answer gives must be that hour's rate in the price table: a rate copied from a neighbouring hour is sent back, and a price range given for a range of hours must be those hours' lowest and highest rate. Hour ranges are read the way the tools write them, ending where the last hour ends (the peak hours 16-20 are 16:00-21:00), so a peak period or a day's best solar hours written one hour short is sent back, as is a range that is not the hours a quoted saving was calculated for. Every dollar figure must be one a tool returned (the model does no money arithmetic of its own); a kWh figure stated for a date must belong to that date (a forecast day that was never retrieved cannot be given another day's figure); a cost must sit next to the kWh it was calculated from; a cut across the whole day at the same price cannot be presented as a peak-hours change or a move off-peak; and "tonight" needs today's prices, and a price range given for tonight's hours must be tonight's (today's rates before midnight, tomorrow's after). A single rate given for several hours must be their average, highest or lowest rate (or the price a calculation used for exactly those hours); a day's best solar hours must be that day's `best_solar_window` (each forecast day has its own); a temperature must be one the forecast or a retrieved tip gave, or a unit conversion of one, so a thermostat setting comes from the knowledge base; a section headed as estimated savings or cost must state one; a device said to be used mostly at peak must be; and an answer about using more of your own solar must not advise running the HVAC less while the panels produce. A move into the solar hours must be valued with the forecast's solar counted; a quoted saving must be for the hours the answer recommends, and a section quoting a saving must name the hours it was calculated for; a temperature given with a source file must be in that file; a cut of half or more of a device's use is not offered unless the question asks for it; a question about the washing machine and dryer is answered for those machines, not every appliance; and solar said to cover "a significant portion" of the home's use must. A kWh figure with decimals must be one a tool returned (the model does not add up energy itself), and a saving quoted for a move out of the midday solar hours must have been priced with the forecast's solar. A rate quoted for tonight cannot be one of this morning's (tonight's hours after midnight are tomorrow's); the off-peak period listed in parts must be the whole period ("00:00-06:00 and 22:00-23:00" leaves out 23:00-24:00); hours called the household's highest use cannot leave out a neighbouring hour used more (from the history's hour-of-day profile); and a suggestion's cited file must hold its advice when another returned file plainly does (a shared four-word phrase there, not even a two-word phrase in the cited file). A sentence about the forecast that names no day is read with the day its own line names. A kWh amount stated in the question must be the amount priced, a savings calculation that returned an error must be made again rather than reported as impossible, and a calculation that compares hours with themselves cannot be offered as a finding. Any answer that uses retrieved tips cites their files, and an answer about using more of your own solar sets the forecast against the household's daily use. If anything is missing, `validate` sends the model back with a note saying what and why, at most twice. When no retries are left, code tries one deterministic completion: it adds a "How these figures were calculated" section listing the summary sentence of every calculation the answer quotes (each states its kWh, hours and period), keeps it only if the amended answer has fewer issues and no new ones, and records that in `completed_by_code`; the test loop prints it under the answer. The example answer at the top of notebook 03 prints its validation record the same way. Anything still unresolved is recorded in `missing_required_tools` and `content_issues`.

`Agent(instructions, model="gpt-4o")` and `Agent.invoke(question, context)` keep the starter's contract; the notebook creates the agent exactly as the starter does, `Agent(instructions=ECOHOME_SYSTEM_PROMPT)`. The default model is `gpt-4o` rather than the starter's `gpt-4o-mini`: in evaluation runs gpt-4o-mini kept missing layered rules (each day's own solar window, which hours a saving was calculated for, which file a figure came from) even after being sent back with the exact fault. Any model the endpoint serves can be passed as `model`. `invoke` returns the final graph state; its `messages` hold the question, every tool call and result, and the answer.

- **Context awareness**: every request is given the current date, the day of the week, tomorrow's date and the next seven dates with their weekday names, so "tomorrow" and "Wednesday" resolve to real dates, plus the context the caller passes (the location).
- **Error handling**: tool exceptions come back to the model as error messages it can correct; tools return `{"error": ...}` for bad arguments; model turns are capped at 12 per question; a failure to reach the model returns a readable message with `error` set instead of raising.

### Evaluation (`03_run_and_evaluate.ipynb`)

- **13 test cases** across EV charging, thermostat settings, appliance scheduling, solar maximisation, cost savings and usage history. Each lists the tools it requires, the extra tools that are reasonable, and what a good answer contains.
- **`test_results`** logs, for every test, the full message history, the tools called, a decision log (each model turn's chosen tools and arguments, each tool result and whether it was an error), the time taken and the tokens used.
- **`evaluate_response`** scores ACCURACY, RELEVANCE, COMPLETENESS and USEFULNESS from 1 to 5 with feedback for each. The judge is `gpt-4o`, the same model as the advisor, and it sees the tool results so it can check the answer's figures. Because a model can favour answers like its own, everything code can decide is decided by code checks instead, and a data error caps the judge's accuracy score. The judge is also given the list of figures that code found in the complete tool results, so it cannot call them unsupported. Two code checks run on the complete tool results: one lists any amount, kWh or percentage in the answer that does not appear in them; the other finds a saving stated for the wrong period (a daily saving called yearly). A third code check lists any cited knowledge-base file that no search returned. A fourth reruns the advisor's own data checks (`data_errors` in `agent.py`) independently of what the advisor reported: a saving computed from usage for a different span of time than its period, an hourly price or price range put against the wrong hours, a dollar figure no tool returned, a day's kWh given to another day, an hour range one hour short or not the hours that were priced, a saving described as a different change than the one calculated, "tonight" quoted from another day's prices, and the newer checks listed under validation above (tonight's price range, one rate for several hours, each day's solar window, temperatures, empty savings sections, peak-share claims, cutting HVAC in the solar hours, a saving for other hours than recommended, a suggestion without its own retrieved tip). The figure check judges a temperature by the advisor's rule: found when a tool gave it or a unit conversion of it, so 20°C beside a tip's 68°F is not reported, and a humidity of 78 does not make 78°F found. A period error, an invented citation or a data error caps ACCURACY at 2 out of 5, whatever the judge said, and the feedback records why.
- **`evaluate_tool_usage`** measures tool appropriateness (share of calls that went to required or reasonable tools) and tool completeness (share of required tools called), and reports tool errors and repeated calls, with feedback. It also returns `required_calls_ok`, a pass/fail check that every expected tool was called and returned data, kept apart from the scores, and `unused_tools`, the expected tools whose results the answer does not reflect (no figure from them, or for the knowledge base no file it returned cited). Only figures that describe data count (kWh, USD, rates, generation, temperatures), not counts, hours or dates: an answer saying 78°F does not count as using a result that held 78 records.
- **`generate_evaluation_report`** computes the overall and per-metric scores, scores by scenario and efficiency (time, model turns, tokens) in code. It then sends that evidence, with every test's metrics, judge feedback, missing calls, tool errors and code-check flags, to `gpt-4o`, which writes the strengths, weaknesses and recommendations, each tied to test ids. Two safeguards stay in code: a required-call failure the model leaves out of its weaknesses is added and marked, and if the model cannot be reached the report says so and lists findings computed in code. The summary reports three counts separately: tests whose score passed, tests whose required calls were complete, and tests that passed both, so a skipped data step cannot hide behind a good score. **`display_evaluation_report`** prints it with a per-test table; a chart shows the scores and the savings the advisor calculated.
- **Required tool calls** (section 7): a final cell lists every test's expected, missing and failed calls, and stops the notebook with an assertion if any test is missing one.

## Checking the rubric items

| Rubric item | Where to look |
| --- | --- |
| Database contains EnergyUsage and SolarGeneration | `01_db_setup.ipynb` section 6b. The model classes `EnergyUsage` and `SolarGeneration` are stored as the tables `energy_usage` and `solar_generation`. To check the file directly: `sqlite3 data/energy_data.db ".tables"` lists `energy_usage` and `solar_generation`. |
| Vector DB populated | `02_rag_setup.ipynb` sections 2 to 4 (documents loaded, chunks per document, vectors stored) |
| `get_weather_forecast`, `get_electricity_prices`, `search_energy_tips` | `tools.py` |
| LLM and graph (schema, nodes, edges) | `agent.py`: `AdvisorState`, `_agent_node`, `ToolNode`, `_validate_node`, `_build_graph` |
| `ECOHOME_SYSTEM_PROMPT` | `03_run_and_evaluate.ipynb` section 1 |
| At least 10 test cases | `03_run_and_evaluate.ipynb` section 2 (12 cases) |
| Decisions and tool usage logged in `test_results` | `03_run_and_evaluate.ipynb` section 3 |
| `evaluate_response`, `evaluate_tool_usage`, `generate_evaluation_report`, display | `03_run_and_evaluate.ipynb` sections 4 and 5 |
| Every scenario makes the calls it depends on (end-to-end workflow) | `03_run_and_evaluate.ipynb` section 3 (each test prints its validation line) and section 7 (required-call table and assertion) |
| Report findings and recommendations generated by a model from the metrics | `03_run_and_evaluate.ipynb` section 5: FINDINGS, then REQUIRED-CALL VALIDATION and CODE CHECKS |

## Changes from the starter, and why

- `get_weather_forecast` and `get_electricity_prices` implemented (they returned nothing).
- `search_energy_tips` now indexes every document in the folder and rebuilds when documents change; the starter listed two files by name and never rebuilt an existing index, so added documents were ignored.
- File paths are resolved from `tools.py`'s own folder, so the tools find the database wherever Python is started.
- `query_energy_usage` and `query_solar_generation` return summaries instead of every hourly row. One week of usage was about 82,000 characters, 30 days about 354,000, most of the model's context window. Raw rows are available with `include_records=True`, capped.
- The device filter accepts `ev`, `electric car`, `appliances` and similar; the starter matched `EV`, `HVAC` and `appliance` exactly and returned nothing otherwise. `dishwasher`, `washing machine` or `dryer` returns that machine's records only.
- The forecast's daily entries carry the weekday and `best_solar_window`, and the price summary carries `peak_window` and `off_peak_window`, as clock ranges that end where the last hour ends (hours 16-20 are "16:00-21:00"), so the model copies a range instead of writing one. Each forecast day also carries `solar_kwh_in_best_window`, the forecast's solar in that window, so the model quotes it instead of adding hourly figures (a run added them to 15.24 kWh; the forecast gives 17.24).
- `get_recent_energy_summary` reports the weather actually recorded; the starter always reported "sunny".
- `calculate_energy_savings` can value a time shift (same kWh at cheaper hours, with the rates looked up for the date) and annualises by the stated period; the starter could only value using fewer kWh and always multiplied by 365. It has no default price: without a price or a date it returns an error instead of assuming $0.12. Given a device the history records, a date, and either `reduction_percent` or the hours to shift, it reads the device's average daily kWh from the last 7 complete days of history itself and prices it with that date's time-of-use rates, so the model no longer copies kWh figures by hand (in earlier runs it passed a week's total as one day's use, or a day's figure as a week's). A dishwasher, washing machine or dryer means that machine's own records, not every appliance. It refuses inputs that produced wrong savings in earlier runs: prices passed alongside a date (they would replace that date's rates), and a recorded device's kWh passed for a recurring period (a single stated amount with `period='one_time'` is still accepted). With `current_hours` and `reduction_percent` it values a cut in those hours only, priced at those hours' rates. A cut and a shift are separate calculations (combining them is refused). A shift values moving use out of dearer hours, so a shift into dearer hours (for example from off-peak to the peak) is refused with a note on which hours go where. A recorded device's own daily use typed in as a "stated amount" is refused too: pricing it in one set of hours compares nothing, so the tool points to a shift instead. Energy in the solar hours is valued with the panels: given `location`, the tool reads that date's forecast and counts its estimated solar as free, up to the kWh used, and prices only the rest at grid rates. That applies on both sides of a move: in the hours the use moves into, and in the core solar hours (09:00-16:00, when the panels give at least half their noon output) that it leaves, so moving EV charging out of midday to off-peak is not reported as a saving when the panels were covering much of it. A call whose hours touch 09:00-16:00 without `location` is refused. The assumption (the panels' output in those hours goes to this use first) is stated in the result (`solar_kwh_now`, `solar_kwh_after`, `solar_from`), so a move from the evening peak into the solar hours is compared fairly with a move to off-peak, and a move into hours that cost more once solar is counted is refused. A daily habit read from the history recurs on weekdays and at weekends, whose tariffs differ (there is no peak at weekends), so the yearly figure of such a shift prices each of the 7 days from today and scales that week to 365 days (`annual_basis`, with `weekday_savings_usd` and `weekend_savings_usd`); `savings_usd` stays the saving on the date asked. Without this, the same EV move was worth $1035.13 a year on a Saturday's rates and $3249.68 on a Friday's in one run, and a payback depended on the day the model picked. A calculation that counts solar keeps one day's figure times 365, because the forecast's solar differs from day to day. Hours listed in the order they run past midnight (`[22, 23, 0, 1, 2, 3, 4, 5]` on today's date) are tonight's: the hours after midnight are priced at the next day's rates, and the summary says so (a run priced "22:00 tonight to 06:00 tomorrow" with this morning's rates, $4.50 instead of $4.33). Given `upfront_cost_usd` it returns the payback period (`payback_years`) and the first-year return on investment (`first_year_roi_percent`), which the `roi_smart_thermostat` test case exercises. Its summary sentence names the kWh, the hours and each cost with its period, and a change that costs more is worded as costing more.
- `01_db_setup.ipynb` starts from empty tables (running it twice used to double every record), seeds the random generator, and generates data up to the current hour instead of stopping at midnight.
- `agent.py` builds the graph explicitly instead of using `create_react_agent`.

## Limitations

- Weather is simulated unless live weather is switched on; prices are always simulated in this project.
- The sample usage and solar data are generated, not measured. The starter's generator has the EV drawing about 10 kWh in every hour of the day, so EV totals (over 200 kWh a day) and any EV savings are far larger than a real car's; the advisor's arithmetic is right for this data, not for a real household.
- The starter's generator picks the appliance name (dishwasher, washing machine or dryer) at random for every hourly record, so each machine shows about 1.5 kWh in every hour of the day; per-machine figures follow the data, not a real appliance.
- The LLM judge is itself a model: its scores are an informed opinion, and runs differ slightly.

## License

This project is for educational purposes as part of the Udacity Course 2 curriculum.
