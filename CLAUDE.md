# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent trip planner built on LangGraph. It researches a destination, finds
school-break-friendly travel dates from indexed school calendars, searches real flights via
Expedia (through a RapidAPI MCP server), builds a day-by-day itinerary, and produces a budget
breakdown — with a Gradio UI for running and refining a plan through follow-up requests, and
long-term memory of each traveler's preferences across sessions.

## Commands

```bash
# Install deps (incl. dev tools: pytest, ruff)
uv sync --extra dev

# One-time: build the FAISS school-calendar index (needed before calendar_keeper works)
uv run python -m smart_travel_agent.retrieval.build_index

# Run the app (Gradio server at http://127.0.0.1:7860)
uv run python -m smart_travel_agent.app

# Run all unit tests (fast, LLM mocked throughout)
uv run pytest

# Run a single test file / test
uv run pytest tests/test_agents.py
uv run pytest tests/test_agents.py::test_destination_researcher_returns_brief

# Lint
uv run ruff check .

# Regenerate the architecture diagram after changing graph.py's wiring
uv run python -c "from smart_travel_agent.graph import build_graph; open('docs/graph.png', 'wb').write(build_graph().get_graph().draw_mermaid_png())"
```

Evals (`evals/`) are separate from `tests/` and not run by pytest — they make real OpenAI/API
calls instead of mocking the LLM, so they're slower and cost money, but catch output/retrieval
quality regressions mocked unit tests can't:

```bash
uv run python -m evals.eval_retrieval          # real OpenAI embedding calls against the FAISS index
uv run python -m evals.eval_agent_outputs       # real model calls; checks deterministic properties
                                                  # (constraint satisfaction, format, interest coverage,
                                                  # no duplicated responsibilities between agents) —
                                                  # each case traces back to a real bug found manually
uv run python -m evals.eval_itinerary_quality   # LLM-as-judge for coherence/redundancy; the weakest
                                                  # signal by nature, so any "this repeats across days"
                                                  # claim is verified programmatically against the day
                                                  # numbers the judge cites rather than trusted outright
```

Required env vars (`.env`, copy from `.env.example`): `OPENAI_API_KEY`, `EXPEDIA_RAPIDAPI_KEY`.
Flight search also needs `npx` available (it shells out to `mcp-remote`).

## Architecture

### The graph (`graph.py`, `state.py`, `agents.py`)

Five agent nodes wired as a LangGraph `StateGraph` over `TravelPlanState` (a Pydantic model):

```
START → destination_researcher ─┐
START → calendar_keeper ────────┼→ activity_planner → budget_analyst → END
                     └→ tickets_scouter ───────────────↑
```

`destination_researcher` and `calendar_keeper` run in parallel off `START`. `activity_planner`
waits on both. `tickets_scouter` follows `calendar_keeper` alone (it needs the resolved date
range) and runs in the same superstep as `activity_planner`. `budget_analyst` joins
`activity_planner` and `tickets_scouter`.

Each node is a plain function `(TravelPlanState) -> dict` that returns a partial state update.
Every agent's LLM call goes through `_run_agent`/`_arun_agent` in `agents.py`, which wraps
`langchain.agents.create_agent`, captures the full message trajectory for the UI's Trace/Tools
panels, and (for `calendar_keeper`/`tickets_scouter`) times the node. `tickets_scouter` uses the
async variant because its MCP-backed Expedia tools are async-only; it wraps its own graph
invocation in `asyncio.run` since the rest of the graph runs synchronously.

The graph is compiled with an `InMemorySaver` checkpointer: state persists per `thread_id`
across invocations, which is what makes follow-up refinement ("actually make it 7 days")
possible as a partial update instead of resupplying every field. It does **not** yet implement
the backup-date retry loop from the original design doc — that needs `calendar_keeper`/
`tickets_scouter` to return structured primary/backup ranges and a resolved flag instead of
freeform text.

### Dirty-node tracking (`app.py`)

A follow-up message is parsed by an LLM into a structured `_FollowUpUpdate` (only fields the
user actually asked to change are set). `app.py` maps changed fields to affected nodes via two
hand-maintained tables:

- `_NODE_DEPENDS_ON`: which state fields each node reads directly.
- `_NODE_UPSTREAM`: which nodes feed each node's own inputs (e.g. `activity_planner` reads
  `destination_researcher`'s and `calendar_keeper`'s outputs, so if either is dirty,
  `activity_planner` cascades dirty too even if none of its own direct fields changed).

`_compute_dirty_nodes` combines both into the final dirty set, written to
`TravelPlanState.dirty_nodes`. Each node function checks `_should_skip` first and short-circuits
to `{}` (no update) if it isn't listed as dirty — so a follow-up only recomputes what's actually
affected, and untouched nodes keep their prior value from the checkpoint.

One explicit exception: if a follow-up gives dates directly (e.g. "try Feb 12-14 instead"),
`calendar_keeper` is removed from the dirty set even though `date_range` changed — an explicit
date override is trusted as-is rather than triggering a fresh school-calendar lookup that could
talk the agent back into different dates.

When touching this dependency logic, keep `_NODE_DEPENDS_ON`/`_NODE_UPSTREAM` in sync with what
each agent function in `agents.py` actually reads — they're not derived automatically.

### Long-term memory (`memory.py`)

Separate from graph-checkpoint state (which is per-`thread_id`/per-session): `memories.json` on
disk holds durable facts per traveler name, persisted across process restarts. After a run
completes, `remember()` extracts durable facts from the trip via an LLM call (with an explicit
prompt to skip anything trip-specific/one-off), then reconciles them against existing facts for
that user via a second LLM call (merge/update/drop on contradiction) before saving. Agents pull
this back in via `_known_preferences()` in `agents.py`, with instructions that the current
trip's explicit details always win over a conflicting remembered preference.

### Guardrails (`guardrails.py`)

Deterministic (regex/set-based, not another LLM call) checks layered on top of prompt
instructions, since a prompt asking the model to hedge is a probabilistic nudge, not a
guarantee. Currently one check: `check_confidence_calibration` flags absolute-certainty phrases
("guaranteed", "100%", "never", ...) in `destination_researcher`'s output and produces a hedged
rewrite, scoring PASS/WARN/FAIL by the fraction of sentences flagged. New guardrails should
follow the same pattern: auditable/testable in isolation, not another model call judging a
model's output.

### Flight search (`mcp_tools.py`)

`get_expedia_tools()` spins up a `MultiServerMCPClient` over stdio to `mcp-remote`, which proxies
to Expedia's flight-search API via RapidAPI. Two wrappers are applied to the raw MCP tools:

- `_wrap_search_flights`: `searchFlights`' raw response embeds Expedia's full page-rendering
  payload (megabytes of JSON) for a couple dozen offers; this strips it down to airline, route,
  stops, duration, times, and price, and filters to `MAX_STOPS` or fewer, before it ever reaches
  the LLM's context.
- `_wrap_with_retry`: one retry on `McpError` (the RapidAPI gateway occasionally 502s/504s), with
  exponential backoff. Kept to one retry (not the usual default) because each attempt can itself
  take ~60s+ to time out — better to fail fast into `tickets_scouter`'s graceful degradation than
  sit through a multi-minute retry loop.

`searchFlights` only ever returns one direction per call — `TICKETS_SYSTEM_PROMPT` in
`agents.py` explicitly instructs the agent to call it twice for a round trip (once per leg,
omitting `return_date` on both) and to never claim a bundled round-trip fare was found, since
summing two one-way prices isn't the same as an airline's actual round-trip ticket.

### School-calendar retrieval (`retrieval/`)

`build_index.py` embeds the Stratford calendar PDF (`data/Stratford_General-Information.pdf`)
and the scraped Harker calendar (`harker_scraper.py`) into a FAISS index at
`retrieval/faiss_index/` — a one-time step required before `calendar_keeper` can look up school
breaks (see Setup in README). `retriever.py`'s `search_school_calendar` tool does plain cosine
similarity over the pre-embedded chunks (only the query gets embedded at query time). Chunking
strategy differs by school: Stratford's calendar is one dense chunk, Harker's is split into many
small weekly chunks — `CALENDAR_SYSTEM_PROMPT` in `agents.py` instructs `calendar_keeper` to
query each school by name separately rather than with one generic query, since a generic query
silently returns only Harker results.

### Gradio UI (`app.py`)

`plan_trip` starts a brand-new `thread_id` per submission (an unrelated destination never
inherits another trip's trace/checkpoint history); `send_followup` reuses the existing
`thread_id` to continue the same graph checkpoint. Both are generator functions that `yield` UI
tuples as the graph streams (`stream_mode="updates"`), so each node's output appears as soon as
it completes rather than waiting for the whole graph to finish. The UI has four side panels
beyond the trip-plan output: Context (running conversation), Memory (long-term facts for the
named traveler), Tools (filtered tool-call log), and Trace (full raw LLM/tool message log,
tagged per agent).

## Testing conventions

`tests/` mocks the LLM entirely (via `unittest.mock.patch` on `create_agent`) and checks
surrounding logic — prompts built correctly, dirty-node tracking, output formatting — not
actual model/retrieval quality. `evals/` (see Commands above) is where model/retrieval quality
is actually checked, against real API calls, and is not part of the `pytest` run.
