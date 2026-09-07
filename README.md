# Smart Travel Agent

A multi-agent trip planner built on LangGraph: it researches a destination, finds
school-break-friendly travel dates from indexed school calendars, searches real
flights via Expedia, builds a day-by-day itinerary, and produces a budget breakdown —
with a Gradio UI for running and refining a plan through follow-up requests, and
long-term memory of each traveler's preferences across sessions.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Node.js (provides `npx`) — needed to run the Expedia flight-search integration, which
  goes through the `mcp-remote` MCP client
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [RapidAPI](https://rapidapi.com/) key subscribed to the Expedia flight-search API

## Setup

1. Install dependencies (including dev tools like `pytest`/`ruff`):

   ```bash
   uv sync --extra dev
   ```

2. Configure your API keys:

   ```bash
   cp .env.example .env
   ```

   Then fill in `.env`:

   ```
   OPENAI_API_KEY=sk-...
   EXPEDIA_RAPIDAPI_KEY=...
   ```

3. Build the school-calendar search index (one-time; needed before the calendar-keeper
   agent can look up school breaks). This embeds the Stratford calendar PDF in
   `src/smart_travel_agent/data/` and the scraped Harker calendar, and saves a FAISS
   index to `src/smart_travel_agent/retrieval/faiss_index/`:

   ```bash
   uv run python -m smart_travel_agent.retrieval.build_index
   ```

## Running the app

```bash
uv run python -m smart_travel_agent.app
```

This starts a local Gradio server (by default at `http://127.0.0.1:7860`) — open that
URL in your browser to plan a trip and refine it with follow-up requests.

## Running tests

```bash
uv run pytest
```
