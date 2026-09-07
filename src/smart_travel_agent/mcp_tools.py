import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError

from smart_travel_agent.config import get_expedia_rapidapi_key

EXPEDIA_RAPIDAPI_HOST = "expedia-api-expedia-data-scraper.p.rapidapi.com"

MAX_STOPS = 1
# One retry (2 attempts total), not 3 — the RapidAPI gateway in front of Expedia takes
# ~60s+ to itself time out on a bad request, so each extra attempt is a long wait for a
# failure that's rarely transient in practice; better to fail fast into the graceful
# degradation in agents.py's tickets_scouter than sit through a 3+ minute retry loop.
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2


def _expedia_mcp_config() -> dict:
    return {
        "expedia": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "mcp-remote",
                "https://mcp.rapidapi.com",
                "--header",
                f"x-api-host: {EXPEDIA_RAPIDAPI_HOST}",
                "--header",
                f"x-api-key: {get_expedia_rapidapi_key()}",
            ],
        }
    }


def _extract_flight_summaries(raw_result) -> list[dict]:
    """Pulls just the fields an agent needs (airline, route, stops, duration, times, price)
    out of searchFlights' raw response, which otherwise embeds Expedia's full page-rendering
    payload per listing (dialog modals, accessibility trees, sponsored placements, icons) —
    megabytes of JSON for what's really a couple dozen flight offers."""
    text = raw_result[0]["text"] if isinstance(raw_result, list) else raw_result
    data = json.loads(text)
    listings = data.get("data", {}).get("flightsSearch", {}).get("listingResult", {}).get("listings", [])

    summaries = []
    for listing in listings:
        content = listing.get("flightsShoppingOfferContent")
        price_display = listing.get("priceDisplay")
        if not content or not price_display:
            continue

        secondary = content.get("secondarySection", [])
        if not secondary or "stops" not in secondary[0]:
            continue

        try:
            summaries.append(
                {
                    "airline": secondary[1]["contents"][1]["text"],
                    "route": secondary[1]["contents"][0]["text"],
                    "stops": secondary[0]["stops"],
                    "duration": content["tertiarySection"][0]["contents"][0]["text"],
                    "departs": secondary[0]["start"]["primary"],
                    "arrives": secondary[0]["end"]["primary"],
                    "price": price_display["rows"][0]["elements"][0]["price"]["text"],
                }
            )
        except (KeyError, IndexError):
            continue

    return summaries


def _format_flight_summaries(summaries: list[dict]) -> str:
    if not summaries:
        return f"No flights found with {MAX_STOPS} or fewer stops."
    return "\n".join(
        f"{s['airline']} | {s['route']} | {s['duration']} | "
        f"departs {s['departs']} arrives {s['arrives']} | {s['price']}"
        for s in summaries
    )


def _wrap_search_flights(tool):
    """Wraps the tool's coroutine to filter its response before it re-enters the LLM's
    context. The tool's response_format is "content_and_artifact", so the raw coroutine
    returns a (content, artifact) tuple — that shape has to be preserved on the way out."""
    original_coroutine = tool.coroutine

    async def filtered_coroutine(*args, **kwargs):
        raw_result = await original_coroutine(*args, **kwargs)
        content, artifact = raw_result if isinstance(raw_result, tuple) else (raw_result, None)

        summaries = _extract_flight_summaries(content)
        summaries = [s for s in summaries if s["stops"] <= MAX_STOPS]
        filtered_text = _format_flight_summaries(summaries)

        return (filtered_text, artifact) if isinstance(raw_result, tuple) else filtered_text

    tool.coroutine = filtered_coroutine
    return tool


def _wrap_with_retry(tool):
    """Retries a tool call on transient MCP transport failures (e.g. a 504 from the RapidAPI
    gateway in front of Expedia) instead of letting one hiccup crash the whole graph run."""
    original_coroutine = tool.coroutine

    async def retrying_coroutine(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return await original_coroutine(*args, **kwargs)
            except McpError:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))

    tool.coroutine = retrying_coroutine
    return tool


async def get_expedia_tools() -> list:
    client = MultiServerMCPClient(_expedia_mcp_config())
    tools = await client.get_tools()
    for tool in tools:
        if tool.name == "searchFlights":
            _wrap_search_flights(tool)
        _wrap_with_retry(tool)
    return tools
