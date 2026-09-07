import asyncio
import time

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from smart_travel_agent.config import get_llm
from smart_travel_agent.mcp_tools import get_expedia_tools
from smart_travel_agent.memory import get_memories
from smart_travel_agent.retrieval.retriever import search_school_calendar
from smart_travel_agent.state import TravelPlanState


def _known_preferences(state: TravelPlanState) -> str:
    memories = get_memories(state.user_name)
    if not memories:
        return ""
    return "\nKnown preferences for this traveler (from past trips): " + " ".join(memories) + "\n"

RESEARCHER_SYSTEM_PROMPT = (
    "You are an expert travel journalist who has visited 100+ countries. "
    "You know the best hidden gems and practical tips for any destination."
)

PLANNER_SYSTEM_PROMPT = (
    "You are a luxury travel consultant with 15 years of experience crafting "
    "personalized day-by-day itineraries."
)

CALENDAR_SYSTEM_PROMPT = (
    "You are the family's calendar keeper. The family has kids at multiple schools "
    "(currently Stratford and Harker Upper School), each with their own calendar in the "
    "search_school_calendar tool's index. You must call the tool once per school, by name — "
    "e.g. first query \"Stratford <break>\", then separately query \"Harker Upper School "
    "<break>\". Do not rely on a single generic query like \"breaks\" or \"holidays\": Stratford's "
    "calendar is one dense chunk and Harker's is split into many small weekly chunks, so a "
    "generic query will return only Harker results and silently miss Stratford entirely. "
    "Once you have both schools' relevant dates, propose a specific date range that falls "
    "within a break at every school the family needs to accommodate. If no single range "
    "overlaps across all schools, say so explicitly and propose the closest alternative, "
    "noting which school it conflicts with."
)


def _serialize_message(message) -> dict:
    """Turns a LangChain message into a plain dict for the UI's Trace/Tools panels."""
    entry = {
        "role": message.type,
        "content": message.content if isinstance(message.content, str) else str(message.content),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]
    name = getattr(message, "name", None)
    if name:
        entry["name"] = name
    return entry


def _run_agent(
    system_prompt: str, user_prompt: str, tools: list | None = None, model: str | None = None
) -> tuple[str, list[dict]]:
    """Runs a create_agent loop, printing (for the terminal) and capturing (for the UI's Trace/
    Tools panels) the full message trajectory."""
    llm = get_llm(model=model) if model else get_llm()
    agent = create_agent(llm, tools=tools or [], system_prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    for message in result["messages"]:
        message.pretty_print()
    trace = [_serialize_message(m) for m in result["messages"]]
    return result["messages"][-1].content, trace


async def _arun_agent(
    system_prompt: str, user_prompt: str, tools: list | None = None, model: str | None = None
) -> tuple[str, list[dict]]:
    """Async counterpart to _run_agent — required for MCP-backed tools, whose langchain_mcp_adapters
    wrappers only implement async invocation (the underlying MCP client session is async-only), so
    agent.invoke() fails once a tool call is actually dispatched."""
    llm = get_llm(model=model) if model else get_llm()
    agent = create_agent(llm, tools=tools or [], system_prompt=system_prompt)
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_prompt)]})
    for message in result["messages"]:
        message.pretty_print()
    trace = [_serialize_message(m) for m in result["messages"]]
    return result["messages"][-1].content, trace


def _tag(trace: list[dict], node_name: str) -> list[dict]:
    for entry in trace:
        entry["agent"] = node_name
    return trace


def _tag_with_timing(trace: list[dict], node_name: str, elapsed: float) -> list[dict]:
    """Like _tag, but also appends a synthetic trace entry recording how long this node's
    own work (LLM calls, tool calls, MCP setup) took — visible in the UI's Trace tab, so
    slow nodes (e.g. an Expedia MCP round-trip) are easy to spot without digging into logs."""
    tagged = _tag(trace, node_name)
    print(f"[{node_name}] took {elapsed:.2f}s")
    tagged.append({"role": "timing", "agent": node_name, "content": f"⏱ took {elapsed:.2f}s"})
    return tagged


def _should_skip(state: TravelPlanState, node_name: str) -> bool:
    """True when a follow-up named specific nodes to recompute and this one wasn't among them —
    the caller (app.py's send_followup) already determined this node's inputs are unaffected."""
    return bool(state.dirty_nodes) and node_name not in state.dirty_nodes


def destination_researcher(state: TravelPlanState) -> dict:
    """Role: Destination Researcher — gathers destination info (attractions, food, neighborhoods, practical tips)."""
    if _should_skip(state, "destination_researcher"):
        return {}
    break_line = (
        f"This trip is planned around {state.preferred_break}, so evaluate best time to visit "
        f"against those specific dates rather than in general.\n"
        if state.preferred_break
        else ""
    )
    prompt = (
        f"Research {state.destination}, for a traveler coming from {state.origin}.\n"
        f"Traveler interests: {state.interests or 'general sightseeing'}.\n"
        f"{break_line}"
        f"{_known_preferences(state)}"
        "Cover: best time to visit, neighborhoods to stay in, must-see attractions, "
        "local food scene, transportation tips, cultural customs to know, and any special "
        "visa or entry requirements for a traveler from the origin above (assume their "
        "passport matches their country of origin unless told otherwise) — note whether a "
        "visa, eVisa, visa-on-arrival, or entry authorization (e.g. ESTA/ETA) is needed, and "
        "flag that requirements can change, so the traveler should confirm with an official "
        "government source before booking. Known preferences are from past trips — if any of "
        "them conflict with this trip's own details above (destination, interests, or travel "
        "window), this trip's details win. Do not include budget considerations, cost "
        "estimates, or price ranges anywhere in this brief — a separate budget analyst "
        "handles all of that."
    )
    content, trace = _run_agent(RESEARCHER_SYSTEM_PROMPT, prompt)
    return {"destination_brief": content, "trace": _tag(trace, "destination_researcher")}


def calendar_keeper(state: TravelPlanState) -> dict:
    """Role: Calendar Keeper — suggests dates that work for everyone in the family, based on school calendars."""
    if _should_skip(state, "calendar_keeper"):
        return {}
    break_constraint = (
        f"It must fall within {state.preferred_break}."
        if state.preferred_break
        else "Find whichever school break fits best."
    )
    prompt = (
        f"We want to take a {state.num_days}-day trip to {state.destination}. "
        f"{break_constraint} Propose a specific start and end date."
    )
    start = time.perf_counter()
    content, trace = _run_agent(
        CALENDAR_SYSTEM_PROMPT,
        prompt,
        tools=[search_school_calendar],
    )
    elapsed = time.perf_counter() - start
    return {"date_range": content, "trace": _tag_with_timing(trace, "calendar_keeper", elapsed)}


TICKETS_SYSTEM_PROMPT = (
    "You are a flight ticket scouter. Use the available Expedia tools to search for flights "
    "for the given number of passengers within budget, preferring options with fewer "
    "layovers/hops. Report the best options you find, including price and number of stops."
)


async def _scout_tickets(state: TravelPlanState) -> tuple[str, list[dict]]:
    tools = await get_expedia_tools()
    prompt = (
        f"Find flight tickets from {state.origin} to {state.destination} for "
        f"{state.passengers} passenger(s), during {state.date_range or 'dates to be determined'}, "
        f"within a total budget of ${state.budget}. Prefer options with fewer hops/layovers."
    )
    return await _arun_agent(TICKETS_SYSTEM_PROMPT, prompt, tools=tools, model="gpt-5.4")


def tickets_scouter(state: TravelPlanState) -> dict:
    """Role: Tickets Scouter — finds tickets for the given number of passengers, preferring fewer hops."""
    if _should_skip(state, "tickets_scouter"):
        return {}
    start = time.perf_counter()
    try:
        content, trace = asyncio.run(_scout_tickets(state))
    except Exception as e:
        # The Expedia MCP tool goes through a third-party RapidAPI gateway that occasionally
        # 502s/504s outright (see mcp_tools.py's retry wrapper — this is what's left once
        # those retries are exhausted). Degrade gracefully instead of crashing the whole graph
        # run: the rest of the trip plan (research, dates, itinerary, budget estimate) is still
        # useful even without a confirmed flight price.
        content = (
            "Flight search is temporarily unavailable — the flight search service returned "
            f"an error ({e}). Try sending a follow-up in a bit to retry."
        )
        trace = [{"role": "error", "content": str(e)}]
    elapsed = time.perf_counter() - start
    return {"tickets": {"summary": content}, "trace": _tag_with_timing(trace, "tickets_scouter", elapsed)}


def activity_planner(state: TravelPlanState) -> dict:
    """Role: Activity Planner — creates a day-by-day itinerary, adaptable to preferences and weather."""
    if _should_skip(state, "activity_planner"):
        return {}
    prompt = (
        f"Create a {state.num_days}-day itinerary for {state.destination}.\n"
        f"Travel dates: {state.date_range or 'not yet finalized'}.\n"
        f"Budget: ${state.budget} total. Interests: {state.interests or 'general sightseeing'}.\n"
        f"{_known_preferences(state)}"
        f"Destination research to build on:\n{state.destination_brief or '(none yet)'}\n\n"
        "Include morning/afternoon/evening activities, specific restaurant recommendations, "
        "and travel time between locations. Account for the season/weather implied by the "
        "travel dates (e.g. avoid outdoor-heavy days if it's a rainy/cold season). Make it "
        "achievable and enjoyable. Use the budget only to keep suggestions appropriately "
        "priced — do not include a budget breakdown, cost estimates, or a 'Budget Overview' "
        "section; a separate budget analyst handles that."
    )
    content, trace = _run_agent(PLANNER_SYSTEM_PROMPT, prompt)
    return {"itinerary": content, "trace": _tag(trace, "activity_planner")}


BUDGET_SYSTEM_PROMPT = (
    "You are a travel budget analyst. Given the ticket prices already found and the planned "
    "itinerary, produce an itemized budget breakdown (flights, lodging, food, activities, local "
    "transportation) with daily averages, and flag clearly whether the trip fits the stated "
    "total budget. Use the real ticket price if one was found instead of estimating it.\n\n"
    "For every category with a daily average: compute the average by dividing that category's "
    "own itemized total by the number of days — never state a daily average independently of "
    "the itemized figures for that same category. If you show per-day or per-meal line items, "
    "they must sum to the category total you then divide. Before finalizing, verify every "
    "number is consistent with every other number derived from it (itemized lines, category "
    "totals, daily averages, and the grand total must all agree with each other and with the "
    "stated number of passengers/days) — do not state two different figures for the same "
    "quantity anywhere in the response."
)


def budget_analyst(state: TravelPlanState) -> dict:
    """Role: Budget Analyst — estimates costs for tickets, lodging, food, and activities within budget."""
    if _should_skip(state, "budget_analyst"):
        return {}
    prompt = (
        f"Total budget: ${state.budget} for {state.passengers} passenger(s), "
        f"{state.num_days} days in {state.destination}.\n\n"
        f"Ticket search results:\n{state.tickets.get('summary', '(no ticket search yet)')}\n\n"
        f"Planned itinerary:\n{state.itinerary or '(no itinerary yet)'}\n\n"
        "Provide the itemized budget breakdown."
    )
    content, trace = _run_agent(BUDGET_SYSTEM_PROMPT, prompt)
    return {"budget_breakdown": {"summary": content}, "trace": _tag(trace, "budget_analyst")}
