import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from smart_travel_agent.config import get_llm
from smart_travel_agent.mcp_tools import get_expedia_tools
from smart_travel_agent.retrieval.retriever import search_school_calendar
from smart_travel_agent.state import TravelPlanState

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
    "search_school_calendar tool's index. Use the tool to look up each school's breaks and "
    "holidays, then propose a specific date range that falls within a break at every school "
    "the family needs to accommodate. If no single range overlaps across all schools, say so "
    "explicitly and propose the closest alternative, noting which school it conflicts with."
)


def _run_agent(system_prompt: str, user_prompt: str, tools: list | None = None, model: str | None = None) -> str:
    """Runs a create_agent loop and prints the full message trajectory for visibility."""
    llm = get_llm(model=model) if model else get_llm()
    agent = create_agent(llm, tools=tools or [], system_prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    for message in result["messages"]:
        message.pretty_print()
    return result["messages"][-1].content


async def _arun_agent(system_prompt: str, user_prompt: str, tools: list | None = None, model: str | None = None) -> str:
    """Async counterpart to _run_agent — required for MCP-backed tools, whose langchain_mcp_adapters
    wrappers only implement async invocation (the underlying MCP client session is async-only), so
    agent.invoke() fails once a tool call is actually dispatched."""
    llm = get_llm(model=model) if model else get_llm()
    agent = create_agent(llm, tools=tools or [], system_prompt=system_prompt)
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_prompt)]})
    for message in result["messages"]:
        message.pretty_print()
    return result["messages"][-1].content


def destination_researcher(state: TravelPlanState) -> dict:
    """Role: Destination Researcher — gathers destination info (attractions, food, neighborhoods, practical tips)."""
    prompt = (
        f"Research {state.destination} for a {state.num_days}-day trip.\n"
        f"Traveler interests: {state.interests or 'general sightseeing'}.\n"
        "Cover: best time to visit, neighborhoods to stay in, must-see attractions, "
        "local food scene, transportation tips, and cultural customs to know."
    )
    return {"destination_brief": _run_agent(RESEARCHER_SYSTEM_PROMPT, prompt)}


def calendar_keeper(state: TravelPlanState) -> dict:
    """Role: Calendar Keeper — suggests dates that work for everyone in the family, based on school calendars."""
    prompt = (
        f"We want to take a {state.num_days}-day trip to {state.destination}. "
        "Find a school break that fits and propose a specific start and end date."
    )
    date_range = _run_agent(
        CALENDAR_SYSTEM_PROMPT,
        prompt,
        tools=[search_school_calendar],
        model="gpt-5.4",
    )
    return {"date_range": date_range}


TICKETS_SYSTEM_PROMPT = (
    "You are a flight ticket scouter. Use the available Expedia tools to search for flights "
    "for the given number of passengers within budget, preferring options with fewer "
    "layovers/hops. Report the best options you find, including price and number of stops."
)


async def _scout_tickets(state: TravelPlanState) -> str:
    tools = await get_expedia_tools()
    prompt = (
        f"Find flight tickets from {state.origin} to {state.destination} for "
        f"{state.passengers} passenger(s), during {state.date_range or 'dates to be determined'}, "
        f"within a total budget of ${state.budget}. Prefer options with fewer hops/layovers."
    )
    return await _arun_agent(TICKETS_SYSTEM_PROMPT, prompt, tools=tools, model="gpt-5.4")


def tickets_scouter(state: TravelPlanState) -> dict:
    """Role: Tickets Scouter — finds tickets for the given number of passengers, preferring fewer hops."""
    return {"tickets": {"summary": asyncio.run(_scout_tickets(state))}}


def activity_planner(state: TravelPlanState) -> dict:
    """Role: Activity Planner — creates a day-by-day itinerary, adaptable to preferences and weather."""
    prompt = (
        f"Create a {state.num_days}-day itinerary for {state.destination}.\n"
        f"Budget: ${state.budget} total. Interests: {state.interests or 'general sightseeing'}.\n"
        f"Destination research to build on:\n{state.destination_brief or '(none yet)'}\n\n"
        "Include morning/afternoon/evening activities, specific restaurant recommendations, "
        "and travel time between locations. Make it achievable and enjoyable."
    )
    return {"itinerary": _run_agent(PLANNER_SYSTEM_PROMPT, prompt)}


BUDGET_SYSTEM_PROMPT = (
    "You are a travel budget analyst. Given the ticket prices already found and the planned "
    "itinerary, produce an itemized budget breakdown (flights, lodging, food, activities, local "
    "transportation) with daily averages, and flag clearly whether the trip fits the stated "
    "total budget. Use the real ticket price if one was found instead of estimating it."
)


def budget_analyst(state: TravelPlanState) -> dict:
    """Role: Budget Analyst — estimates costs for tickets, lodging, food, and activities within budget."""
    prompt = (
        f"Total budget: ${state.budget} for {state.passengers} passenger(s), "
        f"{state.num_days} days in {state.destination}.\n\n"
        f"Ticket search results:\n{state.tickets.get('summary', '(no ticket search yet)')}\n\n"
        f"Planned itinerary:\n{state.itinerary or '(no itinerary yet)'}\n\n"
        "Provide the itemized budget breakdown."
    )
    return {"budget_breakdown": {"summary": _run_agent(BUDGET_SYSTEM_PROMPT, prompt)}}
