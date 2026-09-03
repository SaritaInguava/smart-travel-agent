from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from smart_travel_agent.config import get_llm
from smart_travel_agent.state import TravelPlanState

_RESEARCHER_SYSTEM_PROMPT = (
    "You are an expert travel journalist who has visited 100+ countries. "
    "You know the best hidden gems and practical tips for any destination."
)

_PLANNER_SYSTEM_PROMPT = (
    "You are a luxury travel consultant with 15 years of experience crafting "
    "personalized day-by-day itineraries."
)


def _run_agent(system_prompt: str, user_prompt: str) -> str:
    """Runs a tool-less create_agent loop and prints the full message trajectory for visibility."""
    agent = create_agent(get_llm(), tools=[], system_prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
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
    return {"destination_brief": _run_agent(_RESEARCHER_SYSTEM_PROMPT, prompt)}


def calendar_keeper(state: TravelPlanState) -> dict:
    """Role: Calendar Keeper — suggests dates that work for everyone in the family, based on school calendars."""
    return {"date_range": "Date suggestion not yet implemented."}


def tickets_scouter(state: TravelPlanState) -> dict:
    """Role: Tickets Scouter — finds tickets for the given number of passengers, preferring fewer hops."""
    return {"tickets": {}}


def activity_planner(state: TravelPlanState) -> dict:
    """Role: Activity Planner — creates a day-by-day itinerary, adaptable to preferences and weather."""
    prompt = (
        f"Create a {state.num_days}-day itinerary for {state.destination}.\n"
        f"Budget: ${state.budget} total. Interests: {state.interests or 'general sightseeing'}.\n"
        f"Destination research to build on:\n{state.destination_brief or '(none yet)'}\n\n"
        "Include morning/afternoon/evening activities, specific restaurant recommendations, "
        "and travel time between locations. Make it achievable and enjoyable."
    )
    return {"itinerary": _run_agent(_PLANNER_SYSTEM_PROMPT, prompt)}


def budget_analyst(state: TravelPlanState) -> dict:
    """Role: Budget Analyst — estimates costs for tickets, lodging, food, and activities within budget."""
    return {"budget_breakdown": {}}
