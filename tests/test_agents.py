from unittest.mock import AsyncMock, MagicMock, patch

from smart_travel_agent.agents import (
    activity_planner,
    budget_analyst,
    calendar_keeper,
    destination_researcher,
    tickets_scouter,
)
from smart_travel_agent.retrieval.retriever import search_school_calendar
from smart_travel_agent.state import TravelPlanState

BASE_STATE = TravelPlanState(
    origin="San Francisco, CA",
    destination="Tokyo, Japan",
    num_days=7,
    budget=3000.0,
    passengers=3,
    interests="food, culture, history",
)


def _fake_agent(content: str) -> MagicMock:
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"messages": [MagicMock(content=content)]}
    return fake_agent


def _fake_async_agent(content: str) -> MagicMock:
    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content=content)]})
    return fake_agent


def test_destination_researcher_returns_brief():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Tokyo brief: ...")):
        result = destination_researcher(BASE_STATE)

    assert result == {"destination_brief": "Tokyo brief: ..."}


def test_calendar_keeper_returns_date_range():
    with patch(
        "smart_travel_agent.agents.create_agent",
        return_value=_fake_agent("Spring break: March 16-20, 2027"),
    ) as mock_create_agent:
        result = calendar_keeper(BASE_STATE)

    assert result == {"date_range": "Spring break: March 16-20, 2027"}
    _, kwargs = mock_create_agent.call_args
    assert kwargs["tools"] == [search_school_calendar]


def test_tickets_scouter_returns_tickets():
    with (
        patch("smart_travel_agent.agents.get_expedia_tools", new=AsyncMock(return_value=[])),
        patch(
            "smart_travel_agent.agents.create_agent",
            return_value=_fake_async_agent("Flight XYZ, $450, 1 stop"),
        ) as mock_create_agent,
    ):
        result = tickets_scouter(BASE_STATE)

    assert result == {"tickets": {"summary": "Flight XYZ, $450, 1 stop"}}
    _, kwargs = mock_create_agent.call_args
    assert kwargs["tools"] == []


def test_activity_planner_returns_itinerary():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Day 1: ...")):
        result = activity_planner(BASE_STATE)

    assert result == {"itinerary": "Day 1: ..."}


def test_budget_analyst_returns_breakdown():
    with patch(
        "smart_travel_agent.agents.create_agent",
        return_value=_fake_agent("Flights: $2,400. Lodging: $400. Total: $2,900, within budget."),
    ):
        result = budget_analyst(BASE_STATE)

    assert result == {
        "budget_breakdown": {"summary": "Flights: $2,400. Lodging: $400. Total: $2,900, within budget."}
    }
