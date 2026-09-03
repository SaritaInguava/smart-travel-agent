from unittest.mock import MagicMock, patch

from smart_travel_agent.agents import (
    activity_planner,
    budget_analyst,
    calendar_keeper,
    destination_researcher,
    tickets_scouter,
)
from smart_travel_agent.state import TravelPlanState

BASE_STATE = TravelPlanState(
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


def test_destination_researcher_returns_brief():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Tokyo brief: ...")):
        result = destination_researcher(BASE_STATE)

    assert result == {"destination_brief": "Tokyo brief: ..."}


def test_calendar_keeper_returns_date_range():
    result = calendar_keeper(BASE_STATE)
    assert "date_range" in result


def test_tickets_scouter_returns_tickets():
    result = tickets_scouter(BASE_STATE)
    assert "tickets" in result


def test_activity_planner_returns_itinerary():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Day 1: ...")):
        result = activity_planner(BASE_STATE)

    assert result == {"itinerary": "Day 1: ..."}


def test_budget_analyst_returns_breakdown():
    result = budget_analyst(BASE_STATE)
    assert "budget_breakdown" in result
