from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

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
    fake_agent.invoke.return_value = {"messages": [AIMessage(content=content)]}
    return fake_agent


def _fake_async_agent(content: str) -> MagicMock:
    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=content)]})
    return fake_agent


def _expected_trace(agent_name: str, content: str) -> list[dict]:
    return [{"role": "ai", "content": content, "agent": agent_name}]


def _assert_timing_entry(entry: dict, agent_name: str) -> None:
    """calendar_keeper/tickets_scouter append a timing entry with a non-deterministic
    duration, so its content can't be asserted exactly — just its shape."""
    assert entry["role"] == "timing"
    assert entry["agent"] == agent_name
    assert entry["content"].startswith("⏱ took")


def test_destination_researcher_returns_brief():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Tokyo brief: ...")):
        result = destination_researcher(BASE_STATE)

    assert result == {
        "destination_brief": "Tokyo brief: ...",
        "trace": _expected_trace("destination_researcher", "Tokyo brief: ..."),
    }


def test_destination_researcher_hedges_overconfident_claims():
    overconfident = "This neighborhood is completely safe. It is guaranteed you'll have a great trip."
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent(overconfident)):
        result = destination_researcher(BASE_STATE)

    assert "completely safe" not in result["destination_brief"].lower()
    assert "guaranteed" not in result["destination_brief"].lower()
    assert result["trace"][0]["role"] == "ai"
    guardrail_entries = [e for e in result["trace"] if e["role"] == "guardrail"]
    assert len(guardrail_entries) == 1
    assert guardrail_entries[0]["agent"] == "destination_researcher"
    assert "FAIL" in guardrail_entries[0]["content"] or "WARN" in guardrail_entries[0]["content"]


def test_calendar_keeper_returns_date_range():
    with patch(
        "smart_travel_agent.agents.create_agent",
        return_value=_fake_agent("Spring break: March 16-20, 2027"),
    ) as mock_create_agent:
        result = calendar_keeper(BASE_STATE)

    assert result["date_range"] == "Spring break: March 16-20, 2027"
    assert result["trace"][0] == _expected_trace("calendar_keeper", "Spring break: March 16-20, 2027")[0]
    _assert_timing_entry(result["trace"][1], "calendar_keeper")
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

    assert result["tickets"] == {"summary": "Flight XYZ, $450, 1 stop"}
    assert result["trace"][0] == _expected_trace("tickets_scouter", "Flight XYZ, $450, 1 stop")[0]
    _assert_timing_entry(result["trace"][1], "tickets_scouter")
    _, kwargs = mock_create_agent.call_args
    assert kwargs["tools"] == []


def test_tickets_scouter_degrades_gracefully_on_tool_failure():
    with (
        patch("smart_travel_agent.agents.get_expedia_tools", new=AsyncMock(return_value=[])),
        patch(
            "smart_travel_agent.agents.create_agent",
            side_effect=RuntimeError("502 Bad Gateway"),
        ),
    ):
        result = tickets_scouter(BASE_STATE)

    assert "temporarily unavailable" in result["tickets"]["summary"]
    assert "502 Bad Gateway" in result["tickets"]["summary"]
    _assert_timing_entry(result["trace"][-1], "tickets_scouter")


def test_activity_planner_returns_itinerary():
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Day 1: ...")):
        result = activity_planner(BASE_STATE)

    assert result == {
        "itinerary": "Day 1: ...",
        "trace": _expected_trace("activity_planner", "Day 1: ..."),
    }


def test_budget_analyst_returns_breakdown():
    with patch(
        "smart_travel_agent.agents.create_agent",
        return_value=_fake_agent("Flights: $2,400. Lodging: $400. Total: $2,900, within budget."),
    ):
        result = budget_analyst(BASE_STATE)

    assert result == {
        "budget_breakdown": {"summary": "Flights: $2,400. Lodging: $400. Total: $2,900, within budget."},
        "trace": _expected_trace(
            "budget_analyst", "Flights: $2,400. Lodging: $400. Total: $2,900, within budget."
        ),
    }


def test_agent_skips_when_not_in_dirty_nodes():
    state = BASE_STATE.model_copy(update={"dirty_nodes": ["calendar_keeper"]})

    with patch("smart_travel_agent.agents.create_agent") as mock_create_agent:
        result = destination_researcher(state)

    assert result == {}
    mock_create_agent.assert_not_called()


def test_agent_runs_when_in_dirty_nodes():
    state = BASE_STATE.model_copy(update={"dirty_nodes": ["destination_researcher"]})

    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Tokyo brief: ...")):
        result = destination_researcher(state)

    assert result == {
        "destination_brief": "Tokyo brief: ...",
        "trace": _expected_trace("destination_researcher", "Tokyo brief: ..."),
    }


def test_agent_runs_when_dirty_nodes_empty():
    """Empty dirty_nodes means no skip info is available (a fresh plan) — every agent runs."""
    with patch("smart_travel_agent.agents.create_agent", return_value=_fake_agent("Tokyo brief: ...")):
        result = destination_researcher(BASE_STATE)

    assert result == {
        "destination_brief": "Tokyo brief: ...",
        "trace": _expected_trace("destination_researcher", "Tokyo brief: ..."),
    }
