from unittest.mock import patch

from smart_travel_agent.graph import build_graph
from smart_travel_agent.state import TravelPlanState


def test_graph_runs_all_agents_and_merges_state():
    with (
        patch(
            "smart_travel_agent.graph.destination_researcher",
            return_value={"destination_brief": "Tokyo brief"},
        ) as mock_research,
        patch(
            "smart_travel_agent.graph.calendar_keeper",
            return_value={"date_range": "Dec 21-25, 2026"},
        ) as mock_calendar,
        patch(
            "smart_travel_agent.graph.tickets_scouter",
            return_value={"tickets": {"summary": "Flight XYZ"}},
        ) as mock_tickets,
        patch(
            "smart_travel_agent.graph.activity_planner",
            return_value={"itinerary": "Day 1: ..."},
        ) as mock_planner,
        patch(
            "smart_travel_agent.graph.budget_analyst",
            return_value={"budget_breakdown": {"summary": "Within budget"}},
        ) as mock_budget,
    ):
        graph = build_graph()
        input_state = TravelPlanState(
            origin="San Francisco, CA",
            destination="Tokyo, Japan",
            num_days=5,
            budget=6000.0,
            passengers=3,
        )
        config = {"configurable": {"thread_id": "test-thread"}}
        result = graph.invoke(input_state, config)

    assert result["destination_brief"] == "Tokyo brief"
    assert result["date_range"] == "Dec 21-25, 2026"
    assert result["tickets"] == {"summary": "Flight XYZ"}
    assert result["itinerary"] == "Day 1: ..."
    assert result["budget_breakdown"] == {"summary": "Within budget"}

    for mock in (mock_research, mock_calendar, mock_tickets, mock_planner, mock_budget):
        mock.assert_called_once()


def test_graph_reuses_prior_state_on_same_thread():
    with (
        patch(
            "smart_travel_agent.graph.destination_researcher",
            return_value={"destination_brief": "Tokyo brief"},
        ),
        patch(
            "smart_travel_agent.graph.calendar_keeper",
            return_value={"date_range": "Dec 21-25, 2026"},
        ),
        patch(
            "smart_travel_agent.graph.tickets_scouter",
            return_value={"tickets": {"summary": "Flight XYZ"}},
        ),
        patch(
            "smart_travel_agent.graph.activity_planner",
            side_effect=lambda state: {"itinerary": f"{state.num_days}-day itinerary"},
        ),
        patch(
            "smart_travel_agent.graph.budget_analyst",
            return_value={"budget_breakdown": {"summary": "Within budget"}},
        ),
    ):
        graph = build_graph()
        config = {"configurable": {"thread_id": "refinement-thread"}}

        input_state = TravelPlanState(
            origin="San Francisco, CA",
            destination="Tokyo, Japan",
            num_days=5,
            budget=6000.0,
            passengers=3,
        )
        first = graph.invoke(input_state, config)
        assert first["itinerary"] == "5-day itinerary"

        second = graph.invoke({"num_days": 7}, config)

    assert second["origin"] == "San Francisco, CA"
    assert second["destination"] == "Tokyo, Japan"
    assert second["budget"] == 6000.0
    assert second["itinerary"] == "7-day itinerary"
