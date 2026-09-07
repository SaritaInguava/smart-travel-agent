from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from smart_travel_agent.agents import (
    activity_planner,
    budget_analyst,
    calendar_keeper,
    destination_researcher,
    tickets_scouter,
)
from smart_travel_agent.state import TravelPlanState


def build_graph():
    """Wires the five agents into the graph-based flow from checkpoint 5.1's Option 2:
    destination_researcher and calendar_keeper run in parallel off START; activity_planner
    joins their outputs; tickets_scouter follows calendar_keeper (it needs the date range);
    budget_analyst joins activity_planner and tickets_scouter before END.

    This does not (yet) implement the backup-date retry loop from the design doc — that
    needs calendar_keeper/tickets_scouter to return structured primary/backup ranges and a
    resolved flag instead of freeform text, which is a bigger change than wiring the graph.

    Compiled with an in-memory checkpointer (short-term memory, per checkpoint 2.1): state
    persists across invocations sharing the same thread_id in the run config, so a follow-up
    like "actually make it 7 days" can be a partial update rather than resupplying every field.
    """
    graph = StateGraph(TravelPlanState)

    graph.add_node("destination_researcher", destination_researcher)
    graph.add_node("calendar_keeper", calendar_keeper)
    graph.add_node("tickets_scouter", tickets_scouter)
    graph.add_node("activity_planner", activity_planner)
    graph.add_node("budget_analyst", budget_analyst)

    graph.add_edge(START, "destination_researcher")
    graph.add_edge(START, "calendar_keeper")
    graph.add_edge("destination_researcher", "activity_planner")
    graph.add_edge("calendar_keeper", "activity_planner")
    graph.add_edge("calendar_keeper", "tickets_scouter")
    graph.add_edge("activity_planner", "budget_analyst")
    graph.add_edge("tickets_scouter", "budget_analyst")
    graph.add_edge("budget_analyst", END)

    return graph.compile(checkpointer=InMemorySaver())
