import time
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from smart_travel_agent import memory
from smart_travel_agent.config import get_llm
from smart_travel_agent.graph import build_graph
from smart_travel_agent.state import TravelPlanState

_GRAPH = build_graph()

_NODE_LABELS = {
    "destination_researcher": "🌍 Destination Brief",
    "calendar_keeper": "📅 Travel Dates",
    "tickets_scouter": "✈️ Flights",
    "activity_planner": "🗺️ Itinerary",
    "budget_analyst": "💰 Budget Breakdown",
}


def _format_update(node_name: str, update: dict) -> str:
    if node_name == "destination_researcher":
        return update["destination_brief"]
    if node_name == "calendar_keeper":
        return update["date_range"]
    if node_name == "tickets_scouter":
        return update["tickets"]["summary"]
    if node_name == "activity_planner":
        return update["itinerary"]
    if node_name == "budget_analyst":
        return update["budget_breakdown"]["summary"]
    return ""


def _format_state_snapshot(state: dict) -> dict:
    """Same shape as _format_update's results, but read from a full state dict (e.g. the
    checkpointed state from a prior run) rather than one node's partial update."""
    return {
        "destination_researcher": state.get("destination_brief") or "",
        "calendar_keeper": state.get("date_range") or "",
        "tickets_scouter": (state.get("tickets") or {}).get("summary", ""),
        "activity_planner": state.get("itinerary") or "",
        "budget_analyst": (state.get("budget_breakdown") or {}).get("summary", ""),
    }


def _render_status(statuses: dict) -> str:
    return "\n".join(f"{'✅' if done else '⏳'} {label}" for label, done in statuses.items())


def render_context(history: list[dict], user_name: str | None = None) -> str:
    """CONTEXT: the running conversation — the initial request plus each follow-up."""
    if not history:
        return "_No conversation yet — plan a trip to start._"
    display_name = user_name.strip() if user_name and user_name.strip() else "You"
    lines = ["### Conversation"]
    for turn in history:
        role_label = display_name if turn["role"] == "user" else turn["role"]
        latency = turn.get("latency")
        suffix = f" _(took {latency:.1f}s)_" if latency is not None else ""
        lines.append(f"**{role_label}:** {turn['content']}{suffix}")
    return "\n\n".join(lines)


def render_memory(user_name: str) -> str:
    """MEMORY: durable long-term facts about this traveler, across sessions."""
    if not user_name or not user_name.strip():
        return "_Enter your name to enable long-term memory (recall across sessions)._"
    facts = memory.get_memories(user_name)
    if not facts:
        return f"_No long-term memories yet for **{user_name}**. They appear here after a completed trip._"
    return f"### Long-term memory for **{user_name}**\n\n" + "\n".join(f"- {f}" for f in facts)


def render_tools(trace: list[dict]) -> str:
    """TOOLS: the live tool-call log, filtered out of the full trace."""
    lines = []
    for entry in trace:
        for tool_call in entry.get("tool_calls", []):
            lines.append(f"- **{entry.get('agent', '?')}** called `{tool_call['name']}({tool_call['args']})`")
        if entry.get("role") == "tool":
            preview = entry.get("content", "")
            if len(preview) > 300:
                preview = preview[:300] + "..."
            lines.append(f"  → **{entry.get('name', 'result')}**: {preview}")
    if not lines:
        return "_No tool calls yet._"
    return "### Tool call log\n\n" + "\n".join(lines)


_ROLE_LABELS = {
    "human": "🧑 Human",
    "ai": "🤖 AI",
    "system": "⚙️ System",
    "tool": "🔧 Tool result",
    "timing": "⏱️ Timing",
    "error": "❌ Error",
}


def render_trace(trace: list[dict]) -> str:
    """TRACE: the full raw LLM/tool message log across every agent that ran."""
    if not trace:
        return "_No trace yet — plan a trip to see the agents' raw message log._"
    lines = ["### Full message trace"]
    current_agent = None
    for entry in trace:
        agent = entry.get("agent", "?")
        if agent != current_agent:
            lines.append(f"\n---\n#### {_NODE_LABELS.get(agent, agent)}")
            current_agent = agent
        role_label = _ROLE_LABELS.get(entry.get("role"), entry.get("role", "?"))
        lines.append(f"\n**{role_label}**")
        for tool_call in entry.get("tool_calls", []):
            lines.append(f"- 🔧 calling `{tool_call['name']}({tool_call['args']})`")
        if entry.get("name"):
            lines.append(f"- _tool: `{entry['name']}`_")
        content = entry.get("content", "")
        if content:
            if len(content) > 800:
                content = content[:800] + "..."
            lines.append(f"> {content}")
    return "\n".join(lines)


# What each agent actually reads directly, plus which agents feed into which (so a change to
# an upstream agent's output correctly marks its downstream consumers dirty too, even if the
# downstream agent's own direct inputs didn't change).
_NODE_DEPENDS_ON = {
    "destination_researcher": {"destination", "interests", "origin", "preferred_break"},
    "calendar_keeper": {"destination", "num_days", "preferred_break"},
    # date_range is calendar_keeper's own OUTPUT, not an input — it's listed here (not in
    # calendar_keeper's set) so a follow-up that overrides date_range directly (see
    # _FollowUpUpdate) marks these two dirty WITHOUT marking calendar_keeper dirty, i.e. the
    # explicit date is trusted as-is rather than triggering a fresh calendar lookup.
    "tickets_scouter": {"origin", "destination", "passengers", "budget", "date_range"},
    "activity_planner": {"destination", "num_days", "budget", "interests", "date_range"},
    "budget_analyst": {"budget", "passengers", "num_days", "destination"},
}
_NODE_UPSTREAM = {
    "destination_researcher": set(),
    "calendar_keeper": set(),
    "tickets_scouter": {"calendar_keeper"},
    "activity_planner": {"destination_researcher", "calendar_keeper"},
    "budget_analyst": {"activity_planner", "tickets_scouter"},
}
_NODE_ORDER = ["destination_researcher", "calendar_keeper", "tickets_scouter", "activity_planner", "budget_analyst"]


def _compute_dirty_nodes(changed_fields: set[str]) -> set[str]:
    dirty = {node for node, deps in _NODE_DEPENDS_ON.items() if deps & changed_fields}
    for node in _NODE_ORDER:
        if any(upstream in dirty for upstream in _NODE_UPSTREAM[node]):
            dirty.add(node)
    return dirty


class _FollowUpUpdate(BaseModel):
    origin: str | None = None
    destination: str | None = None
    num_days: int | None = None
    budget: float | None = None
    passengers: int | None = None
    interests: str | None = None
    preferred_break: str | None = None
    date_range: str | None = None


_FOLLOWUP_SYSTEM_PROMPT = (
    "Extract only the trip fields the user explicitly wants to change from this follow-up "
    "message, given the current trip details. Leave every other field null — do not guess or "
    "fill in values that weren't mentioned. If the message describes a relative change (e.g. "
    "\"add 2 more passengers\", \"bump the budget up a bit\"), compute the new absolute value "
    "from the current trip details and return that absolute value, not the delta.\n\n"
    "If the user gives specific new dates directly (e.g. \"try Feb 12-14 instead\"), set "
    "date_range to a clearly formatted description of those dates (infer the year from the "
    "current trip details if not stated) — this is an explicit override that skips a fresh "
    "school-calendar lookup, which is exactly what's wanted when the user already named the "
    "dates. Also set num_days to match the number of days spanned, unless the user separately "
    "states a different day count."
)


def _parse_followup(message: str, current_state: dict) -> dict:
    context = "\n".join(
        f"{field}: {current_state[field]}" for field in _FollowUpUpdate.model_fields if field in current_state
    )
    parsed = get_llm().with_structured_output(_FollowUpUpdate).invoke(
        [
            SystemMessage(content=_FOLLOWUP_SYSTEM_PROMPT),
            HumanMessage(content=f"Current trip details:\n{context}\n\nFollow-up message: {message}"),
        ]
    )
    update = {k: v for k, v in parsed.model_dump().items() if v is not None}
    # Drop anything the model echoed back unchanged (e.g. num_days recomputed from an explicit
    # date range, landing on the same value it already was) — only real changes should mark
    # any node dirty.
    return {k: v for k, v in update.items() if current_state.get(k) != v}


def plan_trip(origin, destination, num_days, budget, passengers, interests, preferred_break, user_name, thread_id):
    start_time = time.perf_counter()
    state = TravelPlanState(
        origin=origin,
        destination=destination,
        num_days=int(num_days),
        budget=float(budget),
        passengers=int(passengers),
        interests=interests or "",
        preferred_break=preferred_break or None,
        user_name=user_name or None,
    )
    config = {"configurable": {"thread_id": thread_id}}

    outputs = dict.fromkeys(_NODE_LABELS, "")
    statuses = dict.fromkeys(_NODE_LABELS.values(), False)
    history = [{"role": "user", "content": f"Plan a {num_days}-day trip to {destination} from {origin}."}]

    def emit(status_text: str) -> tuple:
        trace = _GRAPH.get_state(config).values.get("trace", [])
        return (
            status_text,
            *outputs.values(),
            gr.skip(),  # leave the follow-up textbox alone — plan_trip never sets it, and
            # overwriting it on every yield would fight with anything typed while this streams.
            render_context(history, user_name),
            render_memory(user_name),
            render_tools(trace),
            render_trace(trace),
            history,
        )

    yield emit(_render_status(statuses))

    for chunk in _GRAPH.stream(state, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name not in _NODE_LABELS:
                continue
            if update:
                outputs[node_name] = _format_update(node_name, update)
            statuses[_NODE_LABELS[node_name]] = True
            yield emit(_render_status(statuses))

    final_state = _GRAPH.get_state(config).values
    memory.remember(user_name, final_state)
    history[-1]["latency"] = time.perf_counter() - start_time
    yield emit(_render_status(statuses))


def send_followup(message, user_name, thread_id, history):
    start_time = time.perf_counter()
    history = history or []
    empty_outputs = dict.fromkeys(_NODE_LABELS, "")
    empty_statuses = dict.fromkeys(_NODE_LABELS.values(), False)

    def emit(status_text: str, outputs: dict, followup_value: str, trace: list[dict]) -> tuple:
        return (
            status_text,
            *outputs.values(),
            followup_value,
            render_context(history, user_name),
            render_memory(user_name),
            render_tools(trace),
            render_trace(trace),
            history,
        )

    if not message or not message.strip():
        yield emit(_render_status(empty_statuses), empty_outputs, "", [])
        return

    config = {"configurable": {"thread_id": thread_id}}
    current_state = _GRAPH.get_state(config).values
    if not current_state:
        yield emit("Plan a trip first, then send follow-ups to refine it.", empty_outputs, message, [])
        return

    previous_snapshot = _format_state_snapshot(current_state)

    update = _parse_followup(message, current_state)
    if not update:
        yield emit(
            "Didn't catch a specific change — try mentioning what you'd like different "
            "(e.g. destination, days, budget).",
            dict(previous_snapshot),
            message,
            current_state.get("trace", []),
        )
        return

    history = [*history, {"role": "user", "content": message}]

    dirty = _compute_dirty_nodes(set(update.keys()))
    if "date_range" in update:
        # An explicit date override always wins for this follow-up, regardless of what else
        # changed alongside it (e.g. num_days recomputed to match the given span) — the whole
        # point of giving dates directly is skipping a fresh calendar lookup that could talk
        # the agent back out of the dates just specified.
        dirty.discard("calendar_keeper")
    update["dirty_nodes"] = list(dirty)

    # Seed from the prior result instead of blanking the screen: nodes we already know will be
    # skipped show as done immediately (with their unchanged value); only the dirty ones start
    # as pending and update live as the graph actually recomputes them.
    outputs = dict(previous_snapshot)
    statuses = {label: (name not in dirty) for name, label in _NODE_LABELS.items()}

    # This is the one point that actually clears the box (the just-submitted message). Every
    # later yield in this run uses gr.skip() for it instead of "" — repeatedly forcing it back
    # to empty would fight with the user typing their next follow-up while this one streams.
    yield emit(_render_status(statuses), outputs, "", current_state.get("trace", []))

    for chunk in _GRAPH.stream(update, config, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            if node_name not in _NODE_LABELS:
                continue
            if node_update:
                outputs[node_name] = _format_update(node_name, node_update)
            statuses[_NODE_LABELS[node_name]] = True
            trace = _GRAPH.get_state(config).values.get("trace", [])
            yield emit(_render_status(statuses), outputs, gr.skip(), trace)

    final_state = _GRAPH.get_state(config).values
    memory.remember(user_name, final_state)
    history[-1]["latency"] = time.perf_counter() - start_time
    yield emit(_render_status(statuses), outputs, gr.skip(), final_state.get("trace", []))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Smart Travel Agent") as demo:
        thread_id = gr.State(lambda: str(uuid.uuid4()))
        history_state = gr.State([])

        gr.Markdown("# Smart Travel Agent")
        gr.Markdown(
            "A multi-agent LangGraph pipeline: destination research, school-calendar-aware "
            "date finding, real flight search, day-by-day planning, and budget analysis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                user_name = gr.Textbox(
                    label="Your name", placeholder="e.g. Sarita — used to remember your preferences"
                )
                origin = gr.Textbox(label="Origin", value="San Francisco, CA")
                destination = gr.Textbox(label="Destination", value="Tokyo, Japan")
                num_days = gr.Number(label="Number of days", value=5, precision=0)
                budget = gr.Number(label="Total budget ($)", value=5000)
                passengers = gr.Number(label="Passengers", value=1, precision=0)
                interests = gr.Textbox(label="Interests (optional)", placeholder="food, culture, history")
                preferred_break = gr.Textbox(
                    label="Preferred school break (optional)", placeholder="e.g. Thanksgiving break"
                )
                submit = gr.Button("Plan my trip", variant="primary")
                status = gr.Textbox(label="Progress", interactive=False)

                gr.Markdown("---\n**Refine the plan** (after your first run):")
                followup = gr.Textbox(
                    label="Follow-up", placeholder="e.g. actually make it 7 days instead"
                )
                followup_submit = gr.Button("Send follow-up")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Trip Plan"):
                        with gr.Accordion(_NODE_LABELS["destination_researcher"], open=True):
                            destination_brief = gr.Markdown()
                        with gr.Accordion(_NODE_LABELS["calendar_keeper"], open=True):
                            date_range = gr.Markdown()
                        with gr.Accordion(_NODE_LABELS["tickets_scouter"], open=True):
                            tickets = gr.Markdown()
                        with gr.Accordion(_NODE_LABELS["activity_planner"], open=True):
                            itinerary = gr.Markdown()
                        with gr.Accordion(_NODE_LABELS["budget_analyst"], open=True):
                            budget_breakdown = gr.Markdown()

                    with gr.Tab("Context"):
                        context_display = gr.Markdown(render_context([]))

                    with gr.Tab("Memory"):
                        memory_display = gr.Markdown(render_memory(""))

                    with gr.Tab("Tools"):
                        tools_display = gr.Markdown(render_tools([]))

                    with gr.Tab("Trace"):
                        trace_display = gr.Markdown(render_trace([]))

        all_outputs = [
            status,
            destination_brief,
            date_range,
            tickets,
            itinerary,
            budget_breakdown,
            followup,
            context_display,
            memory_display,
            tools_display,
            trace_display,
            history_state,
        ]

        submit.click(
            fn=plan_trip,
            inputs=[
                origin,
                destination,
                num_days,
                budget,
                passengers,
                interests,
                preferred_break,
                user_name,
                thread_id,
            ],
            outputs=all_outputs,
        )

        followup_inputs = [followup, user_name, thread_id, history_state]
        followup_submit.click(fn=send_followup, inputs=followup_inputs, outputs=all_outputs)
        followup.submit(fn=send_followup, inputs=followup_inputs, outputs=all_outputs)

    return demo


if __name__ == "__main__":
    build_ui().launch()
