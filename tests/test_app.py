from smart_travel_agent.app import _compute_dirty_nodes


def test_num_days_change_skips_destination_researcher_only():
    dirty = _compute_dirty_nodes({"num_days"})

    assert "destination_researcher" not in dirty
    assert dirty == {"calendar_keeper", "tickets_scouter", "activity_planner", "budget_analyst"}


def test_destination_change_marks_everything_dirty():
    dirty = _compute_dirty_nodes({"destination"})

    assert dirty == {
        "destination_researcher",
        "calendar_keeper",
        "tickets_scouter",
        "activity_planner",
        "budget_analyst",
    }


def test_interests_change_cascades_to_planning_and_budget():
    """interests directly affects research and planning; budget_analyst cascades dirty too
    since it reads activity_planner's (now-stale) itinerary output."""
    dirty = _compute_dirty_nodes({"interests"})

    assert "calendar_keeper" not in dirty
    assert "tickets_scouter" not in dirty
    assert dirty == {"destination_researcher", "activity_planner", "budget_analyst"}


def test_budget_change_does_not_touch_calendar_or_research():
    dirty = _compute_dirty_nodes({"budget"})

    assert "calendar_keeper" not in dirty
    assert "destination_researcher" not in dirty
    assert dirty == {"tickets_scouter", "activity_planner", "budget_analyst"}
