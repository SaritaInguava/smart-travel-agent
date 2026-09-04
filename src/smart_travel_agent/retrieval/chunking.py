from collections import defaultdict
from datetime import datetime


def merge_consecutive_events(events: list[dict], max_gap_days: int = 3) -> list[dict]:
    """Merges same-title events across consecutive school days into one ranged event.

    A multi-day break like "Winter Break" is scraped as one daily entry per school day
    (weekends have no entries at all), so a Friday->Monday gap of up to 3 days is still
    treated as consecutive. Without this, a break that crosses a week boundary gets split
    across two separate weekly chunks, and the agent has to notice they're the same break.

    Tracks the open range per event title (not just the immediately preceding event),
    since a differently-titled event on the same day — e.g. "New Years Day" landing
    on the last day of "Winter Break" — would otherwise interrupt the chain.
    """
    sorted_events = sorted(events, key=lambda e: e["start"])
    open_range_by_title: dict[str, dict] = {}
    merged: list[dict] = []

    for event in sorted_events:
        open_range = open_range_by_title.get(event["summary"])
        if open_range is not None:
            gap_days = (
                datetime.fromisoformat(event["start"]).date()
                - datetime.fromisoformat(open_range["end"]).date()
            ).days
            if gap_days <= max_gap_days:
                open_range["end"] = event["end"]
                continue

        new_range = dict(event)
        open_range_by_title[event["summary"]] = new_range
        merged.append(new_range)

    return sorted(merged, key=lambda e: e["start"])


def chunk_calendar_by_week(events: list[dict]) -> list[dict]:
    """Groups events by year and ISO calendar week, per checkpoint 3.1's time-window
    chunking approach for calendar-shaped data sources."""
    weekly_chunks = defaultdict(list)

    for event in events:
        start_dt = datetime.fromisoformat(event["start"])
        year, week, _ = start_dt.isocalendar()
        weekly_chunks[(year, week)].append(event)

    formatted_chunks = []
    for (year, week), group in weekly_chunks.items():
        group.sort(key=lambda e: e["start"])

        chunk_text = f"Calendar Events for Year {year}, Week {week}:\n"
        for event in group:
            chunk_text += f"- [{event['start']}] to [{event['end']}] {event['summary']}\n"

        formatted_chunks.append(
            {
                "time_window": f"{year}-W{week:02d}",
                "start_date": group[0]["start"],
                "end_date": group[-1]["end"],
                "text": chunk_text.strip(),
                "event_count": len(group),
            }
        )

    return formatted_chunks
