from datetime import timedelta

from pcomp.emd.extraction import extract_service_time_traces, extract_traces
from pcomp.utils import ensure_start_timestamp_column


def test_service_time_trace_extraction_simple_event_log(simple_event_log):
    internal_log = ensure_start_timestamp_column(simple_event_log)

    service_time_traces = extract_service_time_traces(
        internal_log,
        activity_key="concept:name",
        start_time_key="start_timestamp",
        end_time_key="time:timestamp",
    )

    day = timedelta(days=1).total_seconds()
    expected = [(("a", 0 * day), ("b", 1 * day), ("c", 1 * day), ("d", 0 * day))]
    assert service_time_traces == expected


def test_service_time_trace_extraction_event_log(event_log):
    internal_log = ensure_start_timestamp_column(event_log)

    service_time_traces = extract_service_time_traces(
        internal_log,
        activity_key="concept:name",
        start_time_key="start_timestamp",
        end_time_key="time:timestamp",
    )

    day = timedelta(days=1).total_seconds()

    # If we order by completion timestamp, we get b, a_2, a_1 for the second case
    expected = [
        (("a", 2 * day), ("c", 1 * day), ("b", 4 * day), ("d", 0 * day)),
        (("b", 2 * day), ("a", 2 * day), ("a", 5 * day)),
    ]
    assert service_time_traces == expected


def test_control_flow_extraction_simple_event_log(simple_event_log):
    # Filters to only retain complete events
    traces = extract_traces(simple_event_log, filter_complete_lifecycle=True)
    expected = [("a", "b", "c", "d")]
    assert traces == expected


def test_control_flow_extraction_event_log(event_log):
    # Filters to only retain complete events
    traces = extract_traces(event_log, filter_complete_lifecycle=True)
    expected = [
        ("a", "c", "b", "d"),
        ("b", "a", "a"),
    ]
    assert traces == expected
