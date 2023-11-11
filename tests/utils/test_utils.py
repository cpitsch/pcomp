from .test_lifecycle_log_conversion import large_event_log, event_log
from pcomp.utils import log_len, split_log_cases


def test_log_len(event_log):
    assert log_len(event_log) == 2


def test_log_splitting(large_event_log):
    """Test that log splitting is random, i.e., multiple runs return different results"""
    run1 = split_log_cases(large_event_log, 0.5)
    run2 = split_log_cases(large_event_log, 0.5)

    # No need to check for run1[1] and run2[1] because they are the complements of run1[0] and run2[0]
    assert not run1[0].equals(run2[0])
