from .test_lifecycle_log_conversion import event_log
from pcomp.utils import log_len


def test_log_len(event_log):
    assert log_len(event_log) == 2


# def test_log_splitting(event_log):
#     """Test that log splitting is random, i.e., multiple runs return different results"""
#     assert split_log_cases(event_log, 0.5) != split_log_cases(event_log, 0.5)
