from pcomp.utils import constants, log_len, split_log_cases


def test_log_len(event_log):
    assert log_len(event_log) == 2


def test_large_log_len(large_event_log):
    assert log_len(large_event_log) == 200


def test_log_splitting(large_event_log):
    """
    Test that log splitting is random, i.e., multiple runs return different results.
    """
    run1 = split_log_cases(large_event_log, 0.5)
    run2 = split_log_cases(large_event_log, 0.5)

    # No need to check for run1[1] and run2[1] because they are the complements of
    # run1[0] and run2[0]
    assert not run1[0].equals(run2[0])


def test_seeded_log_splitting(large_event_log):
    """
    Test that seeded log splitting, i.e., repeated application yields same result.
    """
    run1 = split_log_cases(large_event_log, 0.5, seed=42)
    run2 = split_log_cases(large_event_log, 0.5, seed=42)

    # No need to check for run1[1] and run2[1] because they are the complements of
    # run1[0] and run2[0]
    assert run1[0].equals(run2[0])


def test_log_splitting_retains_all_cases(large_event_log):
    """Test that log splitting keeps all cases, and no cases are duplicated"""
    frac = 0.5

    sample1, sample2 = split_log_cases(large_event_log, frac)

    # Check that:
    # - Same number of rows (no case is duplicated)
    # - Same number of cases
    assert len(sample1) + len(sample2) == len(large_event_log)
    assert log_len(sample1) + log_len(sample2) == log_len(large_event_log)
    source_num_unique_caseids = large_event_log[constants.DEFAULT_TRACEID_KEY].nunique()
    splitted_num_unique_caseids = (
        sample1[constants.DEFAULT_TRACEID_KEY].nunique()
        + sample2[constants.DEFAULT_TRACEID_KEY].nunique()
    )
    assert source_num_unique_caseids == splitted_num_unique_caseids
