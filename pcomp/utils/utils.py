from pm4py import read_xes
from pm4py.utils import sample_cases

from pandas import DataFrame
from . import constants
def import_log(path: str, show_progress_bar: bool=False) -> DataFrame:
    return read_xes(path, show_progress_bar=show_progress_bar)

def log_len(log: DataFrame, traceid_key: str=constants.DEFAULT_TRACEID_KEY) -> int:
    return len(log[traceid_key].unique())

def split_log_cases(log: DataFrame, frac: float, traceid_key: str=constants.DEFAULT_TRACEID_KEY) -> tuple[DataFrame, DataFrame]:
    num_cases = log_len(log)
    num_sample_cases = int(num_cases * frac)

    sample1 = sample_cases(log, num_sample_cases, case_id_key=traceid_key)
    sample2 = log[~log[traceid_key].isin(sample1[traceid_key].unique())]
    
    return sample1, sample2