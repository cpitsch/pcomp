from pm4py import read_xes
from pandas import DataFrame

def import_log(path: str) -> DataFrame:
    return read_xes(path, show_progress_bar=False)
