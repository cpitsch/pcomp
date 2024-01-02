from typing import Any, Callable, Literal, ForwardRef
import numpy as np
import pydantic
from pandas import DataFrame, Timestamp, Timedelta
from pcomp.utils import constants
from random import choice
from pm4py import write_xes
from tqdm.auto import tqdm
import re

ProcessTreeRef = ForwardRef("ProcessTree")


# Dataclasses so that the distribution functions are comparable
class NormalDistribution(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    mean: Timedelta
    std: Timedelta

    def __call__(self) -> Timedelta:
        return Timedelta(
            seconds=abs(
                np.random.normal(
                    loc=self.mean.total_seconds(), scale=self.std.total_seconds()
                )
            )
        )


# Dataclasses so that the distribution functions are comparable
class UniformDistribution(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    min: Timedelta
    max: Timedelta

    def __call__(self) -> Timedelta:
        return abs(
            Timedelta(
                seconds=np.random.uniform(
                    low=self.min.total_seconds(), high=self.max.total_seconds()
                )
            )
        )


# Dataclasses so that the distribution functions are comparable
class FixedValueDistribution(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    value: Timedelta

    def __call__(self) -> Timedelta:
        return self.value


class ProcessTree(pydantic.BaseModel):
    operator: Literal[
        "sequence", "parallel", "xor", "activity"
    ]  # Leave out loop for now
    activity: str  # For now required. If operator is not activity, then this is ignored

    get_duration: Callable[[], Timedelta]
    children: list[
        tuple[ProcessTreeRef, Callable[[], Timedelta]]
    ]  # Subtree, waiting time function

    def __str__(self) -> str:
        if self.operator == "activity":
            return self.activity
        match self.operator:
            case "sequence":
                prefix = "→"
            case "parallel":
                prefix = "∧"
            case "xor":
                prefix = "⨉"
            case _:
                prefix = ""

        return f"{prefix}({', '.join([str(child) for child, _ in self.children])})"


ProcessTree.model_rebuild()


def operator_symbol_to_name(operator: str) -> Literal["sequence", "parallel", "xor"]:
    match operator:
        case "→":
            return "sequence"
        case "∧":
            return "parallel"
        case "⨉":
            return "xor"
        case _:
            raise ValueError(f"Unknown operator {operator}")


def simulate_process_tree(
    process_tree: ProcessTree, case_id: int, base_timestamp: Timestamp
) -> list[dict[str, Any]]:
    if process_tree.operator == "activity":
        start_event: dict[str, Any] = {
            constants.DEFAULT_TRACEID_KEY: str(case_id),
            constants.DEFAULT_NAME_KEY: process_tree.activity,
            constants.DEFAULT_LIFECYCLE_KEY: "start",
            constants.DEFAULT_TIMESTAMP_KEY: base_timestamp,
        }
        complete_event: dict[str, Any] = {
            constants.DEFAULT_TRACEID_KEY: str(case_id),
            constants.DEFAULT_NAME_KEY: process_tree.activity,
            constants.DEFAULT_LIFECYCLE_KEY: "complete",
            constants.DEFAULT_TIMESTAMP_KEY: base_timestamp
            + process_tree.get_duration(),
        }

        return [start_event, complete_event]

    elif process_tree.operator == "sequence":
        events: list[dict[str, Any]] = []
        current_timestamp = base_timestamp
        for child, waiting_time in process_tree.children:
            events += simulate_process_tree(
                child, case_id, waiting_time() + current_timestamp
            )
            current_timestamp = max(
                event[constants.DEFAULT_TIMESTAMP_KEY] for event in events
            )

        return events

    elif process_tree.operator == "parallel":
        events: list[dict[str, Any]] = []
        for child, waiting_time in process_tree.children:
            events += simulate_process_tree(
                child, case_id, base_timestamp + waiting_time()
            )

        return events

    elif process_tree.operator == "xor":
        # Choose random child

        child, waiting_time = choice(process_tree.children)
        return simulate_process_tree(child, case_id, base_timestamp + waiting_time())

    else:
        raise ValueError(f"Unknown operator {process_tree.operator}")


def simulate_process_tree_event_log(
    process_tree: ProcessTree,
    num_cases: int,
    inter_case_arrival_time: Timedelta,
    initial_case_id: int = 0,
) -> DataFrame:
    events: list[dict[str, Any]] = []
    current_timestamp = Timestamp.now()
    for case_id in tqdm(
        range(initial_case_id, initial_case_id + num_cases),
        desc="simulating, cases completed :",
    ):
        events += simulate_process_tree(process_tree, case_id, current_timestamp)
        current_timestamp += inter_case_arrival_time

    return DataFrame(events).sort_values(
        by=constants.DEFAULT_TIMESTAMP_KEY, ascending=True, inplace=False
    )


def normal_distribution(mean: Timedelta, std: Timedelta) -> Callable[[], Timedelta]:
    return NormalDistribution(mean=mean, std=std)


def normal_distribution_h(
    mean_hours: float, std_hours: float
) -> Callable[[], Timedelta]:
    return normal_distribution(Timedelta(hours=mean_hours), Timedelta(hours=std_hours))


def uniform_distribution(min: Timedelta, max: Timedelta) -> Callable[[], Timedelta]:
    return UniformDistribution(min=min, max=max)


def uniform_distribution_h(
    min_hours: float, max_hours: float
) -> Callable[[], Timedelta]:
    return uniform_distribution(Timedelta(hours=min_hours), Timedelta(hours=max_hours))


def no_time_needed() -> Timedelta:
    """Just to make sure that the get_duration function isn't used when it shouldn't be"""
    raise NotImplementedError("No time needed")


def _split_child_match(input_str: str) -> list[str]:
    result = []
    current = ""
    stack = 0

    for char in input_str:
        if char == "," and stack == 0:
            result.append(current.strip())
            current = ""
        else:
            current += char
            if char == "(":
                stack += 1
            elif char == ")":
                stack = max(0, stack - 1)

    result.append(current.strip())
    return result


def create_process_tree(
    representation: str,
    duration_distributions: dict[str, Callable[[], Timedelta]],
) -> ProcessTree:
    regex_match = re.match("(?P<operator>[→∧⨉])\((?P<contents>.*)\)", representation)
    if regex_match is not None:
        operator = regex_match.group("operator")
        contents = regex_match.group("contents")

        children = _split_child_match(contents)
        return ProcessTree(
            operator=operator_symbol_to_name(operator),
            children=[
                (
                    create_process_tree(child, duration_distributions),
                    FixedValueDistribution(value=Timedelta(hours=0)),
                )
                for child in children
            ],
            activity="",
            get_duration=duration_distributions.get("", no_time_needed),
        )
    else:  # It is an activity
        return ProcessTree(
            operator="activity",
            activity=representation,
            children=[],
            get_duration=duration_distributions.get(representation, no_time_needed),
        )


if __name__ == "__main__":
    duration_distributions = {
        "a": normal_distribution_h(2, 1),
        "b": normal_distribution_h(3, 1),
        "c": normal_distribution_h(1, 0.5),
        "d": normal_distribution_h(3, 1),
        "e": uniform_distribution_h(0.5, 1),
        "f": normal_distribution_h(4, 2),
        "g": normal_distribution_h(3, 0.5),
    }

    # duration_distributions = {char: lambda: Timedelta(hours=1) for char in "abcdefg"}

    control_flow_repr = "→(a, ∧(→(b, c), f, →(d, e)), g)"

    process_tree = create_process_tree(control_flow_repr, duration_distributions)

    # Make f take longer
    duration_distributions_2 = {
        "a": normal_distribution_h(2, 1),
        "b": normal_distribution_h(3, 1),
        "c": normal_distribution_h(1, 0.5),
        "d": normal_distribution_h(3, 1),
        "e": uniform_distribution_h(0.5, 1),
        "f": uniform_distribution_h(
            6, 9
        ),  # f takes longer, and now a uniform distribution
        "g": normal_distribution_h(3, 0.5),
    }

    process_tree_2 = create_process_tree(control_flow_repr, duration_distributions)
    # print("Simulating Process Tree", process_tree)

    log_1 = simulate_process_tree_event_log(process_tree, 1000, Timedelta(days=1))
    log_2 = simulate_process_tree_event_log(process_tree_2, 1000, Timedelta(days=1))

    # Lower the timestamp precision down to seconds, as otherwise the timestamps are saved with
    # Too many decimal places, which makes invalid ISO 8601 timestamps, which yields NaT (not a timestamp)
    # When loading the log in pm4py

    log_1[constants.DEFAULT_TIMESTAMP_KEY] = log_1[
        constants.DEFAULT_TIMESTAMP_KEY
    ].dt.floor("s")

    log_2[constants.DEFAULT_TIMESTAMP_KEY] = log_2[
        constants.DEFAULT_TIMESTAMP_KEY
    ].dt.floor("s")

    write_xes(log_1, "Testing Logs/simulated_log_1.xes")
    write_xes(log_2, "Testing Logs/simulated_log_2.xes")
