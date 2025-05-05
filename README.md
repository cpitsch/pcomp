An implementation of a process hypothesis testing technique using the Earth Mover's Distance
and a permutation test.

- This is the process hypothesis testing technique proposed in my Master's Thesis "Hypothesis
Testing for Processes: Bridging Statistical Methods and Treatment Disparities in Healthcare"
    - The thesis is based on commit [9aaac79 ](https://github.com/cpitsch/pcomp/tree/9aaac7994ed0e026d63df216f56649de73a4692a) ([v0.1.0](https://github.com/cpitsch/pcomp/releases/tag/v0.1.0))
    - The experiments conducted in the thesis are in the [cpitsch/pcomp-experiments](https://github.com/cpitsch/pcomp-experiments) repository

## Examples
```py
from pm4py import read_xes

from pcomp.emd.comparators.permutation_test import (
    Timed_Levenshtein_PermutationComparator,
)

log_1 = read_xes("path/to/log_1.xes")
log_2 = read_xes("path/to/log_2.xes")
comparator = Timed_Levenshtein_PermutationComparator(
    log_1, log_2, distribution_size=10_000, seed=1337, weighted_time_cost=True
)
result = comparator.compare()
print(f"P-Value: {result.pvalue}")
result.plot().show()
```

The project also contains an implementation of the P-P-UP (Process-Process-Unknown Process)
test proposed in "Statistical Tests and Association Measures for Business Processes" by
Leemans et al.:


```py
from pm4py import read_xes

from pcomp.emd.comparators.bootstrap import ControlFlowBootstrapComparator

log_1 = read_xes("path/to/log_1.xes")
log_2 = read_xes("path/to/log_2.xes")

comparator = ControlFlowBootstrapComparator(
    log_1,
    log_2,
    bootstrapping_dist_size=10_000,
    resample_size=1.0,
    seed=1337,
)
result = comparator.compare()
print(f"P-Value: {result.pvalue}")
result.plot().show()
```

## Dependencies
This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
After installing `uv`, the dependencies can be installed using `uv sync`. This
creates a virtual environment. 

- The environment can be activated using `source .venv/bin/activate` (linux) or `.venv/Scripts/activate`
- Alternatively, commands can be run using, e.g., `uv run app.py` without activating the environment manually
- A requirements file can be generated using `uv export > requirements.txt`
