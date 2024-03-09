"""
Translated from the Javascript implementation https://github.com/sunesimonsen/ukkonen/blob/master/index.js
"""

from typing import Sequence

import numpy as np


def ukkonen_distance(
    a: Sequence,
    b: Sequence,
    threshold: int | None = None,
) -> int:
    if a == b:
        return 0

    threshold = (
        threshold
        if threshold is not None
        else len(a)
        + len(b)
        + 1  # "infinity", but the maximum possible distance should be the length of the longer sequence
    )

    # Ensure the longer sequence is b
    if len(a) > len(b):
        a, b = b, a

    len_1 = len(a)
    len_2 = len(b)

    # Perform suffix trimming:
    # We can linearly drop suffix common to both sequences since they
    # don't increase the distance at all
    while len_1 > 0 and a[len_1 - 1] == b[len_2 - 1]:
        len_1 -= 1
        len_2 -= 1

    if len_1 == 0:
        return len_2 if len_2 < threshold else threshold

    # Perform prefix trimming
    # We can linearly drop prefix common to both sequences since they
    # don't increase distance at all
    t_start = 0
    while t_start < len_1 and a[t_start] == b[t_start]:
        t_start += 1

    len_1 -= t_start
    len_2 -= t_start

    if len_1 == 0:
        return len_2 if len_2 < threshold else threshold

    threshold = len_2 if len_2 < threshold else threshold
    d_len = len_2 - len_1

    if threshold < d_len:
        return threshold

    # floor(min(threshold, aLen) / 2) + 2
    ZERO_K = (len_1 if len_1 < threshold else threshold >> 1) + 2

    arr_len = d_len + ZERO_K * 2 + 2
    current_row = np.full(arr_len, -1)
    next_row = np.full(arr_len, -1)

    chars_1 = np.array(list(a[t_start : t_start + len_1]))
    chars_2 = np.array(list(b[t_start : t_start + len_2]))

    j = 0
    condition_row = d_len + ZERO_K
    end_max = condition_row << 1  # condition_row * 2
    first_run = True
    while first_run or next_row[condition_row] < len_1 and j <= threshold:
        if first_run:
            first_run = False

        j += 1
        current_row, next_row = next_row, current_row

        previous_cell: int | None = None
        current_cell = -1
        if j <= ZERO_K:
            start = -j + 1
            next_cell = j - 2
        else:
            start = j - (ZERO_K << 1) + 1
            next_cell = current_row[ZERO_K + start]

        if j <= condition_row:
            end = j
            next_row[ZERO_K + j] = -1
        else:
            end = end_max - j

        k = start
        row_index = start + ZERO_K
        while k < end:
            previous_cell = current_cell
            current_cell = next_cell
            next_cell = current_row[row_index + 1]

            # max(t, previous_cell, next_cell + 1)
            t = current_cell + 1
            t = previous_cell if t < previous_cell else t
            t = next_cell + 1 if t < next_cell + 1 else t

            while t < len_1 and t + k < len_2 and chars_1[t] == chars_2[t + k]:
                t += 1

            next_row[row_index] = t

            k += 1
            row_index += 1

    return j - 1


if __name__ == "__main__":
    print(ukkonen_distance("Ukkonen", "Levenshtein"))
    assert ukkonen_distance("Ukkonen", "Levenshtein") == 8
    assert ukkonen_distance("Ukkonen", "Levenshtein", 6) == 6
    assert ukkonen_distance("Ukkonen", "Levenshtein", 10) == 8
