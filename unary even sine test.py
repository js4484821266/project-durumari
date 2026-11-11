import numpy as np
from math import lcm

NDIM = lcm(*list(range(1, 9)))


def uesgen(
    start,
    p_quarter=1,
    dim=NDIM,
):
    if start == 1:
        u = [1] * p_quarter + [0] * p_quarter + [-1] * p_quarter + [0] * p_quarter
    elif start == 0:
        u = [0] * p_quarter + [1] * p_quarter + [0] * p_quarter + [-1] * p_quarter
    else:
        raise ValueError
    return np.array([u[i % len(u)] for i in range(dim)])


arrays = [uesgen(1, pq) for pq in range(1, 9)]
for i in range(len(arrays)):
    # t = arrays[i].dot(arrays[i])
    # assert t == arrays[i].shape[0] // 2, f"not normalized: [{i}] -> {t}"
    for j in range(i + 1, len(arrays)):
        t = arrays[i].dot(arrays[j])
        assert not t, f"not orthogonal: [{i}] and [{j}] -> {t}"
