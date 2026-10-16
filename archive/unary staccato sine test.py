import numpy as np

NDIM = 0x100


def ussgen(
    p_half=1,
    q=0,
    dim=NDIM,
):
    u = [1] + [0] * (p_half - 1) + [-1] + [0] * (p_half - 1)
    if q:
        u = u[-q:] + u[:-q]
    return np.array([u[i % len(u)] for i in range(dim)])


arrays = [ussgen(NDIM // 2, q) for q in range(NDIM)]

for i in range(len(arrays)):
    # t = arrays[i].dot(arrays[i])
    # assert t == arrays[i].shape[0] // 2, f"not normalized: [{i}] -> {t}"
    for j in range(i + 1, len(arrays)):
        t = arrays[i].dot(arrays[j])
        assert not t, f"not orthogonal: [{i}].[{j}]={t}"
