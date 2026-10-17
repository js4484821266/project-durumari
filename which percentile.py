import numpy as np
from scipy.fft import dct, idct
from struct import unpack

schunk = 0x10000
print("schunk", schunk)  #
f = open("Wildlife.wmv", "rb")
p00s = []
while True:
    chunk = f.read(schunk)
    if not chunk:
        break
    else:
        while len(chunk) < schunk:
            chunk += chunk
        chunk = chunk[:schunk]
    chunk = unpack("<" + "b" * schunk, chunk)
    chunk = np.array(chunk)
    coef = dct(chunk)
    coef = np.array(coef).round().astype(np.int64)
    reconstructed = idct(coef)
    reconstructed = np.array(reconstructed).round().astype(np.int64)
    print("diffs:", (chunk != reconstructed).sum())

    l = 0
    r = schunk
    coef_sorted = coef
    coef_sorted.sort()
    while r - l > 1:
        m = (l + r) // 2
        coef_mid = coef_sorted[m]
        if (coef < coef_mid).sum() < schunk // 4:
            l = m
        else:
            r = m
    print(l)
    p00s.append(l)
print(np.mean(p00s))
