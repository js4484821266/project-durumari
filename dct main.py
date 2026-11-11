import numpy as np
from scipy.fft import dct, idct
from struct import unpack

schunk = 0x10000
print("schunk", schunk)  #
f = open("20231108_181233.jpg", "rb")
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
print(np.mean(p00s))
