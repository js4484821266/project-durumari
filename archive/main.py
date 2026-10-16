import numpy as np
from sympy import isprime, nextprime, primerange

# primes
P = nextprime(1 << 8)

# Vandermonde
V = np.array([[pow(a, n, P) for n in range(P - 1)] for a in range(1, P)])

# Analysis
A = np.array([[-pow(a, -n, P) % P for n in range(P - 1)] for a in range(1, P)])

# Test File
PATH = "./Wildlife.wmv"

# Open Test File
f = open(PATH, "rb")
test_vector = np.frombuffer(f.read(P - 1), dtype=np.int8)
test_vector = np.array(test_vector, dtype=np.int64) % 0x100
print(test_vector)
f.close()

# Assertion
assert np.array_equal(V.T @ (A @ test_vector) % P, test_vector)
print("V.T @ (A @ test_vector) % P == test_vector")
assert np.array_equal(A @ (V.T @ test_vector) % P, test_vector)
print("A @ (V.T @ test_vector) % P == test_vector")

# Find Fourier1 with minimum Huffman-coded residual cost
import heapq

def huffman_bits(values):
    counts = np.bincount(values, minlength=P)
    heap = [int(x) for x in counts if x > 0]

    if len(heap) <= 1:
        return 0

    heapq.heapify(heap)

    total_bits = 0

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)

        merged = a + b
        total_bits += merged
        heapq.heappush(heap, merged)

    return total_bits


best = None

# a=1은 constant와 같으므로 제외
for basis_idx in range(1, P - 1):
    wave = V[basis_idx]

    # c=0은 zero model이므로 제외
    for coefficient in range(1, P):

        predicted = (coefficient * wave) % P
        residual = (test_vector - predicted) % P

        bits = huffman_bits(residual)

        if best is None or bits < best[0]:
            best = (
                bits,
                basis_idx,
                coefficient,
                residual.copy()
            )


bits, basis_idx, coefficient, residual = best

print("\nBest Fourier1 by Huffman residual")
print("a           :", basis_idx + 1)
print("coefficient :", coefficient)
print("Huffman bits:", bits)
print("Huffman bytes:", (bits + 7) // 8)
print("unique residuals:", len(np.unique(residual)))
print("residual:")
print(residual)

raw_huffman_bits = huffman_bits(test_vector)

print("\nComparison")
print("original Huffman bits :", raw_huffman_bits)
print("Fourier residual bits :", bits)
print("difference            :", raw_huffman_bits - bits)