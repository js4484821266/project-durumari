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