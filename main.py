import numpy as np
from sympy import isprime, nextprime, primerange

# primes
P = nextprime(1 << 8)

# Vandermonde
V = np.array([[pow(a, n, P) for n in range(P - 1)] for a in range(1, P)])

# Analysis
A = np.array([[-pow(a, -n, P) % P for n in range(P - 1)] for a in range(1, P)])

# Assertion
test_vector = np.random.randint(0, P - 1, (P - 1,))
assert np.array_equal(A @ (V.T @ test_vector) % P, test_vector)