import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)
a = a.reshape((5, 1))
print(a)
print(np.concat([a, np.array([6, 7, 9, 9, 9]).reshape((5, 1))], axis=1))
print(
    np.concat([a, np.array([6, 7, 9, 9, 9]).reshape((5, 1))], axis=1).flatten(order="F")
)
