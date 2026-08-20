t = n = 6
while t != 1:
    print(t, end=", ")
    t = t * n % 251
print(t)
